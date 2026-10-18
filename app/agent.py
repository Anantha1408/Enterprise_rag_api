"""LangGraph multi-agent research graph: Router → Retriever → Synthesizer."""

from __future__ import annotations

import logging
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from sqlalchemy import select

from app.config import get_settings
from app.database import AsyncSessionLocal, get_sync_session
from app.embeddings import embed_query
from app.llm import get_chat_model
from app.models import DocumentChunk

logger = logging.getLogger(__name__)
settings = get_settings()


class AgentState(TypedDict):
    """Shared graph state.

    `messages` uses LangGraph's `add_messages` reducer so each node can append
    without clobbering prior turns. `query` is the original user question.
    `context` accumulates retrieved chunk texts for the synthesizer.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    context: list[str]
    route: Literal["retriever", "synthesizer"]
    answer: str
    top_k: int


# ---------------------------------------------------------------------------
# Vector search tool
# ---------------------------------------------------------------------------
@tool("vector_search")
def vector_search(query: str, top_k: int = 5) -> list[dict]:
    """Embed `query` and return the top-k most similar document chunks.

    pgvector similarity search implementation
    -----------------------------------------
    SentenceTransformer vectors are L2-normalized at encode time. For unit
    vectors, cosine similarity = inner product, and pgvector's cosine
    *distance* operator `<=>` equals `1 - cosine_similarity`.

    SQLAlchemy's `Vector.cosine_distance(q)` compiles to:

        embedding <=> :query_embedding

    Ordering by that expression (ASC) therefore ranks nearest neighbors first.
    The HNSW index on `documents.embedding` (`vector_cosine_ops`) is used
    when the planner chooses an approximate index scan.

    Similarity reported to the agent is reconstructed as:

        similarity = 1 - cosine_distance
    """
    k = top_k or settings.retrieval_top_k
    query_vector = embed_query(query)

    session = get_sync_session()
    try:
        rows = session.execute(_similarity_statement(query_vector, k)).all()
    finally:
        session.close()
    hits = _rows_to_hits(rows)
    logger.info("vector_search query=%r hits=%s", query[:80], len(hits))
    return hits


def _rows_to_hits(rows) -> list[dict]:
    hits: list[dict] = []
    for row in rows:
        cosine_distance = float(row.distance)
        hits.append(
            {
                "id": str(row.id),
                "content": row.content,
                "metadata": row.chunk_metadata,
                "source_path": row.source_path,
                "similarity": 1.0 - cosine_distance,
            }
        )
    return hits


def _similarity_statement(query_vector: list[float], k: int):
    """ORDER BY embedding <=> :q  (pgvector cosine distance, ASC = nearest)."""
    distance = DocumentChunk.embedding.cosine_distance(query_vector)
    return (
        select(
            DocumentChunk.id,
            DocumentChunk.content,
            DocumentChunk.chunk_metadata,
            DocumentChunk.source_path,
            distance.label("distance"),
        )
        .order_by(distance)
        .limit(k)
    )


async def async_vector_search(query: str, top_k: int = 5) -> list[dict]:
    """Async counterpart used by the FastAPI / LangGraph retriever node."""
    k = top_k or settings.retrieval_top_k
    query_vector = embed_query(query)
    stmt = _similarity_statement(query_vector, k)
    async with AsyncSessionLocal() as session:
        result = await session.execute(stmt)
        rows = result.all()
    hits = _rows_to_hits(rows)
    logger.info("async_vector_search query=%r hits=%s", query[:80], len(hits))
    return hits


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
_ROUTER_SYSTEM = """You are the Router Agent in an enterprise RAG system.
Classify the user query. Reply with EXACTLY one token:

- retrieve  → the answer likely depends on ingested documents, policies, reports,
              or facts that should be looked up in the vector store.
- direct    → greetings, meta questions about the system, or general chat that
              does not need the knowledge base.

Routing decision: output only `retrieve` or `direct`.
"""


def router_node(state: AgentState) -> dict:
    """Node 1 — decide whether the retriever tool is required.

    LangGraph routing logic
    -----------------------
    This node does NOT call the synthesizer itself. It only writes `route`
    onto the state. `route_after_router` is registered as a *conditional edge*
    function: LangGraph inspects the returned string and jumps to either the
    `retriever` node or directly to `synthesizer` (skipping retrieval).

    Heuristic short-circuit: obvious lookup language always retrieves, so we
    do not spend an LLM call on "according to the Q3 report…".
    """
    query = state["query"]
    lowered = query.lower()
    lookup_markers = (
        "according to",
        "document",
        "policy",
        "report",
        "contract",
        "knowledge base",
        "ingested",
        "cite",
        "source",
        "find",
        "summarize",
    )
    if any(marker in lowered for marker in lookup_markers):
        decision: Literal["retriever", "synthesizer"] = "retriever"
    else:
        llm = get_chat_model()
        response = llm.invoke(
            [
                SystemMessage(content=_ROUTER_SYSTEM),
                HumanMessage(content=f"Classify the user query:\n{query}"),
            ]
        )
        token = str(response.content).strip().split()[0].lower()
        decision = "retriever" if token.startswith("retriev") else "synthesizer"

    logger.info("router_node query=%r route=%s", query[:80], decision)
    return {
        "route": decision,
        "messages": [
            AIMessage(content=f"Router selected '{decision}' for the incoming query.")
        ],
    }


def route_after_router(state: AgentState) -> Literal["retriever", "synthesizer"]:
    """Conditional-edge adapter. Must return a node name present in the graph."""
    return state.get("route") or "synthesizer"


async def retriever_node(state: AgentState) -> dict:
    """Node 2 — invoke vector search and append passages to `context`.

    Uses async SQLAlchemy so FastAPI's event loop is not blocked on Postgres.
    The `@tool` wrapper remains available for LangChain tool-calling agents.
    """
    query = state["query"]
    top_k = state.get("top_k") or settings.retrieval_top_k
    hits = await async_vector_search(query=query, top_k=top_k)

    passages: list[str] = []
    for hit in hits:
        source = hit.get("source_path", "unknown")
        sim = hit.get("similarity", 0.0)
        passages.append(f"[source={source} similarity={sim:.3f}]\n{hit['content']}")

    note = (
        f"Retriever fetched {len(passages)} chunk(s) via cosine similarity search."
        if passages
        else "Retriever found no matching documents in the vector store."
    )
    return {
        "context": passages,
        "messages": [AIMessage(content=note)],
    }


_SYNTH_SYSTEM = """You are the Synthesizer Agent for an enterprise research desk.
Write a structured, comprehensive answer to the user query.
If retrieved context is present, ground every factual claim in that context.
If context is empty, say so and answer only what you can without inventing documents.
"""


def synthesizer_node(state: AgentState) -> dict:
    """Node 3 — produce the final answer from query + optional retrieved context."""
    query = state["query"]
    context = state.get("context") or []
    context_block = (
        "\n\n---\n\n".join(context) if context else "No documents retrieved."
    )

    llm = get_chat_model()
    response = llm.invoke(
        [
            SystemMessage(content=_SYNTH_SYSTEM),
            HumanMessage(
                content=(
                    f"User query:\n{query}\n\n"
                    f"Retrieved context:\n{context_block}"
                )
            ),
        ]
    )
    answer = str(response.content).strip()
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


def build_graph():
    """Compile Router, Retriever, and Synthesizer into a StateGraph."""
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "router")
    # Conditional routing: retrieve first, or skip straight to synthesis.
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retriever": "retriever",
            "synthesizer": "synthesizer",
        },
    )
    graph.add_edge("retriever", "synthesizer")
    graph.add_edge("synthesizer", END)
    return graph.compile()


research_graph = build_graph()


async def run_research_graph(query: str, top_k: int | None = None) -> AgentState:
    """Async entrypoint used by POST /api/v1/ask."""
    initial: AgentState = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "context": [],
        "route": "synthesizer",
        "answer": "",
        "top_k": top_k or settings.retrieval_top_k,
    }
    result = await research_graph.ainvoke(initial)
    return result  # type: ignore[return-value]
