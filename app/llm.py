"""LLM factory: local llama.cpp GGUF, or a deterministic mock for tests."""

from __future__ import annotations

from typing import Any, Iterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.config import get_settings


class MockChatModel(BaseChatModel):
    """Drop-in ChatModel that never calls the network.

    Used when LLM_BACKEND=mock so the graph, router, and API can be exercised
    without a GGUF file. Routing decisions are keyword-based and synthesis
    cites whatever context the retriever placed on the last HumanMessage.
    """

    model_name: str = "mock-enterprise-chat"

    @property
    def _llm_type(self) -> str:
        return "mock-enterprise-chat"

    @property
    def _identifying_params(self) -> dict[str, str]:
        return {"model_name": self.model_name}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self._respond(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _respond(self, messages: list[BaseMessage]) -> str:
        blob = "\n".join(str(m.content) for m in messages).lower()
        last = str(messages[-1].content) if messages else ""

        # Router prompts ask for a single token: retrieve | direct
        if "classify the user query" in blob or "routing decision" in blob:
            retrieve_hints = (
                "document",
                "according to",
                "policy",
                "report",
                "contract",
                "what does",
                "summarize",
                "find",
                "who",
                "when",
                "where",
                "how many",
                "cite",
                "source",
                "ingested",
                "knowledge base",
                "rag",
            )
            if any(h in last.lower() for h in retrieve_hints) or "?" in last:
                return "retrieve"
            return "direct"

        context_block = ""
        if "retrieved context" in blob:
            context_block = last

        if context_block.strip() and "no documents retrieved" not in blob:
            return (
                "Based on the retrieved knowledge-base context:\n\n"
                f"{context_block.strip()[:2000]}\n\n"
                "This is a structured synthesis of the passages most similar "
                "to the query (mock LLM). Replace LLM_BACKEND with llama_cpp "
                "for generative answers."
            )
        return (
            "I can answer general questions without retrieval. "
            "Ingest documents via POST /api/v1/ingest and ask again for "
            "grounded, citation-style answers. (mock LLM)"
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        yield ChatGeneration(message=AIMessage(content=self._respond(messages)))


def get_chat_model() -> BaseChatModel:
    """Return the configured chat model (mock by default)."""
    settings = get_settings()
    if settings.llm_backend == "llama_cpp":
        if not settings.llama_model_path:
            raise RuntimeError("LLAMA_MODEL_PATH is required when LLM_BACKEND=llama_cpp.")
        try:
            from langchain_community.chat_models import ChatLlamaCpp
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "llama-cpp-python is not installed. pip install llama-cpp-python"
            ) from exc
        return ChatLlamaCpp(
            model_path=settings.llama_model_path,
            n_ctx=settings.llama_n_ctx,
            n_gpu_layers=settings.llama_n_gpu_layers,
            temperature=0.1,
            verbose=False,
        )
    return MockChatModel()
