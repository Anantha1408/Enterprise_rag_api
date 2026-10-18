"""FastAPI application: ingest (202) and ask (LangGraph) endpoints."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app import __version__
from app.agent import run_research_graph
from app.config import get_settings
from app.database import dispose_engines, init_db
from app.schemas import (
    IngestAcceptedResponse,
    IngestRequest,
    QueryRequest,
    QueryResponse,
)
from app.tasks import process_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Bootstrap pgvector + tables on startup; dispose pools on shutdown."""
    await init_db()
    logger.info("Database initialized (pgvector extension + documents table).")
    yield
    await dispose_engines()


app = FastAPI(
    title="Multi-Agent Enterprise Research and RAG API",
    version=__version__,
    description=(
        "Asynchronous document ingestion with SentenceTransformers embeddings, "
        "pgvector retrieval, and a LangGraph router/retriever/synthesizer loop."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    static_file = Path(__file__).parent / "static" / "index.html"
    if static_file.exists():
        return FileResponse(static_file)
    return JSONResponse({"status": "ok", "service": settings.app_name, "version": __version__})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name, "version": __version__}


@app.post(
    "/api/v1/ingest",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IngestAcceptedResponse,
)
async def ingest_document(payload: IngestRequest) -> JSONResponse:
    """Enqueue `process_document` on Celery. The API does not block on embeddings."""
    if payload.file_path:
        source = payload.file_path
        if not Path(source).exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"file_path does not exist: {source}",
            )
    else:
        filename = f"{uuid.uuid4()}.txt"
        dest = settings.upload_path / filename
        dest.write_text(payload.text or "", encoding="utf-8")
        source = str(dest.resolve())

    async_result = process_document.delay(file_path=source, metadata=payload.metadata)
    body = IngestAcceptedResponse(
        task_id=async_result.id,
        status="accepted",
        source=source,
    )
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=body.model_dump())


@app.post("/api/v1/ask", response_model=QueryResponse)
async def ask(payload: QueryRequest) -> QueryResponse:
    """Run the compiled LangGraph application asynchronously."""
    try:
        result = await run_research_graph(payload.query, top_k=payload.top_k)
    except Exception as exc:
        logger.exception("LangGraph invocation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent graph failed: {exc}",
        ) from exc

    route = result.get("route") or "synthesizer"
    context = result.get("context") or []
    return QueryResponse(
        answer=result.get("answer") or "",
        query=payload.query,
        context=context,
        used_retrieval=route == "retriever" or bool(context),
        route=route,
    )
