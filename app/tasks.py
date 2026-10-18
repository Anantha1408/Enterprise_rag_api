"""Celery application and the asynchronous document-ingestion pipeline."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from celery import Celery
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.database import get_sync_session
from app.embeddings import embed_texts
from app.models import DocumentChunk

logger = logging.getLogger(__name__)
settings = get_settings()

celery_app = Celery(
    "enterprise_rag",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


def _load_source_text(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = PyPDFLoader(str(path)).load()
        return "\n\n".join(page.page_content for page in pages)
    # Default: treat as UTF-8 text (txt, md, csv, json, logs, …)
    loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)


@celery_app.task(name="process_document", bind=True, max_retries=3)
def process_document(self, file_path: str, metadata: dict | None = None) -> dict:
    """Chunk a document, embed each chunk, and persist rows into pgvector.

    This runs on a Celery worker so FastAPI can return 202 immediately.
    """
    metadata = metadata or {}
    logger.info("Ingesting document path=%s task_id=%s", file_path, self.request.id)

    try:
        raw_text = _load_source_text(file_path)
    except Exception as exc:
        logger.exception("Failed to load %s", file_path)
        raise self.retry(exc=exc, countdown=10)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = [c.strip() for c in splitter.split_text(raw_text) if c.strip()]
    if not chunks:
        logger.warning("No non-empty chunks produced from %s", file_path)
        return {"document_id": None, "chunks": 0, "file_path": file_path}

    embeddings = embed_texts(chunks)
    document_id = uuid.uuid4()

    session = get_sync_session()
    try:
        rows = [
            DocumentChunk(
                document_id=document_id,
                source_path=file_path,
                content=chunk,
                chunk_metadata={
                    **metadata,
                    "source": file_path,
                    "chunk_index": index,
                    "char_count": len(chunk),
                },
                chunk_index=index,
                embedding=vector,
            )
            for index, (chunk, vector) in enumerate(zip(chunks, embeddings, strict=True))
        ]
        session.add_all(rows)
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.exception("Failed to persist chunks for %s", file_path)
        raise self.retry(exc=exc, countdown=15)
    finally:
        session.close()

    logger.info("Persisted %s chunks document_id=%s", len(chunks), document_id)
    return {
        "document_id": str(document_id),
        "chunks": len(chunks),
        "file_path": file_path,
    }
