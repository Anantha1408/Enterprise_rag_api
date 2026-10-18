"""Pydantic request/response contracts for the public API."""

from typing import Any

from pydantic import BaseModel, Field, model_validator


class IngestRequest(BaseModel):
    """Trigger asynchronous document ingestion.

    Provide either a filesystem `file_path` the worker can read, or raw `text`
    which the API persists under UPLOAD_DIR before enqueueing Celery.
    """

    file_path: str | None = Field(default=None, description="Absolute or relative path to a document.")
    text: str | None = Field(default=None, description="Raw document text when no file is supplied.")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_source(self) -> "IngestRequest":
        if not self.file_path and not (self.text and self.text.strip()):
            raise ValueError("Provide either file_path or non-empty text.")
        return self


class IngestAcceptedResponse(BaseModel):
    task_id: str
    status: str = "accepted"
    source: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language research question.")
    top_k: int | None = Field(default=None, ge=1, le=50)


class QueryResponse(BaseModel):
    answer: str
    query: str
    context: list[str] = Field(default_factory=list)
    used_retrieval: bool = False
    route: str = "synthesizer"
