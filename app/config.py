"""Application settings loaded from environment / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration. Inject this rather than reading os.environ ad-hoc."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "enterprise-rag-api"
    app_env: str = "development"
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/enterprise_rag"
    database_sync_url: str = "postgresql+psycopg://rag:rag@localhost:5432/enterprise_rag"
    redis_url: str = "redis://localhost:6379/0"

    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    llm_backend: str = "mock"  # mock | llama_cpp
    llama_model_path: str = ""
    llama_n_ctx: int = 2048
    llama_n_gpu_layers: int = 0

    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 5
    upload_dir: str = "data/uploads"

    @property
    def upload_path(self) -> Path:
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
