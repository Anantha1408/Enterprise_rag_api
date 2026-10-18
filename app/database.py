"""Async (API) and sync (Celery) SQLAlchemy engines, plus schema bootstrap."""

from collections.abc import AsyncIterator

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.models import Base

settings = get_settings()

# ---------------------------------------------------------------------------
# API layer — fully async
# ---------------------------------------------------------------------------
async_engine = create_async_engine(
    settings.database_url,
    echo=settings.app_env == "development",
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

# ---------------------------------------------------------------------------
# Celery worker — sync (Celery tasks are not asyncio-native by default)
# ---------------------------------------------------------------------------
sync_engine = create_engine(
    settings.database_sync_url,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    class_=Session,
    expire_on_commit=False,
    autoflush=False,
)


async def get_async_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields an AsyncSession per request."""
    async with AsyncSessionLocal() as session:
        yield session


def get_sync_session() -> Session:
    """Open a sync session for Celery / LangChain tools."""
    return SyncSessionLocal()


async def init_db() -> None:
    """Enable pgvector and create tables.

    `CREATE EXTENSION` is also in docker/init.sql for first-boot, but we
    re-run it here so existing volumes (where init.sql will not fire again)
    still get the extension before `create_all`.
    """
    async with async_engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engines() -> None:
    await async_engine.dispose()
    sync_engine.dispose()
