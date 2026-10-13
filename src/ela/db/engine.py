"""Process-wide SQLAlchemy engines.

Before Phase 1 every query created and disposed its own engine. Here a
single async engine (asyncpg, pooled) is created lazily and shared by the
chat application; CLI tools use a sync engine (psycopg2).

Usage::

    from ela.db.engine import fetch_all, execute
    rows = await fetch_all("SELECT 1 AS one")
    await execute("UPDATE users SET last_login = NOW() WHERE identifier = :id", {"id": "x"})
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from ela.config import get_settings

logger = logging.getLogger(__name__)

_async_engine: AsyncEngine | None = None
_sync_engine: Engine | None = None
_lock = threading.Lock()


def get_async_engine() -> AsyncEngine:
    """Return the shared async engine, creating it on first use.

    Returns:
        A pooled ``AsyncEngine`` bound to ``DATABASE_URL`` (asyncpg driver).
    """
    global _async_engine
    if _async_engine is None:
        with _lock:
            if _async_engine is None:
                settings = get_settings()
                _async_engine = create_async_engine(
                    settings.async_database_url,
                    pool_size=settings.db_pool_size,
                    max_overflow=settings.db_max_overflow,
                    pool_pre_ping=True,
                )
                logger.info("Async DB engine created (pool_size=%d)", settings.db_pool_size)
    return _async_engine


def get_sync_engine() -> Engine:
    """Return the shared sync engine (psycopg2), for CLI tools and migrations.

    Returns:
        A pooled ``Engine`` bound to ``DATABASE_URL`` (psycopg2 driver).
    """
    global _sync_engine
    if _sync_engine is None:
        with _lock:
            if _sync_engine is None:
                settings = get_settings()
                _sync_engine = create_engine(
                    settings.sync_database_url,
                    pool_size=2,
                    max_overflow=2,
                    pool_pre_ping=True,
                    connect_args={"client_encoding": "utf8"},
                )
    return _sync_engine


async def dispose() -> None:
    """Dispose the shared engines (tests, graceful shutdown)."""
    global _async_engine, _sync_engine
    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None
    if _sync_engine is not None:
        _sync_engine.dispose()
        _sync_engine = None


async def fetch_all(sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Run a SELECT and return rows as dicts.

    Args:
        sql: SQL text with ``:name`` placeholders.
        params: Bound parameters.

    Returns:
        List of row mappings (empty on no result).
    """
    async with get_async_engine().connect() as conn:
        result = await conn.execute(text(sql), params or {})
        return [dict(r) for r in result.mappings().all()]


async def fetch_one(sql: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Run a SELECT and return the first row, or ``None``."""
    rows = await fetch_all(sql, params)
    return rows[0] if rows else None


async def execute(sql: str, params: dict[str, Any] | None = None) -> int:
    """Run an INSERT/UPDATE/DELETE inside a transaction.

    Args:
        sql: SQL text with ``:name`` placeholders.
        params: Bound parameters.

    Returns:
        Number of affected rows.
    """
    async with get_async_engine().begin() as conn:
        result = await conn.execute(text(sql), params or {})
        return result.rowcount
