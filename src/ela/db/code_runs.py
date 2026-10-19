"""Code runs: persistence and daily quota (table ``code_runs``). Writes never raise."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ela.db.engine import execute, fetch_one

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CodeRun:
    """One executed script."""

    user_identifier: str
    status: str  # ok | error | timeout | unavailable
    origin: str = "workshop"  # workshop | agent | fix
    topic: str | None = None
    language: str = "python"
    elapsed_ms: int = 0
    n_images: int = 0
    error: str | None = None
    thread_id: str | None = None


async def record_run(run: CodeRun) -> bool:
    """Store one run (returns ``False`` instead of raising on failure)."""
    try:
        await execute(
            """
            INSERT INTO code_runs
                (user_identifier, thread_id, origin, topic, language, status, elapsed_ms, n_images, error)
            VALUES (:user, :thread, :origin, :topic, :language, :status, :elapsed, :images, :error)
            """,
            {
                "user": run.user_identifier,
                "thread": run.thread_id,
                "origin": run.origin,
                "topic": run.topic,
                "language": run.language,
                "status": run.status,
                "elapsed": run.elapsed_ms,
                "images": run.n_images,
                "error": (run.error or "")[:2000] or None,
            },
        )
        return True
    except Exception:
        logger.warning("Could not record code run for %s", run.user_identifier, exc_info=True)
        return False


async def runs_today(user_identifier: str) -> int:
    """Count the executions of a student today (UTC day); ``0`` when the database is unreachable."""
    try:
        row = await fetch_one(
            """
            SELECT count(*) AS n FROM code_runs
            WHERE user_identifier = :user AND created_at >= date_trunc('day', now())
              AND status IN ('ok', 'error', 'timeout')
            """,
            {"user": user_identifier},
        )
    except Exception:
        logger.warning("Could not count code runs for %s", user_identifier, exc_info=True)
        return 0
    return int(row["n"]) if row else 0
