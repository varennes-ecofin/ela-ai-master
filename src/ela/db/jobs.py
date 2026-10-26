"""Background jobs (table ``jobs``): ingestion runs launched from the backoffice.

Reads are async (pages). Writes come from the runner thread and use the
sync engine, so that progress and log lines land in the database while the
event loop keeps serving requests.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import text

from ela.db.engine import execute, fetch_all, fetch_one, get_sync_engine

logger = logging.getLogger(__name__)

QUEUED = "queued"
RUNNING = "running"
SUCCESS = "success"
FAILED = "failed"
INTERRUPTED = "interrupted"
ACTIVE = (QUEUED, RUNNING)

KIND_INGEST = "ingest"
MAX_LOG_CHARS = 400_000


@dataclass(frozen=True)
class Job:
    """One row of ``jobs``."""

    id: int
    kind: str
    mode: str | None
    target: str | None
    status: str
    started_by: str
    created_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    progress: int
    step: str | None
    log: str
    result: dict[str, Any]
    error: str | None

    @property
    def active(self) -> bool:
        """Whether the job is queued or running."""
        return self.status in ACTIVE

    @property
    def duration_seconds(self) -> float | None:
        """Elapsed seconds between start and end (or now while running)."""
        if not self.started_at:
            return None
        end = self.finished_at or datetime.now(self.started_at.tzinfo)
        return max(0.0, (end - self.started_at).total_seconds())


_COLUMNS = (
    "id, kind, mode, target, status, started_by, created_at, started_at, finished_at, "
    "progress, step, log, result, error"
)


def _row_to_job(row: dict[str, Any]) -> Job:
    result = row.get("result") or {}
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError:
            result = {}
    return Job(
        id=int(row["id"]),
        kind=row["kind"],
        mode=row.get("mode"),
        target=row.get("target"),
        status=row["status"],
        started_by=row["started_by"],
        created_at=row.get("created_at"),
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
        progress=int(row.get("progress") or 0),
        step=row.get("step"),
        log=row.get("log") or "",
        result=result,
        error=row.get("error"),
    )


# --- async reads (pages) -------------------------------------------------------------


async def get_job(job_id: int) -> Job | None:
    """Fetch one job."""
    row = await fetch_one(f"SELECT {_COLUMNS} FROM jobs WHERE id = :id", {"id": job_id})
    return _row_to_job(row) if row else None


async def list_jobs(limit: int = 30, kind: str | None = None) -> list[Job]:
    """Latest jobs first (log omitted for the listing, kept empty)."""
    where = "WHERE kind = :kind" if kind else ""
    params: dict[str, Any] = {"limit": max(1, min(int(limit), 500))}
    if kind:
        params["kind"] = kind
    rows = await fetch_all(
        f"SELECT {_COLUMNS.replace('log, ', '')}, '' AS log FROM jobs {where} ORDER BY id DESC LIMIT :limit", params
    )
    return [_row_to_job(r) for r in rows]


async def active_job() -> Job | None:
    """Return the queued or running job, if any."""
    row = await fetch_one(f"SELECT {_COLUMNS} FROM jobs WHERE status IN ('queued', 'running') ORDER BY id LIMIT 1")
    return _row_to_job(row) if row else None


async def last_finished(kind: str = KIND_INGEST, success_only: bool = False) -> Job | None:
    """Latest finished job of a kind."""
    statuses = "('success')" if success_only else "('success', 'failed', 'interrupted')"
    row = await fetch_one(
        f"SELECT {_COLUMNS} FROM jobs WHERE kind = :kind AND status IN {statuses} ORDER BY id DESC LIMIT 1",
        {"kind": kind},
    )
    return _row_to_job(row) if row else None


async def mark_interrupted() -> int:
    """At startup: jobs left queued/running by a previous process are over."""
    try:
        return await execute(
            "UPDATE jobs SET status = 'interrupted', finished_at = now(), "
            "error = COALESCE(error, 'interrompu par un redémarrage') WHERE status IN ('queued', 'running')"
        )
    except Exception:
        logger.warning("Could not mark interrupted jobs", exc_info=True)
        return 0


# --- sync writes (runner thread) -----------------------------------------------------


def create_sync(kind: str, *, mode: str | None, target: str | None, started_by: str) -> int:
    """Insert a queued job and return its id."""
    with get_sync_engine().begin() as conn:
        row = conn.execute(
            text(
                "INSERT INTO jobs (kind, mode, target, status, started_by) "
                "VALUES (:kind, :mode, :target, 'queued', :by) RETURNING id"
            ),
            {"kind": kind, "mode": mode, "target": target, "by": started_by},
        ).one()
    return int(row[0])


def start_sync(job_id: int) -> None:
    """Mark a job running."""
    with get_sync_engine().begin() as conn:
        conn.execute(text("UPDATE jobs SET status = 'running', started_at = now() WHERE id = :id"), {"id": job_id})


def append_log_sync(job_id: int, chunk: str, *, progress: int | None = None, step: str | None = None) -> None:
    """Append log text and optionally update progress/step (log capped at ``MAX_LOG_CHARS``)."""
    sets = ["log = left(log || :chunk, :cap)"]
    params: dict[str, Any] = {"id": job_id, "chunk": chunk, "cap": MAX_LOG_CHARS}
    if progress is not None:
        sets.append("progress = :progress")
        params["progress"] = max(0, min(100, int(progress)))
    if step is not None:
        sets.append("step = :step")
        params["step"] = step[:300]
    with get_sync_engine().begin() as conn:
        conn.execute(text(f"UPDATE jobs SET {', '.join(sets)} WHERE id = :id"), params)


def finish_sync(job_id: int, *, status: str, result: dict[str, Any] | None = None, error: str | None = None) -> None:
    """Close a job."""
    with get_sync_engine().begin() as conn:
        conn.execute(
            text(
                "UPDATE jobs SET status = :status, finished_at = now(), progress = CASE WHEN :status = 'success' "
                "THEN 100 ELSE progress END, result = CAST(:result AS jsonb), error = :error WHERE id = :id"
            ),
            {
                "id": job_id,
                "status": status,
                "result": json.dumps(result or {}, ensure_ascii=False, default=str),
                "error": error,
            },
        )
