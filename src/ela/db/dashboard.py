"""Backoffice home page figures (accounts, conversations, messages, latest threads).

These are the queries of the former Streamlit ``show_home``, rewritten on the
shared async engine so that the FastAPI backoffice (and later lots) reuse
them. Every function degrades to empty figures instead of raising.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ela.db.engine import fetch_all, fetch_one

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Overview:
    """Top-level counters shown on the home page."""

    users_total: int = 0
    users_active: int = 0
    students: int = 0
    supervisors: int = 0
    admins: int = 0
    threads: int = 0
    messages_today: int = 0
    messages_yesterday: int = 0
    active_students_7d: int = 0
    rag_chunks: int | None = None
    recent_threads: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def _int(row: dict[str, Any] | None, key: str) -> int:
    return int((row or {}).get(key) or 0)


async def overview(recent_limit: int = 10) -> Overview:
    """Collect the home page figures.

    Args:
        recent_limit: Number of latest conversations to list.

    Returns:
        An :class:`Overview`; ``error`` is set (and counters are zero) when the database is unreachable.
    """
    try:
        users = await fetch_one(
            """
            SELECT COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE is_active) AS active,
                   COUNT(*) FILTER (WHERE role = 'student') AS students,
                   COUNT(*) FILTER (WHERE role = 'supervisor') AS supervisors,
                   COUNT(*) FILTER (WHERE role = 'admin') AS admins
            FROM users
            """
        )
        activity = await fetch_one(
            """
            SELECT (SELECT COUNT(*) FROM threads) AS threads,
                   (SELECT COUNT(*) FROM steps
                     WHERE type = 'user_message' AND "createdAt"::date = CURRENT_DATE) AS today,
                   (SELECT COUNT(*) FROM steps
                     WHERE type = 'user_message'
                       AND "createdAt"::date = CURRENT_DATE - INTERVAL '1 day') AS yesterday,
                   (SELECT COUNT(DISTINCT t."userIdentifier") FROM steps s
                     JOIN threads t ON t.id = s."threadId"
                     JOIN users u ON u.identifier = t."userIdentifier"
                     WHERE s.type = 'user_message' AND u.role = 'student'
                       AND s."createdAt"::date >= CURRENT_DATE - INTERVAL '6 days') AS active_7d
            """
        )
        recent = await fetch_all(
            """
            SELECT t.id, t.name, t."createdAt" AS created_at, t."userIdentifier" AS identifier,
                   (SELECT COUNT(*) FROM steps s WHERE s."threadId" = t.id AND s.type = 'user_message') AS messages
            FROM threads t
            ORDER BY t."createdAt" DESC NULLS LAST
            LIMIT :limit
            """,
            {"limit": max(1, min(int(recent_limit), 100))},
        )
    except Exception as exc:
        logger.warning("Dashboard queries failed", exc_info=True)
        return Overview(error=f"{exc.__class__.__name__}: {exc}")

    chunks: int | None = None
    try:
        row = await fetch_one("SELECT COUNT(*) AS n FROM rag_chunks")
        chunks = _int(row, "n")
    except Exception:  # Chroma backend or table absent: not an error for the home page
        logger.debug("rag_chunks unavailable", exc_info=True)

    for r in recent:
        r["id"] = str(r["id"])
        r["messages"] = int(r.get("messages") or 0)
        r["created_at"] = (r.get("created_at") or "")[:16].replace("T", " ")
    return Overview(
        users_total=_int(users, "total"),
        users_active=_int(users, "active"),
        students=_int(users, "students"),
        supervisors=_int(users, "supervisors"),
        admins=_int(users, "admins"),
        threads=_int(activity, "threads"),
        messages_today=_int(activity, "today"),
        messages_yesterday=_int(activity, "yesterday"),
        active_students_7d=_int(activity, "active_7d"),
        rag_chunks=chunks,
        recent_threads=recent,
    )
