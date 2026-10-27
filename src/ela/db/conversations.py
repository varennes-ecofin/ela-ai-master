"""Read-only access to the conversations (Chainlit tables ``threads``, ``steps``, ``feedbacks``).

Port of the Streamlit *Conversations* page: filtered list of threads with
their kind (see :mod:`ela.db.threads`) and a reader giving the messages of
one thread, with the feedback attached to the answers.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ela.db.engine import fetch_all, fetch_one
from ela.db.threads import THREAD_KIND_SQL, THREAD_KINDS

logger = logging.getLogger(__name__)

MESSAGE_TYPES = ("user_message", "assistant_message")


@dataclass(frozen=True)
class ThreadPage:
    """One page of the filtered list."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    total: int = 0


def _iso(value: Any) -> str:
    return (value or "")[:16].replace("T", " ") if isinstance(value, str) else ("" if value is None else str(value))


def valid_uuid(value: str) -> bool:
    """Whether ``value`` can be used as a thread or step id."""
    try:
        uuid.UUID(value)
    except (ValueError, TypeError):
        return False
    return True


async def list_threads(
    start: date,
    end: date,
    user: str | None = None,
    kind: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> ThreadPage:
    """List the conversations created in the period, latest first.

    Args:
        start: First day (inclusive).
        end: Last day (inclusive).
        user: Restrict to one student.
        kind: One of :data:`ela.db.threads.THREAD_KINDS`; ``None`` for all.
        limit: Page size.
        offset: Rows to skip.

    Returns:
        Rows ``id``, ``name``, ``identifier``, ``created_at``, ``kind``, ``steps``, ``messages`` (student messages),
        ``feedbacks``; ``total`` is the number of matching conversations.
    """
    conditions = ['t."createdAt"::date BETWEEN :start AND :end']
    params: dict[str, Any] = {
        "start": start,
        "end": end,
        "limit": max(1, min(int(limit), 500)),
        "offset": max(0, int(offset)),
    }
    if user:
        conditions.append('t."userIdentifier" = :user')
        params["user"] = user
    if kind in THREAD_KINDS:
        conditions.append(f"{THREAD_KIND_SQL} = :kind")
        params["kind"] = kind
    sql = f"""
        SELECT t.id, t.name, t."userIdentifier" AS identifier, t."createdAt" AS created_at, {THREAD_KIND_SQL} AS kind,
               COUNT(s.id) AS steps, COUNT(s.id) FILTER (WHERE s.type = 'user_message') AS messages,
               COUNT(f.id) AS feedbacks, COUNT(*) OVER () AS total
        FROM threads t
        LEFT JOIN steps s ON s."threadId" = t.id
        LEFT JOIN feedbacks f ON f."forId" = s.id
        WHERE {" AND ".join(conditions)}
        GROUP BY t.id, t.name, t."userIdentifier", t."createdAt"
        ORDER BY t."createdAt" DESC NULLS LAST
        LIMIT :limit OFFSET :offset
    """
    try:
        rows = await fetch_all(sql, params)
    except Exception:
        logger.warning("conversations.list_threads failed", exc_info=True)
        return ThreadPage()
    return ThreadPage(
        rows=[
            {
                "id": str(r["id"]),
                "name": r["name"] or "",
                "identifier": r["identifier"] or "?",
                "created_at": _iso(r["created_at"]),
                "kind": r["kind"],
                "steps": int(r["steps"] or 0),
                "messages": int(r["messages"] or 0),
                "feedbacks": int(r["feedbacks"] or 0),
            }
            for r in rows
        ],
        total=int(rows[0]["total"]) if rows else 0,
    )


async def get_thread(thread_id: str) -> dict[str, Any] | None:
    """Return one conversation (``id``, ``name``, ``identifier``, ``created_at``, ``kind``), or ``None``."""
    if not valid_uuid(thread_id):
        return None
    try:
        row = await fetch_one(
            f'SELECT t.id, t.name, t."userIdentifier" AS identifier, t."createdAt" AS created_at, '
            f"{THREAD_KIND_SQL} AS kind FROM threads t WHERE t.id = :id",
            {"id": thread_id},
        )
    except Exception:
        logger.warning("conversations.get_thread failed", exc_info=True)
        return None
    if row is None:
        return None
    return {
        "id": str(row["id"]),
        "name": row["name"] or "",
        "identifier": row["identifier"] or "?",
        "created_at": _iso(row["created_at"]),
        "kind": row["kind"],
    }


async def thread_steps(thread_id: str, include_tools: bool = False) -> list[dict[str, Any]]:
    """Return the steps of a conversation in chronological order.

    Args:
        thread_id: Thread UUID.
        include_tools: Also return the agent's tool steps (searches, executions…), not only the messages.

    Returns:
        Rows ``id``, ``type``, ``name``, ``content`` (student input or assistant output), ``created_at``,
        ``is_error``, ``feedback`` (``None``, ``0`` or ``1``), ``feedback_comment``.
    """
    if not valid_uuid(thread_id):
        return []
    type_clause = "" if include_tools else " AND s.type IN ('user_message', 'assistant_message')"
    try:
        rows = await fetch_all(
            f"""
            SELECT s.id, s.type, s.name, s.input, s.output, s."createdAt" AS created_at, s."isError" AS is_error,
                   f.value AS feedback, f.comment AS feedback_comment
            FROM steps s LEFT JOIN feedbacks f ON f."forId" = s.id
            WHERE s."threadId" = :id{type_clause}
            ORDER BY s."createdAt", s.id
            """,
            {"id": thread_id},
        )
    except Exception:
        logger.warning("conversations.thread_steps failed", exc_info=True)
        return []
    out = []
    for r in rows:
        kind = r["type"]
        if kind == "user_message":
            content = r["input"] or r["output"] or ""
        elif kind == "assistant_message":
            content = r["output"] or ""
        else:
            content = r["output"] or r["input"] or ""
        out.append(
            {
                "id": str(r["id"]),
                "type": kind,
                "name": r["name"] or "",
                "content": content,
                "created_at": _iso(r["created_at"]),
                "is_error": bool(r["is_error"]),
                "feedback": None if r["feedback"] is None else int(r["feedback"]),
                "feedback_comment": r["feedback_comment"] or "",
            }
        )
    return out
