"""Student feedback on the answers of ELA (Chainlit tables ``feedbacks``, ``steps``, ``threads``).

A feedback is a thumb up (``value = 1``) or down (``0``) on one assistant
message (``feedbacks."forId" = steps.id``), with an optional comment.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ela.db.engine import fetch_all, fetch_one

logger = logging.getLogger(__name__)

MESSAGE_TYPES = ("user_message", "assistant_message")


@dataclass(frozen=True)
class FeedbackStats:
    """Global counters shown at the top of the page."""

    total: int = 0
    positive: int = 0
    negative: int = 0

    @property
    def rate(self) -> float | None:
        """Share of positive feedback in percent, ``None`` without any feedback."""
        return round(100.0 * self.positive / self.total, 1) if self.total else None


@dataclass(frozen=True)
class FeedbackPage:
    """One page of the filtered list."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    total: int = 0


def _iso(value: Any) -> str:
    return (value or "")[:16].replace("T", " ") if isinstance(value, str) else ("" if value is None else str(value))


async def stats() -> FeedbackStats:
    """Count every feedback ever given (not restricted to a period)."""
    try:
        row = await fetch_one(
            "SELECT COUNT(*) AS total, COUNT(*) FILTER (WHERE value = 1) AS positive,"
            " COUNT(*) FILTER (WHERE value = 0) AS negative FROM feedbacks"
        )
    except Exception:
        logger.warning("feedbacks.stats failed", exc_info=True)
        return FeedbackStats()
    row = row or {}
    return FeedbackStats(int(row.get("total") or 0), int(row.get("positive") or 0), int(row.get("negative") or 0))


async def list_feedbacks(
    start: date,
    end: date,
    user: str | None = None,
    score: int | None = None,
    limit: int = 50,
    offset: int = 0,
) -> FeedbackPage:
    """List the feedback given on answers written in the period, latest first.

    Args:
        start: First day of the rated answers (inclusive).
        end: Last day (inclusive).
        user: Restrict to one student.
        score: ``1`` (positive) or ``0`` (negative); ``None`` for both.
        limit: Page size.
        offset: Rows to skip.

    Returns:
        Rows ``id``, ``score``, ``comment``, ``step_id``, ``answer`` (text of the rated message), ``answered_at``,
        ``thread_id``, ``thread_name``, ``identifier``; ``total`` is the number of matching rows.
    """
    conditions = ['s."createdAt"::date BETWEEN :start AND :end']
    params: dict[str, Any] = {
        "start": start,
        "end": end,
        "limit": max(1, min(int(limit), 500)),
        "offset": max(0, int(offset)),
    }
    if user:
        conditions.append('t."userIdentifier" = :user')
        params["user"] = user
    if score is not None:
        conditions.append("f.value = :score")
        params["score"] = int(score)
    sql = f"""
        SELECT f.id, f.value AS score, f.comment, s.id AS step_id, s.output AS answer, s."createdAt" AS answered_at,
               t.id AS thread_id, t.name AS thread_name, t."userIdentifier" AS identifier,
               COUNT(*) OVER () AS total
        FROM feedbacks f
        JOIN steps s ON s.id = f."forId"
        JOIN threads t ON t.id = s."threadId"
        WHERE {" AND ".join(conditions)}
        ORDER BY s."createdAt" DESC
        LIMIT :limit OFFSET :offset
    """
    try:
        rows = await fetch_all(sql, params)
    except Exception:
        logger.warning("feedbacks.list_feedbacks failed", exc_info=True)
        return FeedbackPage()
    total = int(rows[0]["total"]) if rows else 0
    return FeedbackPage(
        rows=[
            {
                "id": str(r["id"]),
                "score": int(r["score"]),
                "comment": r["comment"] or "",
                "step_id": str(r["step_id"]),
                "answer": r["answer"] or "",
                "answered_at": _iso(r["answered_at"]),
                "thread_id": str(r["thread_id"]),
                "thread_name": r["thread_name"] or "",
                "identifier": r["identifier"] or "?",
            }
            for r in rows
        ],
        total=total,
    )


async def context(feedback_id: str, before: int = 3, after: int = 1) -> dict[str, Any] | None:
    """Return the rated message with the exchange around it.

    Args:
        feedback_id: Feedback UUID.
        before: Messages kept before the rated one.
        after: Messages kept after it.

    Returns:
        ``None`` when the feedback (or the id) is unknown; otherwise ``feedback`` (as in :func:`list_feedbacks`)
        and ``steps`` (``id``, ``type``, ``content``, ``created_at``, ``rated``), chronological.
    """
    try:
        uuid.UUID(feedback_id)
    except ValueError:
        return None
    try:
        fb = await fetch_one(
            """
            SELECT f.id, f.value AS score, f.comment, s.id AS step_id, s."threadId" AS thread_id,
                   t.name AS thread_name, t."userIdentifier" AS identifier
            FROM feedbacks f JOIN steps s ON s.id = f."forId" JOIN threads t ON t.id = s."threadId"
            WHERE f.id = :id
            """,
            {"id": feedback_id},
        )
        if fb is None:
            return None
        steps = await fetch_all(
            """
            SELECT id, type, input, output, "createdAt" AS created_at
            FROM steps WHERE "threadId" = :thread AND type IN ('user_message', 'assistant_message')
            ORDER BY "createdAt", id
            """,
            {"thread": fb["thread_id"]},
        )
    except Exception:
        logger.warning("feedbacks.context failed", exc_info=True)
        return None
    rated = str(fb["step_id"])
    index = next((i for i, s in enumerate(steps) if str(s["id"]) == rated), None)
    window = steps if index is None else steps[max(0, index - before) : index + after + 1]
    return {
        "feedback": {
            "id": str(fb["id"]),
            "score": int(fb["score"]),
            "comment": fb["comment"] or "",
            "step_id": rated,
            "thread_id": str(fb["thread_id"]),
            "thread_name": fb["thread_name"] or "",
            "identifier": fb["identifier"] or "?",
        },
        "steps": [
            {
                "id": str(s["id"]),
                "type": s["type"],
                "content": (s["input"] if s["type"] == "user_message" and s["input"] else s["output"]) or "",
                "created_at": _iso(s["created_at"]),
                "rated": str(s["id"]) == rated,
            }
            for s in window
        ],
    }
