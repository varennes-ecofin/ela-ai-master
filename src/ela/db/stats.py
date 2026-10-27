"""Activity statistics for the backoffice (tables ``threads``, ``steps``, ``users``).

Port of the Streamlit *Activité* page: daily volume of student messages,
conversations by kind, ranking of the users, consumption of the daily quotas.
Every function returns empty data instead of raising, so that a page can
still be displayed when the database is unreachable.

Days are the UTC calendar days of the stored timestamps (``::date``), as for
the quota enforcement of the chat and the figures of the home page.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from ela.db.engine import fetch_all
from ela.db.threads import THREAD_KIND_SQL

logger = logging.getLogger(__name__)


def _int(value: Any) -> int:
    return int(value or 0)


def _user_clause(user: str | None, params: dict[str, Any]) -> str:
    if not user:
        return ""
    params["user"] = user
    return 'AND t."userIdentifier" = :user'


def period_days(start: date, end: date) -> list[date]:
    """Return every day of ``[start, end]`` (charts show 0 for the days without activity)."""
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


async def daily_messages(start: date, end: date, user: str | None = None) -> list[dict[str, Any]]:
    """Count the student messages per day.

    Args:
        start: First day (inclusive).
        end: Last day (inclusive).
        user: Restrict to one identifier.

    Returns:
        Rows ``day`` (:class:`date`) and ``messages``, days without activity omitted.
    """
    params: dict[str, Any] = {"start": start, "end": end}
    sql = f"""
        SELECT s."createdAt"::date AS day, COUNT(*) AS messages
        FROM steps s JOIN threads t ON t.id = s."threadId"
        WHERE s.type = 'user_message' AND s."createdAt"::date BETWEEN :start AND :end {_user_clause(user, params)}
        GROUP BY day ORDER BY day
    """
    try:
        rows = await fetch_all(sql, params)
    except Exception:
        logger.warning("daily_messages failed", exc_info=True)
        return []
    return [{"day": r["day"], "messages": _int(r["messages"])} for r in rows]


async def threads_by_kind(start: date, end: date, user: str | None = None) -> list[dict[str, Any]]:
    """Count the conversations created in the period, by kind (see :data:`ela.db.threads.THREAD_KINDS`)."""
    params: dict[str, Any] = {"start": start, "end": end}
    sql = f"""
        SELECT {THREAD_KIND_SQL} AS kind, COUNT(*) AS threads
        FROM threads t
        WHERE t."createdAt"::date BETWEEN :start AND :end {_user_clause(user, params)}
        GROUP BY kind ORDER BY threads DESC
    """
    try:
        rows = await fetch_all(sql, params)
    except Exception:
        logger.warning("threads_by_kind failed", exc_info=True)
        return []
    return [{"kind": r["kind"], "threads": _int(r["threads"])} for r in rows]


async def daily_threads_by_kind(start: date, end: date, user: str | None = None) -> list[dict[str, Any]]:
    """Count the conversations created per day and kind (rows ``day``, ``kind``, ``threads``)."""
    params: dict[str, Any] = {"start": start, "end": end}
    sql = f"""
        SELECT t."createdAt"::date AS day, {THREAD_KIND_SQL} AS kind, COUNT(*) AS threads
        FROM threads t
        WHERE t."createdAt"::date BETWEEN :start AND :end {_user_clause(user, params)}
        GROUP BY day, kind ORDER BY day
    """
    try:
        rows = await fetch_all(sql, params)
    except Exception:
        logger.warning("daily_threads_by_kind failed", exc_info=True)
        return []
    return [{"day": r["day"], "kind": r["kind"], "threads": _int(r["threads"])} for r in rows]


async def messages_by_user(start: date, end: date) -> list[dict[str, Any]]:
    """Rank the users by number of messages in the period.

    Returns:
        Rows ``identifier``, ``role``, ``level``, ``messages``, ``threads``, ``last_message`` (ISO text),
        most active first. Accounts that vanished from ``users`` keep their identifier with an empty role.
    """
    sql = """
        SELECT t."userIdentifier" AS identifier, u.role, u.level,
               COUNT(s.id) AS messages, COUNT(DISTINCT t.id) AS threads, MAX(s."createdAt") AS last_message
        FROM steps s
        JOIN threads t ON t.id = s."threadId"
        LEFT JOIN users u ON u.identifier = t."userIdentifier"
        WHERE s.type = 'user_message' AND s."createdAt"::date BETWEEN :start AND :end
        GROUP BY t."userIdentifier", u.role, u.level
        ORDER BY messages DESC, identifier
    """
    try:
        rows = await fetch_all(sql, {"start": start, "end": end})
    except Exception:
        logger.warning("messages_by_user failed", exc_info=True)
        return []
    return [
        {
            "identifier": r["identifier"] or "?",
            "role": r["role"] or "",
            "level": r["level"] or "",
            "messages": _int(r["messages"]),
            "threads": _int(r["threads"]),
            "last_message": (r["last_message"] or "")[:16].replace("T", " "),
        }
        for r in rows
    ]


async def quota_usage() -> list[dict[str, Any]]:
    """Return today's consumption of every active account.

    Rows ``identifier``, ``role``, ``level``, ``quota`` (``None`` when unlimited), ``used``, ``ratio`` (``used / quota``
    capped at 1) and ``exhausted``.
    """
    sql = """
        SELECT u.identifier, u.role, u.level, u.daily_quota, COALESCE(sq.used, 0) AS used
        FROM users u
        LEFT JOIN (
            SELECT t."userIdentifier" AS identifier, COUNT(s.id) AS used
            FROM steps s JOIN threads t ON t.id = s."threadId"
            WHERE s.type = 'user_message' AND s."createdAt"::date = CURRENT_DATE
            GROUP BY t."userIdentifier"
        ) sq ON sq.identifier = u.identifier
        WHERE u.is_active
        ORDER BY CASE u.role WHEN 'student' THEN 0 WHEN 'supervisor' THEN 1 ELSE 2 END, u.identifier
    """
    try:
        rows = await fetch_all(sql)
    except Exception:
        logger.warning("quota_usage failed", exc_info=True)
        return []
    out = []
    for r in rows:
        quota = r["daily_quota"]
        used = _int(r["used"])
        ratio = 0.0 if not quota else min(used / int(quota), 1.0)
        out.append(
            {
                "identifier": r["identifier"],
                "role": r["role"],
                "level": r["level"],
                "quota": None if quota is None else int(quota),
                "used": used,
                "ratio": ratio,
                "exhausted": quota is not None and used >= int(quota),
            }
        )
    return out


async def user_identifiers() -> list[str]:
    """Return the identifiers of every account (for the user filters), active ones first."""
    try:
        rows = await fetch_all("SELECT identifier FROM users ORDER BY is_active DESC, identifier")
    except Exception:
        logger.warning("user_identifiers failed", exc_info=True)
        return []
    return [r["identifier"] for r in rows]
