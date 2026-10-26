"""Backoffice audit log (table ``admin_audit``).

Every administrative action is recorded with its actor, so that several
teachers can share the backoffice with a usable trail. Writes never
raise: an unavailable audit table must not block the action itself.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ela.db.engine import execute, fetch_all

logger = logging.getLogger(__name__)

# Actions recorded by lot 5.1; later lots add their own (``user.create``, ``ingest.run``...).
LOGIN_OK = "login.ok"
LOGIN_FAILED = "login.failed"
LOGIN_LOCKED = "login.locked"
LOGOUT = "logout"
# Lot 5.2 — accounts
USER_CREATE = "user.create"
USER_UPDATE = "user.update"
USER_PASSWORD_RESET = "user.password_reset"
USER_ACTIVATE = "user.activate"
USER_DEACTIVATE = "user.deactivate"
USER_PURGE = "user.purge"
USERS_IMPORT = "users.import"
ACCOUNT_PASSWORD = "account.password"
# Lot 5.3 — courses and ingestion
COURSE_CREATE = "course.create"
COURSE_UPLOAD = "course.upload"
COURSE_DELETE_FILE = "course.delete_file"
COURSE_DELETE = "course.delete"
COURSE_MOVE = "course.move"
INGEST_START = "ingest.start"


async def record(
    action: str,
    actor: str,
    *,
    target: str | None = None,
    details: dict[str, Any] | None = None,
    ip: str | None = None,
) -> bool:
    """Append one audit row.

    Args:
        action: Dotted action name (``login.ok``, ``user.create``...).
        actor: Identifier of the person acting (the attempted identifier for failed logins).
        target: Identifier of the object acted upon, if any.
        details: Free-form JSON payload (never passwords).
        ip: Client address as seen by the application.

    Returns:
        ``True`` when the row was written, ``False`` on failure (logged).
    """
    try:
        await execute(
            """
            INSERT INTO admin_audit (actor, action, target, details, ip)
            VALUES (:actor, :action, :target, CAST(:details AS jsonb), :ip)
            """,
            {
                "actor": actor[:200],
                "action": action,
                "target": target,
                "details": json.dumps(details or {}, ensure_ascii=False, default=str),
                "ip": ip,
            },
        )
        return True
    except Exception:
        logger.warning("Could not write audit row %s for %s", action, actor, exc_info=True)
        return False


async def recent(limit: int = 50, actor: str | None = None) -> list[dict[str, Any]]:
    """Return the latest audit rows (newest first).

    Args:
        limit: Maximum number of rows.
        actor: Restrict to one actor when given.

    Returns:
        Rows ``id``, ``at``, ``actor``, ``action``, ``target``, ``details``, ``ip``
        (empty list when the table is unavailable).
    """
    where = "WHERE actor = :actor" if actor else ""
    params: dict[str, Any] = {"limit": max(1, min(int(limit), 500))}
    if actor:
        params["actor"] = actor
    try:
        rows = await fetch_all(
            f"SELECT id, at, actor, action, target, details, ip FROM admin_audit {where} "
            "ORDER BY at DESC, id DESC LIMIT :limit",
            params,
        )
    except Exception:
        logger.warning("Could not read the audit log", exc_info=True)
        return []
    for r in rows:
        if isinstance(r.get("details"), str):
            try:
                r["details"] = json.loads(r["details"])
            except ValueError:
                pass
    return rows
