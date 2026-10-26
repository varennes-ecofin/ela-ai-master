"""User accounts: authentication, profile, daily quota, administration.

Accounts live in Chainlit's ``users`` table, extended with
``password_hash``, ``role``, ``level``, ``is_active``, ``daily_quota`` and
``last_login`` (see ``ela.cli.users migrate``). The second half of the
module (``Account``, ``list_accounts``, ``create_account``...) serves the
backoffice and the ``ela-users`` command line.

All functions are async and use the shared engine; bcrypt verification
runs in a worker thread so the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import bcrypt

from ela.db.engine import execute, fetch_all, fetch_one

logger = logging.getLogger(__name__)

UNFILTERED_LEVELS = frozenset({"ALL", "admin"})


@dataclass(frozen=True)
class UserProfile:
    """Account information needed by the chat application."""

    id: str
    identifier: str
    role: str
    level: str
    is_active: bool
    daily_quota: int | None

    @property
    def unlimited(self) -> bool:
        """Whether the account has no daily message quota."""
        return self.daily_quota is None


@dataclass(frozen=True)
class QuotaStatus:
    """Result of a quota check."""

    allowed: bool
    used: int
    limit: int | None


def _row_to_profile(row: dict) -> UserProfile:
    return UserProfile(
        id=str(row["id"]),
        identifier=row["identifier"],
        role=row.get("role") or "student",
        level=row.get("level") or "ALL",
        is_active=bool(row.get("is_active", True)),
        daily_quota=row.get("daily_quota"),
    )


async def get_profile(identifier: str) -> UserProfile | None:
    """Fetch an account profile.

    Args:
        identifier: Login name.

    Returns:
        The profile, or ``None`` if the account does not exist.
    """
    row = await fetch_one(
        """
        SELECT id, identifier, role, level, is_active, daily_quota
        FROM users WHERE identifier = :identifier
        """,
        {"identifier": identifier},
    )
    return _row_to_profile(row) if row else None


def _check_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:  # malformed hash
        return False


async def authenticate(identifier: str, password: str) -> UserProfile | None:
    """Verify credentials and record the login.

    Args:
        identifier: Login name.
        password: Plaintext password.

    Returns:
        The profile on success, ``None`` on unknown account, inactive
        account or wrong password.
    """
    row = await fetch_one(
        """
        SELECT id, identifier, password_hash, role, level, is_active, daily_quota
        FROM users WHERE identifier = :identifier
        """,
        {"identifier": identifier},
    )
    if not row or not row.get("is_active") or not row.get("password_hash"):
        return None

    ok = await asyncio.to_thread(_check_password, password, row["password_hash"])
    if not ok:
        return None

    await execute(
        "UPDATE users SET last_login = NOW() WHERE identifier = :identifier",
        {"identifier": identifier},
    )
    logger.info("Login: %s (role=%s, level=%s)", identifier, row.get("role"), row.get("level"))
    return _row_to_profile(row)


async def messages_today(identifier: str) -> int:
    """Count the user messages sent today by an account.

    Args:
        identifier: Login name.

    Returns:
        Number of ``user_message`` steps created today (server date).
    """
    row = await fetch_one(
        """
        SELECT COUNT(s.id) AS msg_count
        FROM steps s
        JOIN threads t ON s."threadId" = t.id
        WHERE t."userIdentifier" = :identifier
          AND s.type = 'user_message'
          AND s."createdAt"::DATE = CURRENT_DATE
        """,
        {"identifier": identifier},
    )
    return int(row["msg_count"]) if row else 0


async def check_quota(profile: UserProfile) -> QuotaStatus:
    """Check whether an account may still send messages today.

    Args:
        profile: The account profile.

    Returns:
        A :class:`QuotaStatus`; ``limit`` is ``None`` for unlimited accounts.
    """
    if profile.unlimited:
        return QuotaStatus(allowed=True, used=0, limit=None)
    used = await messages_today(profile.identifier)
    return QuotaStatus(allowed=used < profile.daily_quota, used=used, limit=profile.daily_quota)


# ---------------------------------------------------------------------------
# Account administration (backoffice, ``ela-users``) — Phase 5, lot 5.2
# ---------------------------------------------------------------------------

ROLES = ("student", "supervisor", "admin")
LEVELS = ("M1", "M2", "ALL")
IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")
_UNSET: Any = object()


class AccountExistsError(Exception):
    """Raised when creating an account whose identifier is already taken."""


class AccountNotFoundError(Exception):
    """Raised when updating an account that does not exist."""


class AccountHasActivityError(Exception):
    """Raised when purging an account that has conversations or results."""


@dataclass(frozen=True)
class Account:
    """One row of the ``users`` table as shown in the backoffice."""

    id: str
    identifier: str
    role: str
    level: str
    is_active: bool
    daily_quota: int | None
    has_password: bool
    last_login: datetime | None
    created_at: str | None
    messages_today: int = 0

    @property
    def unlimited(self) -> bool:
        """Whether the account has no daily message quota."""
        return self.daily_quota is None


def normalize_identifier(raw: str) -> str:
    """Lower-case ``raw`` and keep only the part before an ``@`` (university e-mail → login)."""
    value = (raw or "").strip().lower()
    if "@" in value:
        value = value.split("@", 1)[0]
    return value


def identifier_problem(identifier: str) -> str | None:
    """Return why ``identifier`` is not acceptable, or ``None`` when it is."""
    if not identifier:
        return "identifiant vide"
    if not IDENTIFIER_RE.match(identifier):
        return "identifiant invalide (2 à 64 caractères : lettres minuscules, chiffres, point, tiret, souligné)"
    return None


def _row_to_account(row: dict) -> Account:
    return Account(
        id=str(row["id"]),
        identifier=row["identifier"],
        role=row.get("role") or "student",
        level=row.get("level") or "ALL",
        is_active=bool(row.get("is_active", True)),
        daily_quota=row.get("daily_quota"),
        has_password=bool(row.get("has_password")),
        last_login=row.get("last_login"),
        created_at=row.get("created_at"),
        messages_today=int(row.get("messages_today") or 0),
    )


_ACCOUNT_SELECT = """
    SELECT u.id, u.identifier, u.role, u.level, u.is_active, u.daily_quota,
           u.password_hash IS NOT NULL AS has_password, u.last_login, u."createdAt" AS created_at,
           (SELECT COUNT(*) FROM steps s JOIN threads t ON t.id = s."threadId"
             WHERE t."userIdentifier" = u.identifier AND s.type = 'user_message'
               AND s."createdAt"::DATE = CURRENT_DATE) AS messages_today
    FROM users u
"""


async def list_accounts(
    *,
    search: str | None = None,
    role: str | None = None,
    level: str | None = None,
    active: bool | None = None,
) -> list[Account]:
    """List accounts, admins first then by identifier.

    Args:
        search: Case-insensitive substring of the identifier.
        role: Restrict to one role.
        level: Restrict to one level.
        active: Restrict to active (``True``) or inactive (``False``) accounts.
    """
    clauses: list[str] = []
    params: dict[str, Any] = {}
    if search:
        clauses.append("u.identifier ILIKE :search")
        params["search"] = f"%{search.strip()}%"
    if role:
        clauses.append("u.role = :role")
        params["role"] = role
    if level:
        clauses.append("u.level = :level")
        params["level"] = level
    if active is not None:
        clauses.append("u.is_active = :active")
        params["active"] = active
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = await fetch_all(
        _ACCOUNT_SELECT
        + f" {where} ORDER BY CASE u.role WHEN 'admin' THEN 0 WHEN 'supervisor' THEN 1 ELSE 2 END, u.identifier",
        params,
    )
    return [_row_to_account(r) for r in rows]


async def get_account(identifier: str) -> Account | None:
    """Fetch one account (with today's usage), or ``None``."""
    row = await fetch_one(_ACCOUNT_SELECT + " WHERE u.identifier = :identifier", {"identifier": identifier})
    return _row_to_account(row) if row else None


async def create_account(
    identifier: str,
    *,
    password_hash: str | None,
    role: str = "student",
    level: str = "M1",
    daily_quota: int | None = 50,
) -> Account:
    """Create an account.

    Args:
        identifier: Normalised, validated login name.
        password_hash: bcrypt hash, or ``None`` for an account that cannot log in yet.
        role: One of :data:`ROLES`.
        level: One of :data:`LEVELS`.
        daily_quota: Messages per day, ``None`` for unlimited (admins are always unlimited).

    Raises:
        AccountExistsError: The identifier is already taken.
        ValueError: Unknown role or level.
    """
    if role not in ROLES:
        raise ValueError(f"Unknown role {role!r}")
    if level not in LEVELS:
        raise ValueError(f"Unknown level {level!r}")
    if role == "admin":
        daily_quota = None
    if await fetch_one("SELECT 1 FROM users WHERE identifier = :i", {"i": identifier}):
        raise AccountExistsError(identifier)
    await execute(
        """
        INSERT INTO users (id, identifier, "createdAt", password_hash, role, level, is_active, daily_quota)
        VALUES (gen_random_uuid(), :identifier, NOW()::TEXT, :hash, :role, :level, TRUE, :quota)
        """,
        {"identifier": identifier, "hash": password_hash, "role": role, "level": level, "quota": daily_quota},
    )
    account = await get_account(identifier)
    assert account is not None
    return account


async def update_account(
    identifier: str,
    *,
    role: str | None = None,
    level: str | None = None,
    daily_quota: int | None = _UNSET,
    is_active: bool | None = None,
) -> Account:
    """Change the role, level, quota and/or state of an account.

    Only the given fields change; ``daily_quota=None`` means unlimited.

    Raises:
        AccountNotFoundError: Unknown identifier.
        ValueError: Unknown role or level.
    """
    fields: dict[str, Any] = {}
    if role is not None:
        if role not in ROLES:
            raise ValueError(f"Unknown role {role!r}")
        fields["role"] = role
    if level is not None:
        if level not in LEVELS:
            raise ValueError(f"Unknown level {level!r}")
        fields["level"] = level
    if daily_quota is not _UNSET:
        fields["daily_quota"] = daily_quota
    if is_active is not None:
        fields["is_active"] = is_active
    if fields:
        assignments = ", ".join(f"{k} = :{k}" for k in fields)
        changed = await execute(
            f"UPDATE users SET {assignments} WHERE identifier = :identifier", {**fields, "identifier": identifier}
        )
        if changed == 0:
            raise AccountNotFoundError(identifier)
    account = await get_account(identifier)
    if account is None:
        raise AccountNotFoundError(identifier)
    return account


async def set_password_hash(identifier: str, password_hash: str) -> None:
    """Replace the password hash of an account (and re-activate it).

    Raises:
        AccountNotFoundError: Unknown identifier.
    """
    changed = await execute(
        "UPDATE users SET password_hash = :hash, is_active = TRUE WHERE identifier = :identifier",
        {"hash": password_hash, "identifier": identifier},
    )
    if changed == 0:
        raise AccountNotFoundError(identifier)


async def get_password_hash(identifier: str) -> str | None:
    """Return the bcrypt hash of an account (``None`` when absent or unknown account)."""
    row = await fetch_one("SELECT password_hash FROM users WHERE identifier = :i", {"i": identifier})
    return row["password_hash"] if row else None


async def count_admins(active_only: bool = True) -> int:
    """Count the ``admin`` accounts (active ones by default)."""
    where = "AND is_active" if active_only else ""
    row = await fetch_one(f"SELECT COUNT(*) AS n FROM users WHERE role = 'admin' {where}")
    return int(row["n"]) if row else 0


async def activity_summary(identifier: str) -> dict[str, int]:
    """Count what an account has produced: threads, questions, quiz answers, exercise attempts, code runs.

    Tables of later phases may be absent on an old database; each count then falls back to 0.
    """
    queries = {
        "threads": 'SELECT COUNT(*) AS n FROM threads WHERE "userIdentifier" = :i',
        "questions": "SELECT COUNT(*) AS n FROM student_activity WHERE user_identifier = :i",
        "quiz_answers": "SELECT COUNT(*) AS n FROM quiz_results WHERE user_identifier = :i",
        "exercise_attempts": "SELECT COUNT(*) AS n FROM exercise_attempts WHERE user_identifier = :i",
        "code_runs": "SELECT COUNT(*) AS n FROM code_runs WHERE user_identifier = :i",
    }
    summary: dict[str, int] = {}
    for key, sql in queries.items():
        try:
            row = await fetch_one(sql, {"i": identifier})
            summary[key] = int(row["n"]) if row else 0
        except Exception:
            logger.debug("activity_summary: %s unavailable", key, exc_info=True)
            summary[key] = 0
    return summary


async def purge_account(identifier: str) -> None:
    """Delete an account that has no activity at all (created by mistake).

    Raises:
        AccountNotFoundError: Unknown identifier.
        AccountHasActivityError: The account has conversations or results; deactivate it instead.
    """
    if await get_account(identifier) is None:
        raise AccountNotFoundError(identifier)
    summary = await activity_summary(identifier)
    if any(summary.values()):
        raise AccountHasActivityError(identifier)
    await execute("DELETE FROM users WHERE identifier = :i", {"i": identifier})
