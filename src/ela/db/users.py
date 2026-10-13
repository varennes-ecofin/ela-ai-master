"""User accounts: authentication, profile, daily quota.

Accounts live in Chainlit's ``users`` table, extended with
``password_hash``, ``role``, ``level``, ``is_active``, ``daily_quota`` and
``last_login`` (see ``ela.cli.users migrate``).

All functions are async and use the shared engine; bcrypt verification
runs in a worker thread so the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import bcrypt

from ela.db.engine import execute, fetch_one

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
