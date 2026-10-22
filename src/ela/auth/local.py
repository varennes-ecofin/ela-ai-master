"""Local password provider: accounts of the ``users`` table (bcrypt)."""

from __future__ import annotations

from ela.auth.base import Credentials, Identity
from ela.db import users as users_db


class LocalPasswordProvider:
    """Authenticate against the ``users`` table through :func:`ela.db.users.authenticate`.

    Inactive accounts and accounts without a password are rejected there;
    a successful login records ``last_login``.
    """

    name = "local"

    async def authenticate(self, credentials: Credentials) -> Identity | None:
        """Check the identifier/password pair.

        Args:
            credentials: ``identifier`` and ``password`` are required.

        Returns:
            The identity on success, ``None`` on wrong password, unknown or inactive account.
        """
        if not credentials.identifier or not credentials.password:
            return None
        profile = await users_db.authenticate(credentials.identifier.strip(), credentials.password)
        if profile is None:
            return None
        return Identity(
            identifier=profile.identifier,
            provider=self.name,
            attributes={"role": profile.role, "level": profile.level},
        )
