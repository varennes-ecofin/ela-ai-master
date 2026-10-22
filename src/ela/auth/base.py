"""Authentication abstraction shared by the chat and the backoffice.

An :class:`AuthProvider` turns credentials into an :class:`Identity`, or
``None`` when they are rejected. The local provider checks a password
against the ``users`` table; an institutional provider (CAS at the
Université d'Orléans, later LDAP/OIDC) will validate a ticket or a token
instead and return the same :class:`Identity`, so that nothing downstream
depends on how the person was authenticated.

Identities carry the raw ``attributes`` released by the provider (CAS
attributes such as the affiliation or the study programme) so that an
account can be provisioned automatically on first login (see
:class:`ProvisioningPolicy`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Credentials:
    """What a person presents to prove who they are.

    The local provider uses ``identifier`` and ``password``; a CAS provider
    will use ``token`` (service ticket) and ``service`` (callback URL).
    """

    identifier: str | None = None
    password: str | None = None
    token: str | None = None
    service: str | None = None


@dataclass(frozen=True)
class Identity:
    """A verified person, independent of the provider that verified them.

    Attributes:
        identifier: Login name, used as ``users.identifier``.
        provider: Name of the provider that verified the identity (``local``, ``cas``...).
        display_name: Optional human-readable name.
        attributes: Raw attributes released by the provider (empty for local accounts).
    """

    identifier: str
    provider: str
    display_name: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AccountPlan:
    """How a provisioned account should be created (role, level, quota)."""

    role: str = "student"
    level: str = "M1"
    daily_quota: int | None = 50


@runtime_checkable
class AuthProvider(Protocol):
    """Verifies credentials and returns an identity."""

    name: str

    async def authenticate(self, credentials: Credentials) -> Identity | None:
        """Return the identity matching ``credentials``, or ``None`` when rejected."""
        ...


class ProvisioningPolicy(Protocol):
    """Decides whether and how an account is created for a new identity.

    Not used by the local provider (local accounts are created by an
    administrator). The CAS provider will call it on first login to derive
    the role and the level from the released attributes.
    """

    def plan(self, identity: Identity) -> AccountPlan | None:
        """Return the account to create, or ``None`` to refuse the login."""
        ...


class AuthError(Exception):
    """Raised when a provider cannot operate (misconfiguration, remote failure)."""
