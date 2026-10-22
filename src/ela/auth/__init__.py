"""Authentication providers (``AUTH_PROVIDER``).

Usage::

    from ela.auth import get_auth_provider, Credentials
    provider = get_auth_provider()
    identity = await provider.authenticate(Credentials(identifier="admin", password="..."))
"""

from __future__ import annotations

from functools import lru_cache

from ela.auth.base import AccountPlan, AuthError, AuthProvider, Credentials, Identity, ProvisioningPolicy
from ela.config import Settings, get_settings

__all__ = [
    "AccountPlan",
    "AuthError",
    "AuthProvider",
    "Credentials",
    "Identity",
    "ProvisioningPolicy",
    "build_auth_provider",
    "get_auth_provider",
]


def build_auth_provider(settings: Settings | None = None) -> AuthProvider:
    """Instantiate the provider selected by ``AUTH_PROVIDER``.

    Args:
        settings: Settings override (defaults to the process settings).

    Returns:
        A provider instance.

    Raises:
        AuthError: For a provider that is declared but not implemented yet (``cas``).
    """
    s = settings or get_settings()
    if s.auth_provider == "local":
        from ela.auth.local import LocalPasswordProvider

        return LocalPasswordProvider()
    raise AuthError(f"AUTH_PROVIDER={s.auth_provider!r} is not available yet (only 'local' is implemented).")


@lru_cache(maxsize=1)
def get_auth_provider() -> AuthProvider:
    """Return the process-wide provider (cached)."""
    return build_auth_provider()


def reset() -> None:
    """Forget the cached provider (tests, settings reload)."""
    get_auth_provider.cache_clear()
