"""Session cookies, CSRF tokens and login throttling for the backoffice.

* **Session**: a signed, timestamped cookie (``itsdangerous``) holding the
  identifier and the role. No server-side state, so several backoffice
  instances can share the same secret; revocation happens by changing
  the secret or deactivating the account (checked at each request).
* **CSRF**: a signed random token in a dedicated cookie, repeated in every
  ``POST`` (form field ``csrf_token`` or header ``X-CSRF-Token``); this
  double-submit scheme also protects the login form, which has no session yet.
* **Throttling**: failed logins per identifier are counted in memory and the
  identifier is locked for a while after too many failures.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

SESSION_COOKIE = "ela_bo_session"
CSRF_COOKIE = "ela_bo_csrf"
CSRF_FIELD = "csrf_token"
CSRF_HEADER = "x-csrf-token"


@dataclass(frozen=True)
class Session:
    """The authenticated person behind a request."""

    identifier: str
    role: str
    level: str = "ALL"
    provider: str = "local"
    issued_at: float = 0.0

    @property
    def is_admin(self) -> bool:
        """Whether the session may change things (accounts, ingestion...)."""
        return self.role == "admin"


class SessionSigner:
    """Sign and verify session and CSRF cookies.

    Args:
        secret_key: Secret shared by all backoffice instances.
        max_age_seconds: Lifetime of a session cookie.
    """

    def __init__(self, secret_key: str, max_age_seconds: float) -> None:
        if not secret_key or len(secret_key) < 16:
            raise ValueError("BACKOFFICE_SECRET_KEY must be at least 16 characters long.")
        self.max_age = int(max_age_seconds)
        self._session = URLSafeTimedSerializer(secret_key, salt="ela-backoffice-session")
        self._csrf = URLSafeTimedSerializer(secret_key, salt="ela-backoffice-csrf")

    # --- session -----------------------------------------------------------------
    def dump_session(self, session: Session) -> str:
        """Serialise a session into a cookie value."""
        return self._session.dumps(
            {
                "sub": session.identifier,
                "role": session.role,
                "level": session.level,
                "prov": session.provider,
                "iat": session.issued_at or time.time(),
            }
        )

    def load_session(self, value: str | None) -> Session | None:
        """Return the session carried by a cookie value, or ``None`` if absent, altered or expired."""
        if not value:
            return None
        try:
            data = self._session.loads(value, max_age=self.max_age)
        except (BadSignature, SignatureExpired):
            return None
        if not isinstance(data, dict) or not data.get("sub") or not data.get("role"):
            return None
        return Session(
            identifier=str(data["sub"]),
            role=str(data["role"]),
            level=str(data.get("level") or "ALL"),
            provider=str(data.get("prov") or "local"),
            issued_at=float(data.get("iat") or 0.0),
        )

    # --- CSRF --------------------------------------------------------------------
    def new_csrf(self) -> str:
        """Return a fresh signed CSRF token."""
        return self._csrf.dumps(secrets.token_urlsafe(24))

    def csrf_valid(self, cookie_value: str | None, submitted: str | None) -> bool:
        """Check that the submitted token equals the signed cookie token."""
        if not cookie_value or not submitted:
            return False
        try:
            self._csrf.loads(cookie_value, max_age=max(self.max_age, 24 * 3600))
        except (BadSignature, SignatureExpired):
            return False
        return secrets.compare_digest(cookie_value, submitted)


class LoginLimiter:
    """Count failed logins per identifier and lock after too many.

    Args:
        max_failures: Failures allowed before the lock (0 disables locking).
        lock_seconds: Duration of the lock.
    """

    def __init__(self, max_failures: int = 5, lock_seconds: float = 900.0) -> None:
        self.max_failures = max(0, int(max_failures))
        self.lock_seconds = float(lock_seconds)
        self._failures: dict[str, int] = {}
        self._locked_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def _key(self, identifier: str) -> str:
        return identifier.strip().lower()

    def locked_for(self, identifier: str) -> float:
        """Seconds remaining on the lock of ``identifier`` (0 when not locked)."""
        key = self._key(identifier)
        with self._lock:
            until = self._locked_until.get(key, 0.0)
            remaining = until - time.monotonic()
            if remaining <= 0:
                self._locked_until.pop(key, None)
                return 0.0
            return remaining

    def register_failure(self, identifier: str) -> bool:
        """Record a failure; return ``True`` when it triggers the lock."""
        if self.max_failures == 0:
            return False
        key = self._key(identifier)
        with self._lock:
            count = self._failures.get(key, 0) + 1
            if count >= self.max_failures:
                self._failures.pop(key, None)
                self._locked_until[key] = time.monotonic() + self.lock_seconds
                return True
            self._failures[key] = count
            return False

    def reset(self, identifier: str) -> None:
        """Forget the failures of ``identifier`` (successful login)."""
        key = self._key(identifier)
        with self._lock:
            self._failures.pop(key, None)
            self._locked_until.pop(key, None)
