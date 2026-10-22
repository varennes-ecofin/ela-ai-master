"""FastAPI dependencies: current session, role checks, CSRF, client address."""

from __future__ import annotations

import logging
from collections.abc import Callable

from fastapi import Depends, Request

from backoffice.security import CSRF_COOKIE, CSRF_FIELD, CSRF_HEADER, SESSION_COOKIE, Session
from ela.db import users as users_db

logger = logging.getLogger("ela.backoffice")

# Roles allowed to open the backoffice; ``admin`` alone may change things.
BACKOFFICE_ROLES = ("admin", "supervisor")


class LoginRequiredError(Exception):
    """Raised when a page needs a session; handled by a redirect to the login page."""

    def __init__(self, next_path: str | None = None) -> None:
        super().__init__("login required")
        self.next_path = next_path


class ForbiddenError(Exception):
    """Raised when the session lacks the required role."""


class CsrfError(Exception):
    """Raised when a ``POST`` carries no valid CSRF token."""


def client_ip(request: Request) -> str | None:
    """Client address, honouring ``X-Forwarded-For`` set by the reverse proxy."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip() or None
    return request.client.host if request.client else None


def get_session(request: Request) -> Session | None:
    """Session carried by the cookie, or ``None`` (no exception)."""
    signer = request.app.state.signer
    session = signer.load_session(request.cookies.get(SESSION_COOKIE))
    request.state.session = session
    return session


async def current_user(request: Request, session: Session | None = Depends(get_session)) -> Session:
    """Require a session with a backoffice role, re-checked against the ``users`` table.

    The account must still exist and be active; its *current* role is used,
    so that deactivating an account or demoting it takes effect at the next
    request without waiting for the cookie to expire.

    Raises:
        LoginRequiredError: No (valid) session cookie, or the account vanished / was deactivated.
        ForbiddenError: The account's role may not use the backoffice.
    """
    if session is None:
        raise LoginRequiredError(next_path=request.url.path)
    try:
        profile = await users_db.get_profile(session.identifier)
    except Exception:
        logger.warning("Could not re-check the account of %s", session.identifier, exc_info=True)
        raise LoginRequiredError(next_path=request.url.path) from None
    if profile is None or not profile.is_active:
        raise LoginRequiredError(next_path=request.url.path)
    if profile.role != session.role or profile.level != session.level:
        session = Session(
            identifier=session.identifier,
            role=profile.role,
            level=profile.level,
            provider=session.provider,
            issued_at=session.issued_at,
        )
        request.state.session = session
    if session.role not in BACKOFFICE_ROLES:
        raise ForbiddenError()
    return session


def require_role(*roles: str) -> Callable[..., Session]:
    """Dependency factory: the session must hold one of ``roles``."""

    def _dependency(session: Session = Depends(current_user)) -> Session:
        if session.role not in roles:
            raise ForbiddenError()
        return session

    return _dependency


require_admin = require_role("admin")


async def csrf_protect(request: Request) -> None:
    """Verify the CSRF token of a state-changing request (form field or header)."""
    submitted = request.headers.get(CSRF_HEADER)
    if submitted is None and request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        content_type = request.headers.get("content-type", "")
        if "form" in content_type:
            form = await request.form()
            value = form.get(CSRF_FIELD)
            submitted = value if isinstance(value, str) else None
    if not request.app.state.signer.csrf_valid(request.cookies.get(CSRF_COOKIE), submitted):
        raise CsrfError()
