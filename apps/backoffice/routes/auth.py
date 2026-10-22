"""Login and logout (local accounts; the provider is pluggable)."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse, Response

from backoffice.deps import BACKOFFICE_ROLES, client_ip, csrf_protect, get_session
from backoffice.rendering import render
from backoffice.security import SESSION_COOKIE, Session
from ela.auth import AuthError, Credentials
from ela.db import audit
from ela.db import users as users_db

logger = logging.getLogger("ela.backoffice.auth")
router = APIRouter()


def _safe_next(request: Request, candidate: str | None) -> str:
    """Only allow redirects to a path of this application (no open redirect)."""
    prefix = request.app.state.prefix
    home = f"{prefix}/"
    if not candidate or not candidate.startswith("/") or candidate.startswith("//"):
        return home
    if prefix and not candidate.startswith(prefix + "/") and candidate != prefix:
        return home
    if candidate.rstrip("/") == f"{prefix}/login".rstrip("/"):
        return home
    return candidate


def set_session_cookie(request: Request, response: Response, session: Session) -> None:
    """Attach a signed session cookie to ``response``."""
    state = request.app.state
    response.set_cookie(
        SESSION_COOKIE,
        state.signer.dump_session(session),
        max_age=state.signer.max_age,
        httponly=True,
        samesite="lax",
        secure=state.cookie_secure,
        path=state.cookie_path,
    )


def clear_session_cookie(request: Request, response: Response) -> None:
    """Remove the session cookie."""
    response.delete_cookie(SESSION_COOKIE, path=request.app.state.cookie_path)


@router.get("/login", name="login")
async def login_form(request: Request, next: str | None = None, session: Session | None = Depends(get_session)):
    """Show the login form (or go home when already signed in)."""
    if session is not None and session.role in BACKOFFICE_ROLES:
        return RedirectResponse(_safe_next(request, next), status_code=303)
    return render(
        request,
        "login.html",
        {"next": _safe_next(request, next), "logged_out": request.query_params.get("logged_out") == "1"},
    )


@router.post("/login", name="login_submit", dependencies=[Depends(csrf_protect)])
async def login_submit(
    request: Request,
    identifier: str = Form(default=""),
    password: str = Form(default=""),
    next: str = Form(default=""),
):
    """Check the credentials, open the session and record the outcome."""
    state = request.app.state
    identifier = identifier.strip()
    ip = client_ip(request)
    target = _safe_next(request, next)

    def failure(message: str, status: int = 401, **extra):
        return render(
            request, "login.html", {"error": message, "identifier": identifier, "next": target, **extra}, status
        )

    if not identifier or not password:
        return failure("Identifiant et mot de passe sont obligatoires.", 400)

    remaining = state.limiter.locked_for(identifier)
    if remaining > 0:
        await audit.record(audit.LOGIN_LOCKED, identifier, ip=ip, details={"remaining_s": int(remaining)})
        return failure(f"Trop de tentatives : compte verrouillé pendant {int(remaining // 60) + 1} min.", 429)

    try:
        identity = await state.auth_provider.authenticate(Credentials(identifier=identifier, password=password))
    except AuthError as exc:
        logger.error("Authentication provider error: %s", exc)
        return failure("Service d'authentification indisponible.", 503)

    if identity is None:
        locked = state.limiter.register_failure(identifier)
        await audit.record(audit.LOGIN_FAILED, identifier, ip=ip, details={"reason": "credentials", "locked": locked})
        return failure("Identifiants invalides.")

    profile = await users_db.get_profile(identity.identifier)
    role = profile.role if profile else str(identity.attributes.get("role") or "student")
    if role not in BACKOFFICE_ROLES:
        await audit.record(audit.LOGIN_FAILED, identity.identifier, ip=ip, details={"reason": "role", "role": role})
        return failure("Ce compte n'a pas accès au backoffice.", 403)

    state.limiter.reset(identifier)
    session = Session(
        identifier=identity.identifier,
        role=role,
        level=(profile.level if profile else "ALL"),
        provider=identity.provider,
        issued_at=time.time(),
    )
    await audit.record(audit.LOGIN_OK, session.identifier, ip=ip, details={"role": role, "provider": identity.provider})
    logger.info("Backoffice login: %s (role=%s)", session.identifier, role)
    response = RedirectResponse(target, status_code=303)
    set_session_cookie(request, response, session)
    return response


@router.post("/logout", name="logout", dependencies=[Depends(csrf_protect)])
async def logout(request: Request, session: Session | None = Depends(get_session)):
    """Close the session."""
    if session is not None:
        await audit.record(audit.LOGOUT, session.identifier, ip=client_ip(request))
    response = RedirectResponse(f"{request.app.state.prefix}/login?logged_out=1", status_code=303)
    clear_session_cookie(request, response)
    return response
