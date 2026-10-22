"""Jinja2 rendering helpers shared by the routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import jinja2
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from backoffice.security import CSRF_COOKIE, CSRF_FIELD
from ela.config import get_settings

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
# Brand assets shared with the chat application (logo, favicon): ``public/`` at the project root
# (``/app/public`` in Docker, see apps/backoffice/Dockerfile).
BRAND_DIR = Path(__file__).resolve().parents[2] / "public"

templates = Jinja2Templates(env=jinja2.Environment(loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True))


def _format_datetime(value: Any, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Jinja filter: format a datetime in the display time zone (``BACKOFFICE_TIMEZONE``)."""
    if not hasattr(value, "strftime"):
        return "" if value is None else str(value)
    if getattr(value, "tzinfo", None) is not None:
        try:
            value = value.astimezone(ZoneInfo(get_settings().backoffice_timezone))
        except (ZoneInfoNotFoundError, ValueError):
            pass
    return value.strftime(fmt)


templates.env.filters["dt"] = _format_datetime


def ensure_csrf(request: Request) -> str:
    """Return the CSRF token of the request, creating it when the cookie is absent or invalid.

    The token is stored on ``request.state`` so that the response can set the cookie.
    """
    signer = request.app.state.signer
    token = request.cookies.get(CSRF_COOKIE)
    if not signer.csrf_valid(token, token):
        token = signer.new_csrf()
        request.state.set_csrf_cookie = token
    return token


def render(request: Request, name: str, context: dict[str, Any] | None = None, status_code: int = 200) -> HTMLResponse:
    """Render a template with the common context (settings, session, CSRF).

    Args:
        request: Current request.
        name: Template file name.
        context: Extra variables.
        status_code: HTTP status of the response.

    Returns:
        The HTML response; the CSRF cookie is set when a new token was issued.
    """
    token = ensure_csrf(request)
    ctx = {
        "prefix": request.app.state.prefix,
        "session": getattr(request.state, "session", None),
        "csrf_token": token,
        "csrf_field": CSRF_FIELD,
        "app_version": request.app.state.version,
        "ela_env": request.app.state.settings.ela_env,
        "htmx_request": request.headers.get("hx-request") == "true",
    }
    ctx.update(context or {})
    response = templates.TemplateResponse(request, name, ctx, status_code=status_code)
    new_token = getattr(request.state, "set_csrf_cookie", None)
    if new_token:
        response.set_cookie(
            CSRF_COOKIE,
            new_token,
            max_age=24 * 3600,
            httponly=True,
            samesite="lax",
            secure=request.app.state.cookie_secure,
            path=request.app.state.cookie_path,
        )
    return response
