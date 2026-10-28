"""Application factory and local launcher of the ELA backoffice.

Local run (project root, ``.env`` loaded by ``ela.config``)::

    python apps/backoffice/main.py --reload          # http://localhost:8502/backoffice/

Docker / production::

    uvicorn backoffice.main:create_app --factory --app-dir apps --host 0.0.0.0 --port 8080

The whole application is served under ``BACKOFFICE_ROOT_PATH`` (default
``/backoffice``), locally as well as behind the reverse proxy, so no URL
rewriting is needed anywhere.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from importlib import metadata
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):  # `python apps/backoffice/main.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from backoffice.deps import CsrfError, ForbiddenError, LoginRequiredError
from backoffice.oneshot import OneShotStore
from backoffice.rendering import BRAND_DIR, STATIC_DIR, render
from backoffice.routes import account as account_routes
from backoffice.routes import activity as activity_routes
from backoffice.routes import auth as auth_routes
from backoffice.routes import conversations as conversations_routes
from backoffice.routes import courses as courses_routes
from backoffice.routes import evaluation as evaluation_routes
from backoffice.routes import feedbacks as feedbacks_routes
from backoffice.routes import health as health_routes
from backoffice.routes import home as home_routes
from backoffice.routes import pedagogy as pedagogy_routes
from backoffice.routes import users as users_routes
from backoffice.security import LoginLimiter, SessionSigner
from ela.auth import build_auth_provider
from ela.config import Settings, get_settings
from ela.db import jobs as jobs_db
from ela.eval.reports import ReportIndex
from ela.jobs.runner import JobRunner
from ela.log import configure_logging

logger = logging.getLogger("ela.backoffice")


def _version() -> str:
    try:
        return metadata.version("ela-ai")
    except metadata.PackageNotFoundError:  # running from a checkout without install
        return "dev"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Conservative headers on every response (the reverse proxy adds HSTS)."""

    async def dispatch(self, request: Request, call_next):
        """Add the headers to the response of ``call_next``."""
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        if "/static/" not in request.url.path and "/brand/" not in request.url.path:
            response.headers.setdefault("Cache-Control", "no-store")
        return response


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Settings override (tests); defaults to the process settings.

    Returns:
        The configured application, mounted under ``settings.backoffice_prefix``.

    Raises:
        ValueError: When ``BACKOFFICE_SECRET_KEY`` is missing or too short.
    """
    s = settings or get_settings()
    configure_logging()
    if s.backoffice_secret_key is None:
        raise ValueError(
            "BACKOFFICE_SECRET_KEY is not set: generate one with "
            '`python -c "import secrets; print(secrets.token_urlsafe(48))"`.'
        )
    prefix = s.backoffice_prefix

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Backoffice %s ready under %s/ (auth=%s)", app.state.version, prefix, app.state.auth_provider.name)
        if await jobs_db.mark_interrupted():
            logger.warning("Jobs left running by a previous process were marked interrupted")
        yield
        from ela.db import engine as db_engine

        await db_engine.dispose()

    app = FastAPI(title="ELA AI — Backoffice", version=_version(), docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.settings = s
    app.state.version = _version()
    app.state.prefix = prefix
    app.state.cookie_path = prefix or "/"
    app.state.cookie_secure = s.ela_env == "prod"
    app.state.signer = SessionSigner(s.backoffice_secret_key.get_secret_value(), s.backoffice_session_hours * 3600)
    app.state.limiter = LoginLimiter(s.backoffice_login_max_failures, s.backoffice_login_lock_minutes * 60)
    app.state.auth_provider = build_auth_provider(s)
    app.state.oneshot = OneShotStore(ttl_seconds=15 * 60)
    app.state.jobs = JobRunner(source_dir=s.latex_source_dir)
    app.state.reports = ReportIndex(Path(s.eval_reports_dir))

    app.add_middleware(SecurityHeadersMiddleware)

    root = APIRouter(prefix=prefix)
    root.include_router(health_routes.router)
    root.include_router(auth_routes.router)
    root.include_router(home_routes.router)
    root.include_router(users_routes.router)
    root.include_router(account_routes.router)
    root.include_router(courses_routes.router)
    root.include_router(activity_routes.router)
    root.include_router(pedagogy_routes.router)
    root.include_router(feedbacks_routes.router)
    root.include_router(conversations_routes.router)
    root.include_router(evaluation_routes.router)
    app.include_router(root)
    app.mount(f"{prefix}/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    if BRAND_DIR.is_dir():
        app.mount(f"{prefix}/brand", StaticFiles(directory=str(BRAND_DIR)), name="brand")

    if prefix:

        @app.get(prefix, include_in_schema=False)
        async def _prefix_root():
            return RedirectResponse(f"{prefix}/", status_code=307)

        @app.get("/", include_in_schema=False)
        async def _site_root():
            return RedirectResponse(f"{prefix}/", status_code=307)

    # --- error handling -----------------------------------------------------------
    @app.exception_handler(LoginRequiredError)
    async def _login_required(request: Request, exc: LoginRequiredError):
        login = f"{prefix}/login"
        if exc.next_path and exc.next_path != f"{prefix}/":
            login += f"?next={exc.next_path}"
        if request.headers.get("hx-request") == "true":
            return JSONResponse({"detail": "login required"}, status_code=401, headers={"HX-Redirect": login})
        return RedirectResponse(login, status_code=303)

    @app.exception_handler(ForbiddenError)
    async def _forbidden(request: Request, exc: ForbiddenError):
        return render(
            request, "error.html", {"title": "Accès refusé", "message": "Ce compte n'a pas accès à cette page."}, 403
        )

    @app.exception_handler(CsrfError)
    async def _csrf(request: Request, exc: CsrfError):
        message = "Jeton de sécurité absent ou expiré : rechargez la page puis recommencez."
        if request.headers.get("hx-request") == "true":
            return JSONResponse({"detail": message}, status_code=403)
        return render(request, "error.html", {"title": "Requête refusée", "message": message}, 403)

    return app


def main(argv: list[str] | None = None) -> None:
    """Local launcher (``python apps/backoffice/main.py [--reload] [--port N]``)."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="ELA AI — backoffice (développement)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--reload", action="store_true", help="Recharger à chaque modification de fichier.")
    args = parser.parse_args(argv)
    uvicorn.run(
        "backoffice.main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(Path(__file__).resolve().parents[2] / "src"), str(Path(__file__).resolve().parent)],
        app_dir=str(Path(__file__).resolve().parents[1]),
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
