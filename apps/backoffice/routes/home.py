"""Home page: figures and latest conversations."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from backoffice.deps import current_user
from backoffice.rendering import render
from backoffice.security import Session
from ela.db import audit, dashboard

router = APIRouter()


@router.get("/", name="home")
async def home(request: Request, session: Session = Depends(current_user)):
    """Dashboard with counters, latest conversations and (admins) the audit trail."""
    overview = await dashboard.overview()
    trail = await audit.recent(limit=15) if session.is_admin else []
    return render(request, "home.html", {"overview": overview, "audit_rows": trail})


@router.get("/partials/overview", name="overview_partial")
async def overview_partial(request: Request, session: Session = Depends(current_user)):
    """HTMX fragment refreshed periodically by the home page."""
    overview = await dashboard.overview()
    return render(request, "partials/overview.html", {"overview": overview})
