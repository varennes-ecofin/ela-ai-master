"""Health endpoint for Docker and the reverse proxy (no authentication)."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ela.db.engine import fetch_one

router = APIRouter()


@router.get("/healthz", name="healthz")
async def healthz(request: Request, deep: bool = False):
    """Return ``{"status": "ok"}``; with ``?deep=1`` also check the database (503 on failure)."""
    payload = {"status": "ok", "version": request.app.state.version}
    if deep:
        try:
            await fetch_one("SELECT 1 AS one")
            payload["database"] = "ok"
        except Exception as exc:
            payload.update(status="degraded", database=f"{exc.__class__.__name__}: {exc}")
            return JSONResponse(payload, status_code=503)
    return JSONResponse(payload)
