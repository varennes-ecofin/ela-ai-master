"""Feedback page: thumbs given by the students on the answers of ELA (lot 5.4)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from backoffice.csvexport import csv_response
from backoffice.deps import current_user
from backoffice.paging import paging
from backoffice.rendering import render
from backoffice.routes.activity import period_of
from backoffice.security import Session
from ela.db import feedbacks, stats

router = APIRouter()

SCORES = {"1": 1, "0": 0}


def _filters(request: Request) -> tuple[str | None, int | None, str]:
    q = request.query_params
    user = (q.get("user") or "").strip() or None
    score_key = q.get("score") or ""
    return user, SCORES.get(score_key), score_key if score_key in SCORES else ""


@router.get("/feedbacks", name="feedbacks")
async def feedbacks_page(request: Request, session: Session = Depends(current_user)):
    """Show the global counters and the filtered list of feedback, one page at a time."""
    period = period_of(request)
    user, score, score_key = _filters(request)
    pg = paging(request.query_params.get("page"))
    result = await feedbacks.list_feedbacks(period.start, period.end, user, score, limit=pg.size, offset=pg.offset)
    ctx = {
        "period": period,
        "user": user,
        "score": score_key,
        "users": await stats.user_identifiers(),
        "stats": await feedbacks.stats(),
        "rows": result.rows,
        "paging": pg.with_total(result.total),
    }
    return render(request, "feedbacks/index.html", ctx)


@router.get("/feedbacks/{feedback_id}/context", name="feedback_context")
async def feedback_context(request: Request, feedback_id: str, session: Session = Depends(current_user)):
    """HTMX fragment: the exchange around one rated answer."""
    ctx = await feedbacks.context(feedback_id)
    return render(request, "feedbacks/_context.html", {"ctx": ctx, "feedback_id": feedback_id}, 200 if ctx else 404)


@router.get("/feedbacks.csv", name="feedbacks_csv")
async def feedbacks_csv(request: Request, session: Session = Depends(current_user)):
    """Download every feedback matching the filters (not only the current page)."""
    period = period_of(request)
    user, score, _ = _filters(request)
    result = await feedbacks.list_feedbacks(period.start, period.end, user, score, limit=500, offset=0)
    rows = result.rows
    while len(rows) < result.total and len(rows) < 5000:
        more = await feedbacks.list_feedbacks(period.start, period.end, user, score, limit=500, offset=len(rows))
        if not more.rows:
            break
        rows += more.rows
    columns = [
        ("answered_at", "Date"),
        ("identifier", "Étudiant"),
        ("thread_name", "Conversation"),
        ("score", "Score"),
        ("comment", "Commentaire"),
        ("answer", "Réponse d'ELA"),
        ("thread_id", "Id conversation"),
    ]
    return csv_response(f"feedbacks-{period.start}-{period.end}", columns, rows)
