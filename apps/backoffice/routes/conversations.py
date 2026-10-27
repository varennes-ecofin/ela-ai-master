"""Conversations: filtered list and read-only reader of one thread (lot 5.4)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from backoffice.csvexport import csv_response
from backoffice.deps import current_user
from backoffice.paging import paging
from backoffice.rendering import render
from backoffice.routes.activity import period_of
from backoffice.security import Session
from ela.db import conversations, stats
from ela.db.threads import KIND_LABELS, THREAD_KINDS

router = APIRouter()


def _filters(request: Request) -> tuple[str | None, str | None]:
    q = request.query_params
    user = (q.get("user") or "").strip() or None
    kind = q.get("kind") or ""
    return user, kind if kind in THREAD_KINDS else None


@router.get("/conversations", name="conversations")
async def conversations_list(request: Request, session: Session = Depends(current_user)):
    """List the conversations of the period, filtered by user and kind."""
    period = period_of(request)
    user, kind = _filters(request)
    pg = paging(request.query_params.get("page"))
    result = await conversations.list_threads(period.start, period.end, user, kind, limit=pg.size, offset=pg.offset)
    ctx = {
        "period": period,
        "user": user,
        "kind": kind or "",
        "kinds": list(THREAD_KINDS),
        "kind_labels": KIND_LABELS,
        "users": await stats.user_identifiers(),
        "rows": result.rows,
        "paging": pg.with_total(result.total),
    }
    return render(request, "conversations/index.html", ctx)


@router.get("/conversations.csv", name="conversations_csv")
async def conversations_csv(request: Request, session: Session = Depends(current_user)):
    """Download the filtered list (metadata only, not the messages)."""
    period = period_of(request)
    user, kind = _filters(request)
    result = await conversations.list_threads(period.start, period.end, user, kind, limit=500, offset=0)
    rows = result.rows
    while len(rows) < result.total and len(rows) < 5000:
        more = await conversations.list_threads(period.start, period.end, user, kind, limit=500, offset=len(rows))
        if not more.rows:
            break
        rows += more.rows
    for r in rows:
        r["kind"] = KIND_LABELS.get(r["kind"], r["kind"])
    columns = [
        ("created_at", "Date"),
        ("identifier", "Étudiant"),
        ("kind", "Type"),
        ("name", "Nom"),
        ("messages", "Messages"),
        ("steps", "Étapes"),
        ("feedbacks", "Feedbacks"),
        ("id", "Id"),
    ]
    return csv_response(f"conversations-{period.start}-{period.end}", columns, rows)


@router.get("/conversations/{thread_id}", name="conversation_detail")
async def conversation_detail(
    request: Request, thread_id: str, tools: str = "", session: Session = Depends(current_user)
):
    """Show one conversation (``?tools=1`` also shows the agent's tool steps)."""
    thread = await conversations.get_thread(thread_id)
    if thread is None:
        return render(
            request,
            "error.html",
            {"title": "Conversation introuvable", "message": "Aucune conversation ne porte cet identifiant."},
            404,
        )
    show_tools = tools == "1"
    steps = await conversations.thread_steps(thread_id, include_tools=show_tools)
    ctx = {
        "thread": thread,
        "kind_label": KIND_LABELS.get(thread["kind"], thread["kind"]),
        "steps": steps,
        "show_tools": show_tools,
        "messages": sum(1 for s in steps if s["type"] == "user_message"),
    }
    return render(request, "conversations/detail.html", ctx)
