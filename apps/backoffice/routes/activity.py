"""Activity page: usage figures, daily volume, conversations by kind, ranking, quotas (lot 5.4).

Read-only for ``admin`` and ``supervisor``. The period comes from the query
string (see :mod:`backoffice.periods`), the optional ``user`` restricts the
daily charts to one account (the ranking and the quotas always cover everyone).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from backoffice import charts
from backoffice.csvexport import csv_response
from backoffice.deps import current_user
from backoffice.periods import Period, parse_period
from backoffice.rendering import render
from backoffice.security import Session
from ela.db import dashboard, stats
from ela.db.threads import KIND_LABELS, THREAD_KINDS

router = APIRouter()


def period_of(request: Request) -> Period:
    """Period described by the ``from`` / ``to`` / ``days`` query parameters."""
    q = request.query_params
    return parse_period(q.get("from"), q.get("to"), q.get("days"))


def _user_of(request: Request) -> str | None:
    return (request.query_params.get("user") or "").strip() or None


async def _daily_series(period: Period, user: str | None) -> dict:
    """Daily messages and daily conversations by kind, aligned on every day of the period."""
    days = stats.period_days(period.start, period.end)
    labels = charts.day_labels(days)
    messages = {r["day"]: r["messages"] for r in await stats.daily_messages(period.start, period.end, user)}
    by_day_kind: dict[tuple, int] = {}
    for r in await stats.daily_threads_by_kind(period.start, period.end, user):
        by_day_kind[(r["day"], r["kind"])] = r["threads"]
    series = [
        (kind, KIND_LABELS[kind], [by_day_kind.get((d, kind), 0) for d in days])
        for kind in THREAD_KINDS
        if any(by_day_kind.get((d, kind), 0) for d in days)
    ]
    return {
        "days": days,
        "labels": labels,
        "messages": [messages.get(d, 0) for d in days],
        "series": series,
        "total_messages": sum(messages.values()),
    }


@router.get("/activity", name="activity")
async def activity(request: Request, session: Session = Depends(current_user)):
    """Show the figures of the period, with the charts rendered as inline SVG."""
    period = period_of(request)
    user = _user_of(request)
    overview = await dashboard.overview(recent_limit=1)
    daily = await _daily_series(period, user)
    by_kind = await stats.threads_by_kind(period.start, period.end, user)
    ranking = await stats.messages_by_user(period.start, period.end)
    quotas = await stats.quota_usage()
    ctx = {
        "period": period,
        "user": user,
        "users": await stats.user_identifiers(),
        "overview": overview,
        "daily": daily,
        "by_kind": by_kind,
        "kind_labels": KIND_LABELS,
        "ranking": ranking,
        "quotas": quotas,
        "chart_messages": charts.line_chart(daily["labels"], daily["messages"], unit="messages", key="messages")
        if daily["total_messages"]
        else None,
        "chart_kinds": charts.donut_chart(
            [(r["kind"], KIND_LABELS[r["kind"]], r["threads"]) for r in by_kind], unit="conversations"
        )
        if by_kind
        else None,
        "chart_daily_kinds": charts.stacked_area_chart(
            daily["labels"], daily["series"], unit="conversations", width=440
        )
        if daily["series"]
        else None,
        "chart_ranking": charts.hbar_chart(
            [(r["identifier"], r["messages"]) for r in ranking], unit="messages", key="messages"
        )
        if ranking
        else None,
        "total_threads": sum(r["threads"] for r in by_kind),
    }
    return render(request, "activity/index.html", ctx)


@router.get("/activity/daily.csv", name="activity_daily_csv")
async def activity_daily_csv(request: Request, session: Session = Depends(current_user)):
    """Download, day by day, the student messages and the conversations created by kind."""
    period = period_of(request)
    daily = await _daily_series(period, _user_of(request))
    series = {kind: values for kind, _, values in daily["series"]}
    rows = [
        {"day": d, "messages": m, **{kind: series.get(kind, [0] * len(daily["days"]))[i] for kind in THREAD_KINDS}}
        for i, (d, m) in enumerate(zip(daily["days"], daily["messages"], strict=True))
    ]
    columns = [("day", "Jour"), ("messages", "Messages")] + [
        (k, f"Conversations {KIND_LABELS[k]}") for k in THREAD_KINDS
    ]
    return csv_response(f"activite-{period.start}-{period.end}", columns, rows)


@router.get("/activity/users.csv", name="activity_users_csv")
async def activity_users_csv(request: Request, session: Session = Depends(current_user)):
    """Download the ranking of the users over the period."""
    period = period_of(request)
    rows = await stats.messages_by_user(period.start, period.end)
    columns = [
        ("identifier", "Identifiant"),
        ("role", "Rôle"),
        ("level", "Niveau"),
        ("messages", "Messages"),
        ("threads", "Conversations"),
        ("last_message", "Dernier message"),
    ]
    return csv_response(f"utilisateurs-{period.start}-{period.end}", columns, rows)


@router.get("/activity/quotas.csv", name="activity_quotas_csv")
async def activity_quotas_csv(request: Request, session: Session = Depends(current_user)):
    """Download today's quota consumption."""
    rows = await stats.quota_usage()
    columns = [
        ("identifier", "Identifiant"),
        ("role", "Rôle"),
        ("level", "Niveau"),
        ("quota", "Quota"),
        ("used", "Utilisé"),
    ]
    return csv_response("quotas", columns, rows)
