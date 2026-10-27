"""Pedagogy page: quizzes, exercises, questions, out-of-corpus subjects, code workshop (lot 5.4).

One page with five tabs (``?tab=``), each built from :mod:`ela.db.pedagogy`
over the period of the query string; every table has a CSV twin.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from backoffice.csvexport import csv_response
from backoffice.deps import current_user
from backoffice.rendering import render
from backoffice.routes.activity import period_of
from backoffice.security import Session
from ela.db import pedagogy

router = APIRouter()

TABS = (
    ("quiz", "Quiz"),
    ("exercises", "Exercices"),
    ("questions", "Questions"),
    ("corpus", "Hors corpus"),
    ("code", "Code"),
)
TAB_KEYS = {key for key, _ in TABS}

CSV_TABLES: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "quiz-sujets": (
        "quiz_by_topic",
        [
            ("topic", "Sujet"),
            ("course", "Cours"),
            ("answers", "Réponses"),
            ("correct", "Correctes"),
            ("success_pct", "Réussite %"),
            ("students", "Étudiants"),
        ],
    ),
    "quiz-questions-ratees": (
        "failed_questions",
        [
            ("question", "Question"),
            ("topic", "Sujet"),
            ("failures", "Échecs"),
            ("students", "Étudiants"),
            ("last_seen", "Dernier échec"),
        ],
    ),
    "exercices": (
        "exercise_stats",
        [
            ("topic", "Sujet"),
            ("exercises", "Exercices"),
            ("attempts", "Étapes soumises"),
            ("students", "Étudiants"),
            ("completed", "Terminés"),
            ("correct_steps_pct", "Étapes correctes %"),
        ],
    ),
    "questions-chapitres": (
        "questions_by_chapter",
        [("course", "Cours"), ("chapter", "Chapitre"), ("questions", "Questions"), ("students", "Étudiants")],
    ),
    "questions-recentes": (
        "recent_questions",
        [
            ("created_at", "Date"),
            ("user_identifier", "Étudiant"),
            ("course", "Cours"),
            ("chapter", "Chapitre"),
            ("question", "Question"),
            ("n_searches", "Recherches"),
            ("n_sources", "Extraits cités"),
            ("general_knowledge", "Connaissances générales"),
            ("refused", "Refusée"),
        ],
    ),
    "hors-corpus": (
        "out_of_corpus",
        [("created_at", "Date"), ("user_identifier", "Étudiant"), ("question", "Question"), ("reason", "Raison")],
    ),
    "code-executions": (
        "code_runs",
        [
            ("status", "Statut"),
            ("origin", "Origine"),
            ("runs", "Exécutions"),
            ("students", "Étudiants"),
            ("avg_seconds", "Durée moyenne (s)"),
            ("figures", "Figures"),
        ],
    ),
    "code-erreurs": ("code_errors", [("error", "Erreur"), ("runs", "Exécutions"), ("last_seen", "Dernière")]),
    "code-etudiants": (
        "code_by_student",
        [("user_identifier", "Étudiant"), ("runs", "Exécutions"), ("ok", "Réussies"), ("today", "Aujourd'hui")],
    ),
}


def _course_of(request: Request) -> str | None:
    return (request.query_params.get("course") or "").strip() or None


@router.get("/pedagogy", name="pedagogy")
async def pedagogy_page(request: Request, tab: str = "quiz", session: Session = Depends(current_user)):
    """Show one tab of the pedagogy dashboard."""
    if tab not in TAB_KEYS:
        tab = "quiz"
    period = period_of(request)
    start, end = period.start, period.end
    ctx: dict = {"period": period, "tab": tab, "tabs": TABS}
    if tab == "quiz":
        ctx["kpis"] = await pedagogy.quiz_kpis(start, end)
        ctx["by_topic"] = await pedagogy.quiz_by_topic(start, end)
        ctx["failed"] = await pedagogy.failed_questions(start, end)
    elif tab == "exercises":
        rows = await pedagogy.exercise_stats(start, end)
        ctx["exercises"] = rows
        ctx["totals"] = {
            "exercises": sum(r["exercises"] for r in rows),
            "completed": sum(r["completed"] for r in rows),
            "attempts": sum(r["attempts"] for r in rows),
        }
    elif tab == "questions":
        course = _course_of(request)
        ctx["course"] = course
        ctx["courses"] = await pedagogy.courses_with_questions()
        ctx["by_chapter"] = await pedagogy.questions_by_chapter(start, end)
        ctx["recent"] = await pedagogy.recent_questions(start, end, course)
    elif tab == "corpus":
        ctx["corpus"] = await pedagogy.out_of_corpus(start, end)
    else:
        runs = await pedagogy.code_runs(start, end)
        ctx["runs"] = runs
        ctx["totals"] = {
            "runs": sum(r["runs"] for r in runs),
            "ok": sum(r["runs"] for r in runs if r["status"] == "ok"),
            "students": max((r["students"] for r in runs), default=0),
        }
        ctx["errors"] = await pedagogy.code_errors(start, end)
        ctx["by_student"] = await pedagogy.code_by_student(start, end)
    return render(request, "pedagogy/index.html", ctx)


@router.get("/pedagogy/{table}.csv", name="pedagogy_csv")
async def pedagogy_csv(request: Request, table: str, session: Session = Depends(current_user)):
    """Download one table of the pedagogy page (same period and course filter)."""
    if table not in CSV_TABLES:
        return render(request, "error.html", {"title": "Export inconnu", "message": "Ce tableau n'existe pas."}, 404)
    period = period_of(request)
    func_name, columns = CSV_TABLES[table]
    func = getattr(pedagogy, func_name)
    if func_name == "recent_questions":
        rows = await func(period.start, period.end, _course_of(request), limit=5000)
    elif func_name in ("failed_questions", "out_of_corpus", "code_errors"):
        rows = await func(period.start, period.end, limit=5000)
    else:
        rows = await func(period.start, period.end)
    return csv_response(f"{table}-{period.start}-{period.end}", columns, rows)
