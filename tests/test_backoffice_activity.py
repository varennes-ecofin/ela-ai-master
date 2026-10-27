"""Offline tests of the lot 5.4 pages: activity, pedagogy, feedback, conversations, CSV exports, charts, periods.

The ``ela.db`` modules are replaced by in-memory fakes (see ``test_backoffice.py``
for the base fixture); the queries themselves are covered by
``test_backoffice_db.py`` against PostgreSQL.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

from ela.db.conversations import ThreadPage  # noqa: E402
from ela.db.feedbacks import FeedbackPage, FeedbackStats  # noqa: E402
from test_backoffice import PREFIX, FakeDb, login, make_settings  # noqa: E402

TODAY = date(2026, 10, 27)
THREAD = "11111111-1111-4111-8111-111111111111"
STEP = "22222222-2222-4222-8222-222222222222"
FEEDBACK = "33333333-3333-4333-8333-333333333333"


class FakeActivityDb:
    """In-memory replacement of ``ela.db.stats``, ``pedagogy``, ``feedbacks`` and ``conversations``."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    # --- stats ---------------------------------------------------------------------
    async def daily_messages(self, start, end, user=None):
        self.calls.append(("daily_messages", start, end, user))
        return [{"day": end, "messages": 4}, {"day": end - timedelta(days=1), "messages": 2}]

    async def threads_by_kind(self, start, end, user=None):
        return [{"kind": "chat", "threads": 3}, {"kind": "quiz", "threads": 1}]

    async def daily_threads_by_kind(self, start, end, user=None):
        return [{"day": end, "kind": "chat", "threads": 3}, {"day": end, "kind": "quiz", "threads": 1}]

    async def messages_by_user(self, start, end):
        return [
            {
                "identifier": "etu1",
                "role": "student",
                "level": "M1",
                "messages": 6,
                "threads": 2,
                "last_message": "2026-10-27 10:00",
            },
            {"identifier": "ghost", "role": "", "level": "", "messages": 1, "threads": 1, "last_message": ""},
        ]

    async def quota_usage(self):
        return [
            {
                "identifier": "etu1",
                "role": "student",
                "level": "M1",
                "quota": 50,
                "used": 50,
                "ratio": 1.0,
                "exhausted": True,
            },
            {
                "identifier": "admin",
                "role": "admin",
                "level": "ALL",
                "quota": None,
                "used": 0,
                "ratio": 0.0,
                "exhausted": False,
            },
        ]

    async def user_identifiers(self):
        return ["admin", "etu1", "prof"]

    # --- pedagogy ------------------------------------------------------------------
    async def quiz_kpis(self, start, end):
        return {"answers": 10, "correct": 7, "quizzes": 2, "students": 1}

    async def quiz_by_topic(self, start, end):
        return [{"topic": "ADF", "course": "UTS", "answers": 10, "correct": 7, "success_pct": 70, "students": 1}]

    async def failed_questions(self, start, end, limit=30):
        return [{"question": "H0 ?", "topic": "ADF", "failures": 3, "students": 1, "last_seen": None}]

    async def exercise_stats(self, start, end):
        return [{"topic": "ADF", "exercises": 2, "attempts": 5, "students": 1, "completed": 1, "correct_steps_pct": 60}]

    async def questions_by_chapter(self, start, end):
        return [{"course": "UTS", "chapter": "Chapitre 3", "questions": 4, "students": 1}]

    async def recent_questions(self, start, end, course=None, limit=50):
        self.calls.append(("recent_questions", course, limit))
        return [
            {
                "created_at": None,
                "user_identifier": "etu1",
                "thread_id": THREAD,
                "course": "UTS",
                "chapter": "Chapitre 3",
                "question": "Le test ADF ?",
                "n_searches": 1,
                "n_sources": 2,
                "general_knowledge": False,
                "refused": False,
            }
        ]

    async def out_of_corpus(self, start, end, limit=50):
        return [
            {
                "created_at": None,
                "user_identifier": "etu1",
                "thread_id": None,
                "question": "Bitcoin ?",
                "reason": "refusée (hors domaine)",
            }
        ]

    async def courses_with_questions(self):
        return ["UTS", "MTS"]

    async def code_runs(self, start, end):
        return [{"status": "ok", "origin": "workshop", "runs": 3, "students": 1, "avg_seconds": 0.4, "figures": 2}]

    async def code_errors(self, start, end, limit=30):
        return [{"error": "NameError: x", "runs": 1, "last_seen": None}]

    async def code_by_student(self, start, end):
        return [{"user_identifier": "etu1", "runs": 3, "ok": 3, "today": 1}]

    # --- feedbacks -----------------------------------------------------------------
    async def feedback_stats(self):
        return FeedbackStats(total=4, positive=3, negative=1)

    async def list_feedbacks(self, start, end, user=None, score=None, limit=50, offset=0):
        self.calls.append(("list_feedbacks", user, score, limit, offset))
        row = {
            "id": FEEDBACK,
            "score": 0,
            "comment": "Trop long",
            "step_id": STEP,
            "answer": "Le test ADF <b>x</b>",
            "answered_at": "2026-10-27 10:00",
            "thread_id": THREAD,
            "thread_name": "Question ADF",
            "identifier": "etu1",
        }
        return FeedbackPage(rows=[row] if offset == 0 else [], total=120)

    async def feedback_context(self, feedback_id, before=3, after=1):
        if feedback_id != FEEDBACK:
            return None
        return {
            "feedback": {
                "id": FEEDBACK,
                "score": 0,
                "comment": "",
                "step_id": STEP,
                "thread_id": THREAD,
                "thread_name": "Question ADF",
                "identifier": "etu1",
            },
            "steps": [
                {
                    "id": "a",
                    "type": "user_message",
                    "content": "Question ?",
                    "created_at": "2026-10-27 09:59",
                    "rated": False,
                },
                {
                    "id": STEP,
                    "type": "assistant_message",
                    "content": "Réponse",
                    "created_at": "2026-10-27 10:00",
                    "rated": True,
                },
            ],
        }

    # --- conversations -------------------------------------------------------------
    async def list_threads(self, start, end, user=None, kind=None, limit=50, offset=0):
        self.calls.append(("list_threads", user, kind, limit, offset))
        row = {
            "id": THREAD,
            "name": "🎓 Quiz : ADF",
            "identifier": "etu1",
            "created_at": "2026-10-27 10:00",
            "kind": "quiz",
            "steps": 4,
            "messages": 2,
            "feedbacks": 1,
        }
        return ThreadPage(rows=[row], total=1)

    async def get_thread(self, thread_id):
        if thread_id != THREAD:
            return None
        return {
            "id": THREAD,
            "name": "🎓 Quiz : ADF",
            "identifier": "etu1",
            "created_at": "2026-10-27 10:00",
            "kind": "quiz",
        }

    async def thread_steps(self, thread_id, include_tools=False):
        self.calls.append(("thread_steps", thread_id, include_tools))
        steps = [
            {
                "id": "a",
                "type": "user_message",
                "name": "m",
                "content": "Question <script>",
                "created_at": "2026-10-27 10:00",
                "is_error": False,
                "feedback": None,
                "feedback_comment": "",
            },
            {
                "id": STEP,
                "type": "assistant_message",
                "name": "a",
                "content": "Réponse [S1]",
                "created_at": "2026-10-27 10:01",
                "is_error": False,
                "feedback": 1,
                "feedback_comment": "Merci",
            },
        ]
        if include_tools:
            steps.insert(
                1,
                {
                    "id": "t",
                    "type": "tool",
                    "name": "🔎 Recherche",
                    "content": "5 extraits",
                    "created_at": "2026-10-27 10:00",
                    "is_error": False,
                    "feedback": None,
                    "feedback_comment": "",
                },
            )
        return steps


@pytest.fixture
def base_db(monkeypatch: pytest.MonkeyPatch) -> FakeDb:
    import ela.db.audit as audit_mod
    import ela.db.dashboard as dashboard_mod
    import ela.db.users as users_mod

    db = FakeDb()
    monkeypatch.setattr(users_mod, "authenticate", db.authenticate)
    monkeypatch.setattr(users_mod, "get_profile", db.get_profile)
    monkeypatch.setattr(audit_mod, "record", db.record)
    monkeypatch.setattr(audit_mod, "recent", db.recent)
    monkeypatch.setattr(dashboard_mod, "overview", db.overview_fn)
    return db


@pytest.fixture
def activity_db(base_db, monkeypatch: pytest.MonkeyPatch) -> FakeActivityDb:
    import ela.db.conversations as conv_mod
    import ela.db.feedbacks as fb_mod
    import ela.db.pedagogy as ped_mod
    import ela.db.stats as stats_mod

    db = FakeActivityDb()
    for name in (
        "daily_messages",
        "threads_by_kind",
        "daily_threads_by_kind",
        "messages_by_user",
        "quota_usage",
        "user_identifiers",
    ):
        monkeypatch.setattr(stats_mod, name, getattr(db, name))
    for name in (
        "quiz_kpis", "quiz_by_topic", "failed_questions", "exercise_stats", "questions_by_chapter", "recent_questions",
        "out_of_corpus", "courses_with_questions", "code_runs", "code_errors", "code_by_student",
    ):  # fmt: skip
        monkeypatch.setattr(ped_mod, name, getattr(db, name))
    monkeypatch.setattr(fb_mod, "stats", db.feedback_stats)
    monkeypatch.setattr(fb_mod, "list_feedbacks", db.list_feedbacks)
    monkeypatch.setattr(fb_mod, "context", db.feedback_context)
    for name in ("list_threads", "get_thread", "thread_steps"):
        monkeypatch.setattr(conv_mod, name, getattr(db, name))
    return db


@pytest.fixture
def app(activity_db):
    from backoffice.main import create_app

    return create_app(make_settings())


@pytest.fixture
async def client(app):
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", follow_redirects=False) as c:
        yield c


# --- access ------------------------------------------------------------------------


async def test_pages_require_login(client):
    for path in (
        "/activity",
        "/pedagogy",
        "/feedbacks",
        "/conversations",
        f"/conversations/{THREAD}",
        "/activity/users.csv",
    ):
        r = await client.get(f"{PREFIX}{path}")
        assert r.status_code == 303 and r.headers["location"].startswith(f"{PREFIX}/login")


async def test_supervisor_can_read_everything(client):
    await login(client, "prof")
    for path in (
        "/activity",
        "/pedagogy?tab=code",
        "/feedbacks",
        "/conversations",
        f"/conversations/{THREAD}",
        "/feedbacks.csv",
    ):
        r = await client.get(f"{PREFIX}{path}")
        assert r.status_code == 200, path


# --- activity ----------------------------------------------------------------------


async def test_activity_page_and_filters(client, activity_db):
    await login(client, "admin")
    r = await client.get(f"{PREFIX}/activity")
    assert r.status_code == 200
    assert "30 derniers jours" in r.text and 'class="chart chart-line"' in r.text and "chart-donut" in r.text
    assert "chart-area" in r.text and "chart-bars" in r.text
    assert "etu1" in r.text and "compte supprimé" in r.text and "quota atteint" in r.text and "illimité" in r.text
    assert 'href="/backoffice/activity/users.csv?from=' in r.text
    start, end, user = activity_db.calls[0][1:]
    assert user is None and (end - start).days == 30

    r = await client.get(f"{PREFIX}/activity?days=7&user=etu1")
    assert r.status_code == 200 and "7 derniers jours" in r.text and 'value="etu1" selected' in r.text
    call = [c for c in activity_db.calls if c[0] == "daily_messages"][-1]
    assert call[3] == "etu1" and (call[2] - call[1]).days == 7

    r = await client.get(f"{PREFIX}/activity?from=2026-10-20&to=2026-10-01")
    assert r.status_code == 200 and "doit précéder" in r.text
    r = await client.get(f"{PREFIX}/activity?from=2026-09-01&to=2026-09-15")
    assert "du 01/09/2026 au 15/09/2026" in r.text


async def test_activity_csv_exports(client):
    await login(client, "admin")
    r = await client.get(f"{PREFIX}/activity/users.csv?from=2026-10-01&to=2026-10-27")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert r.headers["content-disposition"] == 'attachment; filename="ela-utilisateurs-2026-10-01-2026-10-27.csv"'
    lines = r.text.lstrip("﻿").splitlines()
    assert lines[0] == "Identifiant;Rôle;Niveau;Messages;Conversations;Dernier message"
    assert lines[1] == "etu1;student;M1;6;2;2026-10-27 10:00"
    r = await client.get(f"{PREFIX}/activity/daily.csv?days=7")
    lines = r.text.lstrip("﻿").splitlines()
    assert lines[0].startswith("Jour;Messages;Conversations Quiz;") and len(lines) == 1 + 8
    assert lines[-1].startswith(date.today().isoformat() + ";4;")
    r = await client.get(f"{PREFIX}/activity/quotas.csv")
    assert "admin;admin;ALL;;0" in r.text


# --- pedagogy ----------------------------------------------------------------------


async def test_pedagogy_tabs(client, activity_db):
    await login(client, "admin")
    r = await client.get(f"{PREFIX}/pedagogy")
    assert r.status_code == 200 and "70 %" in r.text and "H0 ?" in r.text and "Réussite par sujet" in r.text
    r = await client.get(f"{PREFIX}/pedagogy?tab=exercises")
    assert "Exercices commencés" in r.text and "60 %" in r.text
    r = await client.get(f"{PREFIX}/pedagogy?tab=questions&course=MTS")
    assert "Chapitre 3" in r.text and 'value="MTS" selected' in r.text and f"/conversations/{THREAD}" in r.text
    assert ("recent_questions", "MTS", 50) in activity_db.calls
    r = await client.get(f"{PREFIX}/pedagogy?tab=corpus")
    assert "Bitcoin ?" in r.text and "refusée (hors domaine)" in r.text
    r = await client.get(f"{PREFIX}/pedagogy?tab=code")
    assert "NameError: x" in r.text and "Par statut et origine" in r.text
    r = await client.get(f"{PREFIX}/pedagogy?tab=nope")
    assert r.status_code == 200 and "Réussite par sujet" in r.text


async def test_pedagogy_csv(client, activity_db):
    await login(client, "admin")
    r = await client.get(f"{PREFIX}/pedagogy/quiz-sujets.csv?days=30")
    assert (
        r.status_code == 200
        and "Sujet;Cours;Réponses;Correctes;Réussite %;Étudiants" in r.text
        and "ADF;UTS;10;7;70;1" in r.text
    )
    r = await client.get(f"{PREFIX}/pedagogy/questions-recentes.csv?course=UTS")
    assert r.status_code == 200 and "Le test ADF ?" in r.text and ("recent_questions", "UTS", 5000) in activity_db.calls
    r = await client.get(f"{PREFIX}/pedagogy/code-erreurs.csv")
    assert "NameError: x;1;" in r.text
    r = await client.get(f"{PREFIX}/pedagogy/unknown.csv")
    assert r.status_code == 404


# --- feedbacks ---------------------------------------------------------------------


async def test_feedbacks_list_context_and_paging(client, activity_db):
    await login(client, "admin")
    r = await client.get(f"{PREFIX}/feedbacks?score=0&user=etu1")
    assert r.status_code == 200
    assert "Trop long" in r.text and "Le test ADF &lt;b&gt;x&lt;/b&gt;" in r.text and "<b>x</b>" not in r.text
    assert "Page 1 / 3" in r.text and "page=2" in r.text and 'value="0" selected' in r.text
    assert ("list_feedbacks", "etu1", 0, 50, 0) in activity_db.calls
    assert f'hx-get="{PREFIX}/feedbacks/{FEEDBACK}/context"' in r.text

    r = await client.get(f"{PREFIX}/feedbacks?page=2")
    assert "Page 2 / 3" in r.text and ("list_feedbacks", None, None, 50, 50) in activity_db.calls

    r = await client.get(f"{PREFIX}/feedbacks/{FEEDBACK}/context", headers={"HX-Request": "true"})
    assert r.status_code == 200 and "bubble-rated" in r.text and "Question ?" in r.text and "<html" not in r.text
    r = await client.get(f"{PREFIX}/feedbacks/nope/context")
    assert r.status_code == 404 and "Contexte non disponible" in r.text


async def test_feedbacks_csv(client):
    await login(client, "admin")
    r = await client.get(f"{PREFIX}/feedbacks.csv?score=0")
    assert r.status_code == 200
    lines = r.text.lstrip("﻿").splitlines()
    assert lines[0] == "Date;Étudiant;Conversation;Score;Commentaire;Réponse d'ELA;Id conversation"
    assert lines[1].startswith("2026-10-27 10:00;etu1;Question ADF;0;Trop long;")


# --- conversations -----------------------------------------------------------------


async def test_conversations_list_and_reader(client, activity_db):
    await login(client, "admin")
    r = await client.get(f"{PREFIX}/conversations?kind=quiz&user=etu1")
    assert r.status_code == 200 and "🎓 Quiz : ADF" in r.text and f"/conversations/{THREAD}" in r.text
    assert ("list_threads", "etu1", "quiz", 50, 0) in activity_db.calls
    r = await client.get(f"{PREFIX}/conversations?kind=bogus")
    assert ("list_threads", None, None, 50, 0) in activity_db.calls

    r = await client.get(f"{PREFIX}/conversations/{THREAD}")
    assert r.status_code == 200
    assert "Question &lt;script&gt;" in r.text and "<script>" not in r.text.split("</head>")[1]
    assert "Réponse [S1]" in r.text and "Merci" in r.text and "🔎 Recherche" not in r.text
    assert ("thread_steps", THREAD, False) in activity_db.calls
    r = await client.get(f"{PREFIX}/conversations/{THREAD}?tools=1")
    assert "🔎 Recherche" in r.text and "bubble-tool" in r.text

    r = await client.get(f"{PREFIX}/conversations/not-a-thread")
    assert r.status_code == 404 and "introuvable" in r.text

    r = await client.get(f"{PREFIX}/conversations.csv?kind=quiz")
    lines = r.text.lstrip("﻿").splitlines()
    assert lines[0] == "Date;Étudiant;Type;Nom;Messages;Étapes;Feedbacks;Id"
    assert lines[1] == f"2026-10-27 10:00;etu1;Quiz;🎓 Quiz : ADF;2;4;1;{THREAD}"


async def test_home_and_account_link_to_the_new_pages(client, base_db):
    from ela.db.dashboard import Overview

    base_db.overview = Overview(
        threads=1, recent_threads=[{"id": THREAD, "name": "Fil", "identifier": "etu1", "messages": 1, "created_at": ""}]
    )
    await login(client, "admin")
    r = await client.get(f"{PREFIX}/")
    assert f'href="{PREFIX}/conversations/{THREAD}"' in r.text
    assert f'href="{PREFIX}/activity"' in r.text  # navigation entry replaces the "bientôt" placeholder


# --- pure helpers ------------------------------------------------------------------


def test_parse_period():
    from backoffice.periods import parse_period

    p = parse_period(today=TODAY)
    assert (p.start, p.end, p.days, p.error) == (TODAY - timedelta(days=30), TODAY, 30, None)
    assert p.query == "from=2026-09-27&to=2026-10-27" and p.span == 31 and p.label == "30 derniers jours"
    assert parse_period(days="7", today=TODAY).days == 7
    assert parse_period(days="12", today=TODAY).error and parse_period(days="x", today=TODAY).days == 30
    p = parse_period("2026-10-01", "2026-10-10", today=TODAY)
    assert (p.start, p.end, p.days) == (
        date(2026, 10, 1),
        date(2026, 10, 10),
        None,
    ) and p.label == "du 01/10/2026 au 10/10/2026"
    assert parse_period("2026-09-27", "2026-10-27", today=TODAY).days == 30  # recognised preset
    assert "précéder" in parse_period("2026-10-10", "2026-10-01", today=TODAY).error
    assert "invalides" in parse_period("2026-13-01", "2026-10-01", today=TODAY).error
    assert "trop longue" in parse_period("2020-01-01", "2026-10-01", today=TODAY).error


def test_paging():
    from backoffice.paging import paging

    pg = paging("3").with_total(120)
    assert (pg.page, pg.offset, pg.pages, pg.first, pg.last, pg.has_prev, pg.has_next) == (
        3,
        100,
        3,
        101,
        120,
        True,
        False,
    )
    assert paging(None).page == 1 and paging("-4").page == 1 and paging("x").page == 1
    empty = paging("1").with_total(0)
    assert (empty.pages, empty.first, empty.last) == (1, 0, 0)


def test_classify_thread_and_sql():
    from ela.db.threads import THREAD_KIND_SQL, classify_thread

    assert classify_thread(None) == "chat" and classify_thread("Question ADF") == "chat"
    assert classify_thread("🎓 Quiz : ADF") == "quiz" and classify_thread("✏️ Exercice : GARCH") == "exercise"
    assert classify_thread("📄 Fiche de révision") == "sheet" and classify_thread("💻 Atelier Code") == "code"
    assert THREAD_KIND_SQL.startswith("CASE WHEN t.name LIKE '🎓%' THEN 'quiz'") and THREAD_KIND_SQL.endswith(
        "ELSE 'chat' END"
    )


def test_csv_text():
    from backoffice.csvexport import csv_response, csv_text

    text = csv_text(
        [("a", "A"), ("b", "B;b")], [{"a": 'x"y', "b": 1.5}, {"a": None, "b": True}, {"a": date(2026, 1, 2)}]
    )
    assert text == 'A;"B;b"\r\n"x""y";1,5\r\n;oui\r\n2026-01-02;\r\n'
    r = csv_response("Été 2026/export", [("a", "A")], [])
    assert r.headers["content-disposition"] == 'attachment; filename="ela-t-2026-export.csv"'


def test_charts():
    from backoffice import charts

    svg = charts.line_chart(["01/10", "02/10", "03/10"], [0, 5, 12], unit="messages")
    assert (
        svg.startswith('<svg class="chart chart-line"')
        and svg.count("<circle") == 3
        and "<title>03/10 : 12 messages</title>" in svg
    )
    assert 'points="' in svg and ">15<" in svg  # nice upper tick above 12
    assert charts.line_chart([], []) == ""
    svg = charts.stacked_area_chart(["a", "b"], [("chat", "Chat", [1, 2]), ("quiz", "Quiz", [0, 3])], unit="conv.")
    assert svg.count("<polygon") == 2 and 'class="area s-quiz"' in svg and "Quiz — b : 3 conv." in svg
    svg = charts.donut_chart([("chat", "Chat", 3), ("quiz", "Quiz", 1), ("code", "Code", 0)], unit="conversations")
    assert svg.count("<path") == 2 and "(75 %)" in svg and 'class="donut-total"' in svg and ">4<" in svg
    assert charts.donut_chart([("chat", "Chat", 5)]).count("<path") == 1  # full ring
    assert charts.donut_chart([("chat", "Chat", 0)]) == ""
    svg = charts.hbar_chart([("etu1", 10), ("a-very-long-identifier-that-is-truncated", 2)], unit="messages")
    assert svg.count("<rect") == 2 and "a-very-long-identifier-th…" in svg and 'viewBox="0 0 720 64"' in svg
    assert "<script" not in charts.hbar_chart([("<script>", 1)]) and "&lt;script&gt;" in charts.hbar_chart(
        [("<script>", 1)]
    )
    assert (
        charts._nice_ticks(0) == [0.0, 1.0]
        and charts._nice_ticks(12) == [0, 5, 10, 15]
        and charts._nice_ticks(3) == [0, 1, 2, 3]
    )
    assert charts.day_labels([date(2026, 10, 5)]) == ["05/10"]
