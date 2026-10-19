"""Tests for the student report (offline) and the activity table (PostgreSQL)."""

from __future__ import annotations

import os
import uuid

import pytest

from ela.db.progress import StudentReport, build_suggestions, render_report
from ela.db.quiz import QuizStats

TEST_URL = os.getenv("TEST_DATABASE_URL")


def _report(**kw) -> StudentReport:
    base = dict(
        identifier="u",
        activity={"questions": 0, "refused": 0, "general": 0, "threads": 0},
        chapters=[],
        quiz=QuizStats(0, 0, 0, []),
        exercises=[],
    )
    base.update(kw)
    return StudentReport(**base)


def test_empty_report():
    r = _report()
    assert r.empty and "Rien à montrer" in render_report(r) and build_suggestions(r) == []


def test_suggestions_order_and_dedup():
    r = _report(
        activity={"questions": 3, "refused": 0, "general": 0, "threads": 1},
        chapters=[
            {"course": "UTS", "chapter": "Racines unitaires", "questions": 2},
            {"course": "MTS", "chapter": "VAR", "questions": 1},
        ],
        quiz=QuizStats(
            5,
            2,
            2,
            [
                {"topic": "VAR", "n": 3, "correct": 1, "rate": 1 / 3},
                {"topic": "ADF", "n": 1, "correct": 0, "rate": 0.0},
            ],
        ),
        exercises=[{"exercise_id": "e", "topic": "VAR", "title": "VAR(1)", "attempts": 1, "completed": False}],
    )
    s = build_suggestions(r)
    assert [(x.kind, x.topic) for x in s] == [("quiz", "VAR"), ("exercise", "VAR"), ("quiz", "Racines unitaires")]
    text = render_report(r)
    assert "Points faibles : VAR (33 %)" in text and "ADF" not in text.split("Points faibles")[1].split("\n")[0]
    assert "**Et maintenant ?** 🎓 Quiz : VAR" in text and "UTS › Racines unitaires — 2 question(s)" in text


@pytest.mark.skipif(not TEST_URL, reason="TEST_DATABASE_URL not set")
async def test_activity_roundtrip_and_report(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", TEST_URL)
    from ela.config import get_settings
    from ela.db import engine
    from ela.db.activity import QuestionRecord, activity_counts, chapters_worked, record_question
    from ela.db.migrate import upgrade
    from ela.db.progress import student_report

    get_settings.cache_clear()
    upgrade()
    user = f"pytest_act_{uuid.uuid4().hex[:6]}"
    try:
        assert await record_question(
            QuestionRecord(
                user,
                "ADF ?",
                thread_id="t",
                level="M1",
                course="UTS",
                chapter="Ch 2",
                n_searches=1,
                n_sources=2,
                cited=["a"],
            )
        )
        assert await record_question(QuestionRecord(user, "ADF encore ?", course="UTS", chapter="Ch 2", n_searches=1))
        assert await record_question(QuestionRecord(user, "Foot ?", refused=True))
        assert await record_question(
            QuestionRecord(user, "KPSS ?", course="UTS", chapter="Ch 2", general_knowledge=True)
        )
        counts = await activity_counts(user)
        assert counts == {"questions": 4, "refused": 1, "general": 1, "threads": 1}
        chapters = await chapters_worked(user)
        assert chapters[0]["course"] == "UTS" and chapters[0]["questions"] == 3 and chapters[0]["quizzes"] == 0
        report = await student_report(user)
        assert not report.empty and "Chapitres travaillés" in render_report(report)
        assert [s.label for s in report.suggestions] == ["🎓 Me tester : Ch 2"]

        # A quiz built from another course counts as a chapter worked (quizzes, not questions)
        from ela.db.quiz import QuizAnswer, record_answer

        for i in range(2):
            assert await record_answer(
                QuizAnswer(user, "qz1", "les garch", i, f"q{i}", ["a", "b"], 0, 0, course="HFE", chapter="Volatilité")
            )
        chapters = await chapters_worked(user)
        hfe = next(c for c in chapters if c["course"] == "HFE")
        assert hfe["chapter"] == "Volatilité" and hfe["quizzes"] == 1 and hfe["questions"] == 0
        report = await student_report(user)
        text = render_report(report)
        assert "HFE › Volatilité — 1 quiz" in text and "UTS › Ch 2 — 3 question(s)" in text
        assert "Me tester : Volatilité" not in text  # already quizzed
    finally:
        await engine.execute("DELETE FROM student_activity WHERE user_identifier = :u", {"u": user})
        await engine.execute("DELETE FROM quiz_results WHERE user_identifier = :u", {"u": user})
        await engine.dispose()


@pytest.mark.skipif(not TEST_URL, reason="TEST_DATABASE_URL not set")
async def test_review_sheet_cache_roundtrip(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", TEST_URL)
    from ela.config import get_settings
    from ela.db import engine
    from ela.db.migrate import upgrade
    from ela.db.review_sheets import get_cached_sheet, store_sheet
    from ela.pedagogy.schemas import ReviewSection, ReviewSheet

    get_settings.cache_clear()
    upgrade()
    course = f"pytest_course_{uuid.uuid4().hex[:6]}"
    sheet = ReviewSheet(
        title="T",
        objectives=["o"],
        sections=[ReviewSection(heading="h", key_points=["$x$"])],
        takeaways=["t"],
        course=course,
        chapter="Ch 1",
        level="M1",
        slides=["s1"],
    )
    key = {"prompt_version": "v1", "store_fingerprint": "fp-1"}
    try:
        assert await get_cached_sheet(course, "Ch 1", "M1", **key) is None
        assert await store_sheet(sheet, **key)
        cached = await get_cached_sheet(course, "Ch 1", "M1", **key)
        assert cached is not None and cached.sheet_id == sheet.sheet_id and cached.sections[0].key_points == ["$x$"]
        assert await get_cached_sheet(course, "Ch 1", "M1", prompt_version="v1", store_fingerprint="fp-2") is None
        sheet2 = sheet.model_copy(update={"title": "T2"})
        assert await store_sheet(sheet2, **key)  # replace in place
        assert (await get_cached_sheet(course, "Ch 1", "M1", **key)).title == "T2"
    finally:
        await engine.execute("DELETE FROM review_sheets WHERE course = :c", {"c": course})
        await engine.dispose()


@pytest.mark.skipif(not TEST_URL, reason="TEST_DATABASE_URL not set")
async def test_code_runs_quota_roundtrip(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", TEST_URL)
    from ela.config import get_settings
    from ela.db import engine
    from ela.db.code_runs import CodeRun, record_run, runs_today
    from ela.db.migrate import upgrade

    get_settings.cache_clear()
    upgrade()
    user = f"pytest_code_{uuid.uuid4().hex[:6]}"
    try:
        assert await runs_today(user) == 0
        assert await record_run(CodeRun(user, "ok", "workshop", "AR(1)", elapsed_ms=120, n_images=1, thread_id="t"))
        assert await record_run(CodeRun(user, "error", "agent", error="ZeroDivisionError"))
        assert await record_run(CodeRun(user, "unavailable", "workshop", error="down"))  # not counted
        assert await runs_today(user) == 2
    finally:
        await engine.execute("DELETE FROM code_runs WHERE user_identifier = :u", {"u": user})
        await engine.dispose()
