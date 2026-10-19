"""Tests for ``ela.db.quiz`` (difficulty helper offline; persistence against PostgreSQL)."""

from __future__ import annotations

import os
import uuid

import pytest

from ela.db.quiz import QuizAnswer, next_difficulty

TEST_URL = os.getenv("TEST_DATABASE_URL")


def test_next_difficulty():
    assert next_difficulty("moyen", 3, 3) == "difficile"
    assert next_difficulty("difficile", 3, 3) == "difficile"
    assert next_difficulty("moyen", 1, 3) == "facile"
    assert next_difficulty("facile", 0, 3) == "facile"
    assert next_difficulty("moyen", 2, 3) == "moyen"
    assert next_difficulty("moyen", 0, 0) == "moyen" and next_difficulty("xx", 2, 3) == "moyen"


def test_quiz_answer_correct_flag():
    a = QuizAnswer("u", "q", "t", 0, "Q", ["a", "b"], correct_index=1, answer_index=1)
    assert a.correct and not QuizAnswer("u", "q", "t", 0, "Q", ["a", "b"], 1, 0).correct


@pytest.mark.skipif(not TEST_URL, reason="TEST_DATABASE_URL not set")
async def test_record_and_read_answers(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", TEST_URL)
    from ela.config import get_settings
    from ela.db import engine
    from ela.db.migrate import upgrade
    from ela.db.quiz import recent_errors, record_answer, user_stats

    get_settings.cache_clear()
    upgrade()
    user = f"pytest_quiz_{uuid.uuid4().hex[:6]}"
    quiz_id = uuid.uuid4().hex
    try:
        base = dict(user_identifier=user, quiz_id=quiz_id, topic="ADF", difficulty="moyen", level="M1", course="UTS")
        assert await record_answer(
            QuizAnswer(question_index=0, question="Q0", options=["a", "b"], correct_index=0, answer_index=0, **base)
        )
        assert await record_answer(
            QuizAnswer(
                question_index=1,
                question="Q1",
                options=["a", "b"],
                correct_index=1,
                answer_index=0,
                sources=["UTS › ADF"],
                **base,
            )
        )
        assert await record_answer(
            QuizAnswer(
                question_index=0,
                question="K0",
                options=["a", "b"],
                correct_index=0,
                answer_index=1,
                **{**base, "quiz_id": "other", "topic": "KPSS"},
            )
        )

        errors = await recent_errors(user)
        assert [e["question"] for e in errors] == ["K0", "Q1"] and errors[1]["sources"] == ["UTS › ADF"]
        assert [e["question"] for e in await recent_errors(user, quiz_id=quiz_id)] == ["Q1"]

        stats = await user_stats(user)
        assert (stats.n_answers, stats.n_correct, stats.n_quizzes) == (3, 1, 2) and stats.rate == pytest.approx(1 / 3)
        assert stats.by_topic[0]["topic"] == "KPSS" and stats.by_topic[0]["rate"] == 0.0
        assert stats.by_topic[1] == {"topic": "ADF", "n": 2, "correct": 1, "rate": 0.5}
        assert (await user_stats("nobody")).n_answers == 0
    finally:
        await engine.execute("DELETE FROM quiz_results WHERE user_identifier = :u", {"u": user})
        await engine.dispose()


@pytest.mark.skipif(not TEST_URL, reason="TEST_DATABASE_URL not set")
async def test_record_exercise_attempts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", TEST_URL)
    from ela.config import get_settings
    from ela.db import engine
    from ela.db.exercises import ExerciseAttempt, exercise_summary, record_attempt
    from ela.db.migrate import upgrade

    get_settings.cache_clear()
    upgrade()
    user = f"pytest_ex_{uuid.uuid4().hex[:6]}"
    try:
        base = dict(user_identifier=user, exercise_id="e1", topic="AR(1)", title="AR(1)", level="M1")
        assert await record_attempt(ExerciseAttempt(attempt_index=1, attempt="a", verdict="partiel", **base))
        assert await record_attempt(
            ExerciseAttempt(attempt_index=2, attempt="b", verdict="correct", completed=True, hints_revealed=1, **base)
        )
        rows = await exercise_summary(user)
        assert len(rows) == 1 and rows[0]["attempts"] == 2 and rows[0]["completed"] and rows[0]["hints_revealed"] == 1
    finally:
        await engine.execute("DELETE FROM exercise_attempts WHERE user_identifier = :u", {"u": user})
        await engine.dispose()
