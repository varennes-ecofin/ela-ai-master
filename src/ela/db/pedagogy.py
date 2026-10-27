"""Pedagogy figures for the backoffice (what the students work on and where they struggle).

Port of the Streamlit *Pédagogie* page on the tables fed by the chat:
``quiz_results`` (one row per answered question), ``exercise_attempts``
(Socratic mode), ``student_activity`` (one row per agent answer) and
``code_runs`` (Atelier Code). Every function takes an inclusive period of
calendar days and returns plain rows; a database error gives empty data.
"""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from typing import Any

from ela.db.engine import fetch_all, fetch_one

logger = logging.getLogger(__name__)


def _clean(row: dict[str, Any]) -> dict[str, Any]:
    """Turn PostgreSQL numerics into plain Python numbers and ``None`` counts into 0."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, Decimal):
            value = int(value) if value == value.to_integral_value() else float(value)
        out[key] = value
    return out


async def _rows(sql: str, params: dict[str, Any], what: str) -> list[dict[str, Any]]:
    try:
        return [_clean(r) for r in await fetch_all(sql, params)]
    except Exception:
        logger.warning("pedagogy.%s failed", what, exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Quiz
# ---------------------------------------------------------------------------


async def quiz_kpis(start: date, end: date) -> dict[str, int]:
    """Count answers, correct answers, distinct quizzes and students over the period."""
    try:
        row = await fetch_one(
            """
            SELECT count(*) AS answers, count(*) FILTER (WHERE correct) AS correct,
                   count(DISTINCT quiz_id) AS quizzes, count(DISTINCT user_identifier) AS students
            FROM quiz_results WHERE answered_at::date BETWEEN :start AND :end
            """,
            {"start": start, "end": end},
        )
    except Exception:
        logger.warning("pedagogy.quiz_kpis failed", exc_info=True)
        row = None
    return {k: int((row or {}).get(k) or 0) for k in ("answers", "correct", "quizzes", "students")}


async def quiz_by_topic(start: date, end: date) -> list[dict[str, Any]]:
    """Return the success rate per quiz topic, weakest first.

    Rows ``topic``, ``course``, ``answers``, ``correct``, ``success_pct``, ``students``.
    """
    return await _rows(
        """
        SELECT topic, course, count(*) AS answers, count(*) FILTER (WHERE correct) AS correct,
               round(100.0 * count(*) FILTER (WHERE correct) / count(*), 0) AS success_pct,
               count(DISTINCT user_identifier) AS students
        FROM quiz_results WHERE answered_at::date BETWEEN :start AND :end
        GROUP BY topic, course ORDER BY success_pct ASC, answers DESC
        """,
        {"start": start, "end": end},
        "quiz_by_topic",
    )


async def failed_questions(start: date, end: date, limit: int = 30) -> list[dict[str, Any]]:
    """Return the quiz questions most often answered wrongly.

    Rows ``question``, ``topic``, ``failures``, ``students``, ``last_seen``.
    """
    return await _rows(
        """
        SELECT question, topic, count(*) AS failures,
               count(DISTINCT user_identifier) AS students, max(answered_at) AS last_seen
        FROM quiz_results WHERE answered_at::date BETWEEN :start AND :end AND NOT correct
        GROUP BY question, topic ORDER BY failures DESC, last_seen DESC LIMIT :limit
        """,
        {"start": start, "end": end, "limit": limit},
        "failed_questions",
    )


# ---------------------------------------------------------------------------
# Exercises
# ---------------------------------------------------------------------------


async def exercise_stats(start: date, end: date) -> list[dict[str, Any]]:
    """Return the exercises per topic.

    Rows ``topic``, ``exercises``, ``attempts``, ``students``, ``completed``, ``correct_steps_pct``.
    """
    return await _rows(
        """
        SELECT topic, count(DISTINCT exercise_id) AS exercises, count(*) AS attempts,
               count(DISTINCT user_identifier) AS students,
               count(DISTINCT exercise_id) FILTER (WHERE completed) AS completed,
               round(100.0 * count(*) FILTER (WHERE verdict = 'correct') / count(*), 0) AS correct_steps_pct
        FROM exercise_attempts WHERE created_at::date BETWEEN :start AND :end
        GROUP BY topic ORDER BY attempts DESC
        """,
        {"start": start, "end": end},
        "exercise_stats",
    )


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------


async def questions_by_chapter(start: date, end: date) -> list[dict[str, Any]]:
    """Return the answered questions per course and chapter (``course``, ``chapter``, ``questions``, ``students``)."""
    return await _rows(
        """
        SELECT coalesce(course, '(hors cours)') AS course, coalesce(chapter, '—') AS chapter,
               count(*) AS questions, count(DISTINCT user_identifier) AS students
        FROM student_activity WHERE created_at::date BETWEEN :start AND :end AND NOT refused
        GROUP BY course, chapter ORDER BY questions DESC
        """,
        {"start": start, "end": end},
        "questions_by_chapter",
    )


async def recent_questions(start: date, end: date, course: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Return the latest questions, optionally for one course.

    Rows ``created_at``, ``user_identifier``, ``thread_id``, ``course``, ``chapter``, ``question`` (truncated),
    ``n_searches``, ``n_sources``, ``general_knowledge``, ``refused``.
    """
    params: dict[str, Any] = {"start": start, "end": end, "limit": limit}
    course_clause = ""
    if course:
        course_clause = " AND course = :course"
        params["course"] = course
    return await _rows(
        f"""
        SELECT created_at, user_identifier, thread_id, course, chapter, left(question, 160) AS question,
               n_searches, n_sources, general_knowledge, refused
        FROM student_activity WHERE created_at::date BETWEEN :start AND :end{course_clause}
        ORDER BY created_at DESC LIMIT :limit
        """,
        params,
        "recent_questions",
    )


async def out_of_corpus(start: date, end: date, limit: int = 50) -> list[dict[str, Any]]:
    """Return the questions refused, answered from general knowledge or without any cited passage.

    ``reason`` is a French label of the case.
    """
    return await _rows(
        """
        SELECT created_at, user_identifier, thread_id, left(question, 160) AS question,
               CASE WHEN refused THEN 'refusée (hors domaine)'
                    WHEN general_knowledge THEN 'connaissances générales'
                    ELSE 'aucun extrait cité' END AS reason
        FROM student_activity
        WHERE created_at::date BETWEEN :start AND :end AND (refused OR general_knowledge OR n_sources = 0)
        ORDER BY created_at DESC LIMIT :limit
        """,
        {"start": start, "end": end, "limit": limit},
        "out_of_corpus",
    )


async def courses_with_questions() -> list[str]:
    """Return the courses that appear in ``student_activity`` (for the course filter)."""
    rows = await _rows(
        "SELECT DISTINCT course FROM student_activity WHERE course IS NOT NULL ORDER BY course", {}, "courses"
    )
    return [r["course"] for r in rows]


# ---------------------------------------------------------------------------
# Code workshop
# ---------------------------------------------------------------------------


async def code_runs(start: date, end: date) -> list[dict[str, Any]]:
    """Return the script executions by status and origin.

    Rows ``status``, ``origin``, ``runs``, ``students``, ``avg_seconds``, ``figures``.
    """
    return await _rows(
        """
        SELECT status, origin, count(*) AS runs, count(DISTINCT user_identifier) AS students,
               round(avg(elapsed_ms) / 1000.0, 1) AS avg_seconds, coalesce(sum(n_images), 0) AS figures
        FROM code_runs WHERE created_at::date BETWEEN :start AND :end
        GROUP BY status, origin ORDER BY runs DESC
        """,
        {"start": start, "end": end},
        "code_runs",
    )


async def code_errors(start: date, end: date, limit: int = 30) -> list[dict[str, Any]]:
    """Return the most frequent execution errors (``error`` truncated, ``runs``, ``last_seen``)."""
    return await _rows(
        """
        SELECT left(error, 160) AS error, count(*) AS runs, max(created_at) AS last_seen
        FROM code_runs WHERE created_at::date BETWEEN :start AND :end
          AND status IN ('error', 'timeout') AND error IS NOT NULL
        GROUP BY left(error, 160) ORDER BY runs DESC, last_seen DESC LIMIT :limit
        """,
        {"start": start, "end": end, "limit": limit},
        "code_errors",
    )


async def code_by_student(start: date, end: date) -> list[dict[str, Any]]:
    """Return the executions per student (``user_identifier``, ``runs``, ``ok``, ``today``)."""
    return await _rows(
        """
        SELECT user_identifier, count(*) AS runs, count(*) FILTER (WHERE status = 'ok') AS ok,
               count(*) FILTER (WHERE created_at >= date_trunc('day', now())) AS today
        FROM code_runs WHERE created_at::date BETWEEN :start AND :end
        GROUP BY user_identifier ORDER BY runs DESC
        """,
        {"start": start, "end": end},
        "code_by_student",
    )
