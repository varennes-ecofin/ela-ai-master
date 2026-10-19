"""Student activity: one row per agent answer (table ``student_activity``)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from ela.db.engine import execute, fetch_all

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuestionRecord:
    """One student question answered by the agent."""

    user_identifier: str
    question: str
    thread_id: str | None = None
    level: str | None = None
    course: str | None = None
    chapter: str | None = None
    n_searches: int = 0
    n_sources: int = 0
    cited: list[str] = field(default_factory=list)
    general_knowledge: bool = False
    refused: bool = False


async def record_question(rec: QuestionRecord) -> bool:
    """Store one answered question (returns ``False`` instead of raising on failure)."""
    try:
        await execute(
            """
            INSERT INTO student_activity (
                user_identifier, thread_id, level, question, course, chapter,
                n_searches, n_sources, cited, general_knowledge, refused
            ) VALUES (
                :user, :thread, :level, :question, :course, :chapter,
                :n_searches, :n_sources, CAST(:cited AS jsonb), :general, :refused
            )
            """,
            {
                "user": rec.user_identifier,
                "thread": rec.thread_id,
                "level": rec.level,
                "question": rec.question[:2000],
                "course": rec.course,
                "chapter": rec.chapter,
                "n_searches": rec.n_searches,
                "n_sources": rec.n_sources,
                "cited": json.dumps(rec.cited, ensure_ascii=False),
                "general": rec.general_knowledge,
                "refused": rec.refused,
            },
        )
        return True
    except Exception:
        logger.warning("Could not record activity for %s", rec.user_identifier, exc_info=True)
        return False


async def chapters_worked(user_identifier: str) -> list[dict]:
    """Chapters the student worked, through questions or quizzes (most worked first).

    Returns:
        Rows ``course``, ``chapter``, ``questions`` (agent answers), ``quizzes``
        (distinct quizzes built from that chapter), ``last_seen``.
    """
    try:
        rows = await fetch_all(
            """
            SELECT course, chapter, sum(questions) AS questions, sum(quizzes) AS quizzes, max(last_seen) AS last_seen
            FROM (
                SELECT course, chapter, count(*) AS questions, 0 AS quizzes, max(created_at) AS last_seen
                FROM student_activity
                WHERE user_identifier = :user AND course IS NOT NULL AND NOT refused
                GROUP BY course, chapter
                UNION ALL
                SELECT course, chapter, 0 AS questions, count(DISTINCT quiz_id) AS quizzes,
                       max(answered_at) AS last_seen
                FROM quiz_results
                WHERE user_identifier = :user AND course IS NOT NULL
                GROUP BY course, chapter
            ) AS worked
            GROUP BY course, chapter
            ORDER BY sum(questions) + sum(quizzes) DESC, last_seen DESC
            """,
            {"user": user_identifier},
        )
    except Exception:
        logger.warning("Could not read activity for %s", user_identifier, exc_info=True)
        return []
    for r in rows:
        r["questions"] = int(r.get("questions") or 0)
        r["quizzes"] = int(r.get("quizzes") or 0)
    return rows


async def activity_counts(user_identifier: str) -> dict:
    """Totals for one student: questions, refused, answered from general knowledge."""
    try:
        rows = await fetch_all(
            """
            SELECT count(*) AS questions, count(*) FILTER (WHERE refused) AS refused,
                   count(*) FILTER (WHERE general_knowledge) AS general, count(DISTINCT thread_id) AS threads
            FROM student_activity WHERE user_identifier = :user
            """,
            {"user": user_identifier},
        )
    except Exception:
        logger.warning("Could not count activity for %s", user_identifier, exc_info=True)
        return {"questions": 0, "refused": 0, "general": 0, "threads": 0}
    r = rows[0] if rows else {}
    return {k: int(r.get(k) or 0) for k in ("questions", "refused", "general", "threads")}
