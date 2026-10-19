"""Exercise attempts: persistence and statistics (table ``exercise_attempts``)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ela.db.engine import execute, fetch_all

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExerciseAttempt:
    """One step submitted by a student in Socratic mode."""

    user_identifier: str
    exercise_id: str
    topic: str
    attempt_index: int
    attempt: str
    verdict: str
    completed: bool = False
    hints_revealed: int = 0
    title: str | None = None
    level: str | None = None
    thread_id: str | None = None


async def record_attempt(attempt: ExerciseAttempt) -> bool:
    """Store one attempt (returns ``False`` instead of raising on failure)."""
    try:
        await execute(
            """
            INSERT INTO exercise_attempts (
                user_identifier, thread_id, exercise_id, topic, title, level,
                attempt_index, attempt, verdict, completed, hints_revealed
            ) VALUES (:user, :thread, :exercise, :topic, :title, :level, :ai, :attempt, :verdict, :completed, :hints)
            """,
            {
                "user": attempt.user_identifier,
                "thread": attempt.thread_id,
                "exercise": attempt.exercise_id,
                "topic": attempt.topic,
                "title": attempt.title,
                "level": attempt.level,
                "ai": attempt.attempt_index,
                "attempt": attempt.attempt,
                "verdict": attempt.verdict,
                "completed": attempt.completed,
                "hints": attempt.hints_revealed,
            },
        )
        return True
    except Exception:
        logger.warning("Could not record exercise attempt for %s", attempt.user_identifier, exc_info=True)
        return False


async def exercise_summary(user_identifier: str) -> list[dict]:
    """Per-exercise summary: topic, title, attempts, best verdict, completed (most recent first)."""
    try:
        return await fetch_all(
            """
            SELECT exercise_id, topic, max(title) AS title, count(*) AS attempts,
                   bool_or(completed) AS completed, max(hints_revealed) AS hints_revealed,
                   max(created_at) AS last_attempt
            FROM exercise_attempts WHERE user_identifier = :user
            GROUP BY exercise_id, topic ORDER BY last_attempt DESC
            """,
            {"user": user_identifier},
        )
    except Exception:
        logger.warning("Could not read exercise summary for %s", user_identifier, exc_info=True)
        return []
