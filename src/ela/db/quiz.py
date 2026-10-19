"""Quiz results: persistence and simple statistics (table ``quiz_results``).

Every answered question is stored with the student's choice, the correct
option and the course passages the quiz was built from. Writes never
raise: a database hiccup must not break the quiz in the chat.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from ela.db.engine import execute, fetch_all, fetch_one

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuizAnswer:
    """One answered quiz question."""

    user_identifier: str
    quiz_id: str
    topic: str
    question_index: int
    question: str
    options: list[str]
    correct_index: int
    answer_index: int
    thread_id: str | None = None
    difficulty: str | None = None
    level: str | None = None
    course: str | None = None
    chapter: str | None = None
    sources: list[str] = field(default_factory=list)

    @property
    def correct(self) -> bool:
        """Whether the student picked the right option."""
        return self.answer_index == self.correct_index


async def record_answer(answer: QuizAnswer) -> bool:
    """Store one answer (returns ``False`` instead of raising on failure)."""
    try:
        await execute(
            """
            INSERT INTO quiz_results (
                user_identifier, thread_id, quiz_id, topic, difficulty, level, course, chapter,
                question_index, question, options, correct_index, answer_index, correct, sources
            ) VALUES (
                :user, :thread, :quiz, :topic, :difficulty, :level, :course, :chapter,
                :qi, :question, CAST(:options AS jsonb), :ci, :ai, :correct, CAST(:sources AS jsonb)
            )
            """,
            {
                "user": answer.user_identifier,
                "thread": answer.thread_id,
                "quiz": answer.quiz_id,
                "topic": answer.topic,
                "difficulty": answer.difficulty,
                "level": answer.level,
                "course": answer.course,
                "chapter": answer.chapter,
                "qi": answer.question_index,
                "question": answer.question,
                "options": json.dumps(answer.options, ensure_ascii=False),
                "ci": answer.correct_index,
                "ai": answer.answer_index,
                "correct": answer.correct,
                "sources": json.dumps(answer.sources, ensure_ascii=False),
            },
        )
        return True
    except Exception:
        logger.warning("Could not record quiz answer for %s", answer.user_identifier, exc_info=True)
        return False


async def recent_errors(user_identifier: str, *, limit: int = 10, quiz_id: str | None = None) -> list[dict]:
    """Questions the student got wrong, most recent first.

    Args:
        user_identifier: Student login.
        limit: Maximum number of rows.
        quiz_id: Restrict to one quiz.

    Returns:
        Rows with ``topic``, ``question``, ``options``, ``correct_index``, ``answer_index``, ``sources``,
        ``answered_at``.
    """
    where = "user_identifier = :user AND NOT correct" + (" AND quiz_id = :quiz" if quiz_id else "")
    try:
        return await fetch_all(
            f"""
            SELECT topic, difficulty, course, chapter, question, options, correct_index, answer_index,
                   sources, answered_at
            FROM quiz_results WHERE {where}
            ORDER BY answered_at DESC LIMIT :limit
            """,
            {"user": user_identifier, "quiz": quiz_id, "limit": limit},
        )
    except Exception:
        logger.warning("Could not read quiz errors for %s", user_identifier, exc_info=True)
        return []


@dataclass(frozen=True)
class QuizStats:
    """Aggregate quiz statistics of one student."""

    n_answers: int
    n_correct: int
    n_quizzes: int
    by_topic: list[dict]  # {"topic", "n", "correct", "rate"} sorted by rate ascending (weakest first)

    @property
    def rate(self) -> float | None:
        """Overall success rate, ``None`` without answers."""
        return self.n_correct / self.n_answers if self.n_answers else None


async def user_stats(user_identifier: str) -> QuizStats:
    """Success rate overall and per topic (weakest topics first)."""
    try:
        total = await fetch_one(
            """
            SELECT count(*) AS n, count(*) FILTER (WHERE correct) AS ok, count(DISTINCT quiz_id) AS quizzes
            FROM quiz_results WHERE user_identifier = :user
            """,
            {"user": user_identifier},
        )
        rows = await fetch_all(
            """
            SELECT topic, count(*) AS n, count(*) FILTER (WHERE correct) AS ok
            FROM quiz_results WHERE user_identifier = :user
            GROUP BY topic ORDER BY (count(*) FILTER (WHERE correct))::float / count(*) ASC, count(*) DESC
            """,
            {"user": user_identifier},
        )
    except Exception:
        logger.warning("Could not read quiz stats for %s", user_identifier, exc_info=True)
        return QuizStats(0, 0, 0, [])
    by_topic = [
        {"topic": r["topic"], "n": int(r["n"]), "correct": int(r["ok"]), "rate": int(r["ok"]) / int(r["n"])}
        for r in rows
    ]
    return QuizStats(
        n_answers=int(total["n"]) if total else 0,
        n_correct=int(total["ok"]) if total else 0,
        n_quizzes=int(total["quizzes"]) if total else 0,
        by_topic=by_topic,
    )


def next_difficulty(current: str, score: int, n_questions: int) -> str:
    """Adapt the difficulty of the next quiz to the last score.

    Args:
        current: ``facile`` / ``moyen`` / ``difficile``.
        score: Correct answers.
        n_questions: Questions asked.

    Returns:
        The next difficulty: harder from 80 %, easier at 40 % or below, unchanged otherwise.
    """
    levels = ["facile", "moyen", "difficile"]
    if n_questions <= 0 or current not in levels:
        return current if current in levels else "moyen"
    rate = score / n_questions
    index = levels.index(current)
    if rate >= 0.8:
        index = min(index + 1, 2)
    elif rate <= 0.4:
        index = max(index - 1, 0)
    return levels[index]
