"""Quiz generation grounded in the course (structured output).

Long quizzes are generated in batches of :data:`QUIZ_BATCH` questions: one
model call per batch keeps each JSON answer short enough to be reliable and
lets the next batch avoid the questions already produced.
"""

from __future__ import annotations

import logging
import math
import random
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ela.agent.citations import SourceRegistry
from ela.agent.prompts import render
from ela.pedagogy.relevance import covers
from ela.pedagogy.schemas import Difficulty, Quiz, QuizPayload, QuizQuestion
from ela.structured import ainvoke_structured

logger = logging.getLogger(__name__)

QUIZ_BATCH = 5  # questions per model call
GENERATION_MAX_TOKENS = 4096  # completion budget of one generation call (reasoning tokens included)
MAX_QUESTIONS = 10

# An option that is not a proposition: empty, punctuation only, or a leftover
# of a neighbouring option (« **B)** ») after a broken JSON answer.
_BROKEN_OPTION_RE = re.compile(r"^\W*$|\*\*\s*[A-D]\s*\)\s*\*\*")


def format_passages(refs) -> str:
    """Render registered passages as the context block of a generation prompt.

    Passages are numbered without their ``S#`` labels: the student never sees
    those identifiers in a quiz, so the model must not refer to them.
    """
    return "\n\n".join(f"— Extrait {i} ({r.breadcrumb}) —\n{r.content}" for i, r in enumerate(refs, start=1))


def well_formed(question: QuizQuestion) -> bool:
    """Whether a generated question can be shown to a student.

    Rejects empty statements, options that are empty or garbled, and
    questions whose options are not distinct.
    """
    if not question.question.strip():
        return False
    options = [o.strip() for o in question.options]
    if any(_BROKEN_OPTION_RE.search(o) for o in options):
        return False
    return len({o.lower() for o in options}) == len(options) >= 2


async def generate_quiz(
    llm: BaseChatModel,
    retriever,
    topic: str,
    *,
    n: int = 3,
    difficulty: Difficulty = "moyen",
    level: str = "ALL",
    registry: SourceRegistry | None = None,
    rng: random.Random | None = None,
    focus: list[str] | None = None,
    course: str | None = None,
    chapter: str | None = None,
) -> Quiz | None:
    """Generate a quiz on ``topic`` from the course passages.

    Args:
        llm: Text model.
        retriever: Level-scoped retriever (``invoke(query)``).
        topic: Quiz topic.
        n: Number of questions (1–10).
        difficulty: ``facile`` / ``moyen`` / ``difficile``.
        level: Student level, for the prompt.
        registry: Registry receiving the passages (sources panel); a private one otherwise.
        rng: Random generator used to shuffle the options (tests).
        focus: Questions previously failed; the new quiz targets the same notions
            with different wording (« refaire un quiz sur mes erreurs »).
        course: Restrict the passages to one course folder (suggestions, review sheet).
        chapter: Restrict the passages to one chapter title (with ``course``).

    Returns:
        The quiz, or ``None`` when the course has nothing on the topic (no
        passage, or passages that do not mention it).

    Raises:
        ValueError: When the model produced no usable question.
    """
    n = max(1, min(int(n), MAX_QUESTIONS))
    docs = retrieve(retriever, topic, course=course, chapter=chapter)
    if not docs:
        return None
    registry = registry if registry is not None else SourceRegistry()
    refs = registry.register_all(docs, origin="search", query=topic)
    if not covers(topic, [f"{r.breadcrumb} {r.content}" for r in refs]):
        logger.info("Quiz on « %s »: retrieved passages do not cover the topic", topic)
        return None
    context = format_passages(refs)

    questions: list[QuizQuestion] = []
    seen: set[str] = set()
    max_rounds = math.ceil(n / QUIZ_BATCH) + 1  # one extra round for batches that came back short
    for round_no in range(1, max_rounds + 1):
        want = min(QUIZ_BATCH, n - len(questions))
        if want <= 0:
            break
        system = render(
            "quiz",
            level=level,
            n=str(want),
            topic=topic,
            difficulty=difficulty,
            context=context,
            focus_block=focus_block(focus),
            avoid_block=avoid_block([q.question for q in questions]),
        )
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"Génère le quiz de {want} questions sur « {topic} »."),
        ]
        try:
            payload = await ainvoke_structured(llm, QuizPayload, messages, max_tokens=GENERATION_MAX_TOKENS)
        except Exception:
            if not questions:
                raise
            logger.warning("Quiz batch %d failed; keeping the %d questions already generated", round_no, len(questions))
            break
        fresh = [q for q in payload.questions if well_formed(q) and q.question.strip().lower() not in seen]
        if not fresh:
            logger.warning("Quiz batch %d produced no usable question", round_no)
            break
        for q in fresh[:want]:
            seen.add(q.question.strip().lower())
            questions.append(q)
    if not questions:
        raise ValueError("the model produced no usable quiz question")
    if len(questions) < n:
        logger.info("Quiz on « %s »: %d/%d questions generated", topic, len(questions), n)
    questions = [shuffle_options(q, rng) for q in questions]
    return Quiz(
        topic=topic,
        questions=questions,
        sources=[r.sid for r in refs],
        difficulty=difficulty,
        source_labels=[r.breadcrumb for r in refs],
        course=_dominant(r.course for r in refs),
        chapter=_dominant(r.chapter for r in refs),
    )


def retrieve(retriever, topic: str, *, course: str | None = None, chapter: str | None = None):
    """Passages for ``topic``, restricted to a course / chapter when asked (plain ``invoke`` otherwise)."""
    filters: dict = {}
    if course:
        filters["courses"] = [course]
    if chapter:
        filters["chapter"] = chapter
    return retriever.invoke(topic, **filters) if filters else retriever.invoke(topic)


def focus_block(focus: list[str] | None) -> str:
    """Prompt block listing the questions to re-target (empty when none)."""
    if not focus:
        return ""
    lines = "\n".join(f"- {q}" for q in focus)
    return (
        "\nQUESTIONS RATÉES PRÉCÉDEMMENT (vise les mêmes notions, mais reformule : "
        "autre angle, autre exemple, jamais la même question) :\n" + lines
    )


def avoid_block(already: list[str]) -> str:
    """Prompt block listing the questions of the previous batches (empty for the first)."""
    if not already:
        return ""
    lines = "\n".join(f"- {q}" for q in already)
    return "\nQUESTIONS DÉJÀ POSÉES DANS CE QUIZ (n'en repose aucune, ni une variante) :\n" + lines


def _dominant(values) -> str | None:
    """Most frequent non-empty value of an iterable."""
    counts: dict[str, int] = {}
    for v in values:
        if v:
            counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get) if counts else None


def shuffle_options(question: QuizQuestion, rng: random.Random | None = None) -> QuizQuestion:
    """Return a copy of ``question`` with its options in random order.

    Models tend to put the correct option first whatever the prompt says;
    shuffling after generation makes the position of the correct answer
    uniform. ``rng`` allows deterministic tests.

    Args:
        question: Generated question.
        rng: Random generator (defaults to the module one).

    Returns:
        The question with permuted options and an updated ``correct_index``.
    """
    rng = rng or random
    order = list(range(len(question.options)))
    rng.shuffle(order)
    return question.model_copy(
        update={
            "options": [question.options[i] for i in order],
            "correct_index": order.index(question.correct_index),
        }
    )
