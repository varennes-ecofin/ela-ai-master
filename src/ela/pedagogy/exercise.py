"""Exercise generation grounded in the course (structured output)."""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ela.agent.citations import SourceRegistry
from ela.agent.prompts import render
from ela.pedagogy.quiz import GENERATION_MAX_TOKENS, format_passages
from ela.pedagogy.relevance import covers
from ela.pedagogy.schemas import Exercise, ExercisePayload, StepFeedback
from ela.structured import ainvoke_structured

logger = logging.getLogger(__name__)


async def generate_exercise(
    llm: BaseChatModel,
    retriever,
    topic: str,
    *,
    level: str = "ALL",
    registry: SourceRegistry | None = None,
) -> Exercise | None:
    """Generate an exercise (statement, hints, solution) on ``topic``.

    Args:
        llm: Text model.
        retriever: Level-scoped retriever.
        topic: Exercise topic.
        level: Student level, for the prompt.
        registry: Registry receiving the passages; a private one otherwise.

    Returns:
        The exercise, or ``None`` when the course has nothing on the topic.
    """
    docs = retriever.invoke(topic)
    if not docs:
        return None
    registry = registry if registry is not None else SourceRegistry()
    refs = registry.register_all(docs, origin="search", query=topic)
    if not covers(topic, [f"{r.breadcrumb} {r.content}" for r in refs]):
        logger.info("Exercise on « %s »: retrieved passages do not cover the topic", topic)
        return None
    system = render("exercise", level=level, topic=topic, context=format_passages(refs))
    messages = [SystemMessage(content=system), HumanMessage(content=f"Rédige l'exercice sur « {topic} ».")]
    payload = await ainvoke_structured(llm, ExercisePayload, messages, max_tokens=GENERATION_MAX_TOKENS)
    return Exercise(**payload.model_dump(), topic=topic, sources=[r.sid for r in refs])


async def check_step(
    llm: BaseChatModel,
    exercise: Exercise,
    attempt: str,
    *,
    level: str = "ALL",
    revealed_hints: int = 0,
    previous: list[tuple[str, str]] | None = None,
) -> StepFeedback:
    """Evaluate one step written by the student without revealing the solution.

    Args:
        llm: Text model.
        exercise: The exercise being solved.
        attempt: What the student wrote.
        level: Student level, for the prompt.
        revealed_hints: Number of hints already shown.
        previous: Earlier ``(attempt, verdict)`` pairs of this exercise.

    Returns:
        The structured feedback.
    """
    hints = exercise.hints[:revealed_hints]
    system = render(
        "check_step",
        level=level,
        statement=exercise.statement,
        solution=exercise.solution,
        hints="\n".join(f"- {h}" for h in hints) if hints else "(aucun)",
        previous="\n".join(f"- [{v}] {a}" for a, v in (previous or [])) or "(aucune)",
    )
    messages = [SystemMessage(content=system), HumanMessage(content=f"TENTATIVE DE L'ÉTUDIANT :\n{attempt}")]
    return await ainvoke_structured(llm, StepFeedback, messages)
