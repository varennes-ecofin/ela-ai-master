"""Per-turn context shared between the engine, the tools and the UI.

One :class:`TurnContext` is created for every student message. Tools reach
it through ``runtime.context`` (LangChain ``ToolRuntime``), record what they
read in the :class:`SourceRegistry` and leave pedagogical objects (quiz,
exercise) for the UI to render.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ela.agent.citations import SourceRegistry
from ela.pedagogy.schemas import Exercise, Quiz, ReviewSheet
from ela.sandbox.client import RunResult


@dataclass
class SearchRecord:
    """One ``search_course`` call (for the UI steps and the evaluation)."""

    query: str
    course: str | None
    chapter: str | None
    sids: list[str]

    @property
    def n_results(self) -> int:
        """Number of passages returned."""
        return len(self.sids)


@dataclass
class TurnContext:
    """State of one student message.

    Attributes:
        level: Academic level used to scope retrieval (never chosen by the model).
        registry: Passages handed to the model during this turn.
        retriever: Retriever override (tests); defaults to the shared level retriever.
        llm: Model override for quiz / exercise generation (tests).
        searches: ``search_course`` calls, in order.
        quiz: Quiz produced by the ``generate_quiz`` tool, if any.
        exercise: Exercise produced by the ``generate_exercise`` tool, if any.
        review_sheet: Review sheet produced by the ``generate_review_sheet`` tool, if any.
        slides_opened: Identifiers of the passages opened with ``get_slide``.
        summary: Summary of the turns that left the history window (system prompt).
        quiz_difficulty: Adaptive difficulty of the session, applied by ``generate_quiz``
            unless the student asks for a level.
        user_id: Student login (code-execution quota); ``None`` for anonymous sessions.
        thread_id: Chat thread id (persistence of code runs).
        code_runs: ``(code, result)`` of the scripts run by the ``run_python`` tool.
    """

    level: str = "ALL"
    registry: SourceRegistry = field(default_factory=SourceRegistry)
    retriever: Any = None
    llm: Any = None
    searches: list[SearchRecord] = field(default_factory=list)
    quiz: Quiz | None = None
    exercise: Exercise | None = None
    review_sheet: ReviewSheet | None = None
    slides_opened: list[str] = field(default_factory=list)
    summary: str | None = None
    quiz_difficulty: str | None = None
    user_id: str | None = None
    thread_id: str | None = None
    code_runs: list[tuple[str, RunResult]] = field(default_factory=list)

    def get_retriever(self):
        """Return the retriever for this turn (shared, level-scoped, unless overridden)."""
        if self.retriever is None:
            from ela.retrieval import get_retriever

            self.retriever = get_retriever(self.level)
        return self.retriever

    def get_llm(self):
        """Text model for quiz / exercise generation (shared unless overridden)."""
        if self.llm is None:
            from ela.engine import chat_llm

            self.llm = chat_llm()
        return self.llm

    @property
    def n_searches(self) -> int:
        """Number of course searches performed."""
        return len(self.searches)
