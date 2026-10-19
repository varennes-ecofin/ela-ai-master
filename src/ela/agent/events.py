"""Events streamed by :meth:`ela.engine.ElaEngine.ask`.

The Chainlit app and the evaluation consume the same stream: text tokens of
the answer, tool activity (for ``cl.Step``), the pedagogical objects left by
the tools and a final event carrying the verified answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ela.agent.citations import SourceRef
from ela.pedagogy.schemas import Exercise, Quiz, ReviewSheet
from ela.sandbox.client import RunResult


@dataclass(frozen=True)
class TokenEvent:
    """A piece of the answer being streamed."""

    text: str


@dataclass(frozen=True)
class ResetEvent:
    """Discard the text streamed so far (the model decided to call tools instead)."""


@dataclass(frozen=True)
class ToolStartEvent:
    """A tool call starts."""

    call_id: str
    name: str
    args: dict
    label: str


@dataclass(frozen=True)
class ToolEndEvent:
    """A tool call ended."""

    call_id: str
    name: str
    summary: str
    ok: bool = True


@dataclass(frozen=True)
class QuizEvent:
    """A quiz was generated during the turn."""

    quiz: Quiz


@dataclass(frozen=True)
class ExerciseEvent:
    """An exercise was generated during the turn."""

    exercise: Exercise


@dataclass(frozen=True)
class ReviewSheetEvent:
    """A review sheet was generated (or served from the cache) during the turn."""

    sheet: ReviewSheet


@dataclass(frozen=True)
class CodeRunEvent:
    """A script was executed by the ``run_python`` tool during the turn."""

    code: str
    result: RunResult


@dataclass(frozen=True)
class SourcesEvent:
    """Passages of the turn: those cited by the answer and all those read."""

    cited: list[SourceRef]
    all: list[SourceRef]


@dataclass(frozen=True)
class FinalEvent:
    """The verified final answer and the turn statistics."""

    text: str
    citations_valid: list[str] = field(default_factory=list)
    citations_invalid: list[str] = field(default_factory=list)
    n_model_calls: int = 0
    n_tool_calls: int = 0
    n_searches: int = 0
    limit_reached: bool = False


@dataclass(frozen=True)
class ErrorEvent:
    """The turn failed; ``message`` is safe to show to the student."""

    message: str


AgentEvent = (
    TokenEvent
    | ResetEvent
    | ToolStartEvent
    | ToolEndEvent
    | QuizEvent
    | ExerciseEvent
    | ReviewSheetEvent
    | CodeRunEvent
    | SourcesEvent
    | FinalEvent
    | ErrorEvent
)


def tool_label(name: str, args: dict) -> str:
    """Human-readable (French) label of a tool call for the UI."""
    a = args or {}
    if name == "search_course":
        label = f"🔎 Recherche : « {a.get('query', '')} »"
        if a.get("course"):
            label += f" — cours {a['course']}"
        if a.get("chapter"):
            label += f", chapitre « {a['chapter']} »"
        return label
    if name == "get_course_outline":
        return f"📑 Plan du cours {a['course']}" if a.get("course") else "📑 Liste des cours"
    if name == "get_slide":
        if a.get("source_id"):
            return f"📄 Lecture de l'extrait {a['source_id']}"
        return f"📄 Lecture de la slide « {a.get('slide_title', '')} »"
    if name == "generate_quiz":
        return f"🎓 Génération d'un quiz : {a.get('topic', '')}"
    if name == "generate_review_sheet":
        return f"📄 Fiche de révision : {a.get('course', '')} › {a.get('chapter', '')}"
    if name == "run_python":
        return "💻 Exécution d'un script Python"
    if name == "generate_exercise":
        return f"✏️ Génération d'un exercice : {a.get('topic', '')}"
    if name == "read_image":
        return "👁️ Lecture de l'image jointe"
    return f"🛠️ {name}"


_FAILURE_MARKERS = ("Tool call limit exceeded", "a échoué")


def tool_failed(result: str) -> bool:
    """Whether a tool result (tools never raise) reports a failure."""
    head = result.lstrip()[:160]
    return any(marker in head for marker in _FAILURE_MARKERS)


def tool_summary(name: str, result: str, *, n_sources: int | None = None) -> str:
    """Short description of a tool result for the UI step."""
    if name == "search_course":
        if "Aucun extrait trouvé" in result:
            return "aucun extrait trouvé"
        m = re.search(r"(\d+) extrait\(s\)", result)
        if m:
            return f"{m.group(1)} extrait(s)"
        return f"{n_sources} extrait(s)" if n_sources is not None else result.split("\n", 1)[0]
    if name == "get_course_outline":
        return result.split("\n", 1)[0]
    if name == "get_slide":
        return result.split("\n", 1)[0]
    if name in {"generate_quiz", "generate_exercise", "generate_review_sheet"}:
        return result.split(". ", 1)[0]
    if name == "run_python":
        return result.split("\n", 1)[0]
    return result[:120]
