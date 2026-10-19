"""Tools of the ELA agent.

* ``search_course`` — hybrid search in the course, scoped to the student's level;
* ``get_course_outline`` — chapter › section › slide titles of a course;
* ``get_slide`` — full text of one slide (by identifier or title);
* ``generate_quiz`` / ``generate_exercise`` / ``generate_review_sheet`` —
  structured pedagogical objects left in the turn context for the UI.

Every passage shown to the model is registered in the turn's
:class:`~ela.agent.citations.SourceRegistry` and labelled ``[S1]``,
``[S2]``... so that the answer can cite it and the citation can be verified.
Tools never raise: errors are returned as text the model can act on.
"""

from __future__ import annotations

import logging

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool

from ela.agent.citations import SourceRef
from ela.agent.context import SearchRecord, TurnContext
from ela.agent.outline import (
    courses_elsewhere,
    find_slides,
    get_outlines,
    render_course_list,
    render_outline,
    resolve_course,
)
from ela.config import get_settings
from ela.pedagogy.exercise import generate_exercise as _generate_exercise
from ela.pedagogy.quiz import generate_quiz as _generate_quiz
from ela.pedagogy.review import ReviewLookupError, review_sheet_for
from ela.pedagogy.schemas import Difficulty
from ela.sandbox import service as sandbox

logger = logging.getLogger(__name__)


def format_ref(ref: SourceRef) -> str:
    """Render one passage for the model (identifier, location, text)."""
    score = f"  (score {ref.score:.4f})" if ref.score is not None else ""
    return f"[{ref.sid}] {ref.source} › {ref.breadcrumb}{score}\n{ref.content}"


def _outlines(ctx: TurnContext):
    try:
        return get_outlines(ctx.level)
    except Exception as exc:  # store unreachable
        logger.warning("Outline unavailable: %s", exc)
        return {}


def _course_filter(ctx: TurnContext, course: str | None) -> tuple[list[str] | None, str | None]:
    """Resolve the course given by the model; ``(courses, error)``."""
    if not course:
        return None, None
    outlines = _outlines(ctx)
    resolved = resolve_course(course, outlines)
    if resolved is None:
        known = ", ".join(outlines) or "aucun"
        return None, f"Cours « {course} » inconnu pour ce niveau (cours disponibles : {known}). Recherche sans filtre."
    return [resolved], None


@tool("search_course", parse_docstring=True)
def search_course(
    query: str,
    runtime: ToolRuntime[TurnContext],
    course: str | None = None,
    chapter: str | None = None,
    k: int | None = None,
) -> str:
    """Search the student's course material (slides, definitions, theorems) for a topic.

    Call it once per distinct notion: a question comparing two notions needs
    two searches. Results are labelled [S1], [S2]... — cite them with these
    labels only.

    Args:
        runtime: Injected turn context (hidden from the model).
        query: Short search query in the language of the course (key terms, not a full sentence).
        course: Optional course name to restrict the search (folder name, acronym such as UTS/MTS/HFE, or file name).
        chapter: Optional chapter title fragment to narrow the search.
        k: Number of passages to return (default 5, max 8).
    """
    ctx = runtime.context
    settings = get_settings()
    k = settings.agent_search_k if not k else max(1, min(int(k), settings.agent_search_k_max))
    courses, note = _course_filter(ctx, course)
    try:
        docs = ctx.get_retriever().invoke(query, courses=courses, chapter=chapter, top_n=k)
    except Exception as exc:
        logger.exception("search_course failed")
        return f"La recherche a échoué ({exc.__class__.__name__}). Réessaie avec une requête plus courte."
    refs = ctx.registry.register_all(docs, origin="search", query=query)
    ctx.searches.append(
        SearchRecord(query=query, course=courses[0] if courses else None, chapter=chapter, sids=[r.sid for r in refs])
    )
    if not refs:
        where = f" dans le cours {courses[0]}" if courses else ""
        empty = (
            f"Aucun extrait trouvé pour « {query} »{where}. Reformule avec d'autres termes du cours "
            "ou consulte le plan avec get_course_outline."
        )
        return f"{note} {empty}" if note else empty
    header = f"{len(refs)} extrait(s) pour « {query} »" + (f" (cours {courses[0]})" if courses else "") + " :"
    body = "\n\n".join(format_ref(r) for r in refs)
    return "\n".join(filter(None, [note, header, "", body]))


@tool("get_course_outline", parse_docstring=True)
def get_course_outline(runtime: ToolRuntime[TurnContext], course: str | None = None) -> str:
    """Show the outline of a course (chapters, sections, slide titles), or list the courses available.

    Use it when the student asks what a course or chapter covers, where a
    notion is taught, or before searching a specific chapter.

    Args:
        runtime: Injected turn context (hidden from the model).
        course: Course name (folder, acronym or file name). Omit it to list the available courses.
    """
    ctx = runtime.context
    outlines = _outlines(ctx)
    if not outlines:
        return "Aucun cours n'est disponible (base de connaissances vide ou inaccessible)."
    if not course:
        return render_course_list(outlines)
    resolved = resolve_course(course, outlines)
    if resolved is None:
        return f"Cours « {course} » inconnu. " + render_course_list(outlines)
    return render_outline(outlines[resolved], max_chars=get_settings().agent_outline_max_chars)


@tool("get_slide", parse_docstring=True)
def get_slide(
    runtime: ToolRuntime[TurnContext],
    source_id: str | None = None,
    slide_title: str | None = None,
    course: str | None = None,
) -> str:
    """Read the full text of one slide, either by its label (S3) or by its title.

    Use it to re-read a passage already returned by search_course, or to open
    a slide named in a course outline.

    Args:
        runtime: Injected turn context (hidden from the model).
        source_id: Label of a passage already returned (for example "S3").
        slide_title: Exact or partial slide title, as shown in the outline.
        course: Course name to disambiguate a title (folder, acronym or file name).
    """
    ctx = runtime.context
    if source_id:
        ref = ctx.registry.get(source_id)
        if ref is None:
            return f"Identifiant « {source_id} » inconnu : seuls les identifiants déjà renvoyés ({', '.join(sorted(ctx.registry.ids())) or 'aucun'}) sont valides."
        ctx.slides_opened.append(ref.sid)
        return format_ref(ref)
    if not slide_title:
        return "Indique source_id (ex. S2) ou slide_title."
    outlines = _outlines(ctx)
    resolved = resolve_course(course, outlines) if course else None
    docs = find_slides(outlines, slide_title, course=resolved)
    if not docs:
        return (
            f"Aucune slide dont le titre contient « {slide_title} »" + (f" dans {resolved}" if resolved else "") + "."
        )
    refs = ctx.registry.register_all(docs[:3], origin="slide", query=slide_title)
    ctx.slides_opened.extend(r.sid for r in refs)
    note = f"{len(docs)} slides correspondent ; les {len(refs)} premières :\n\n" if len(docs) > len(refs) else ""
    return note + "\n\n".join(format_ref(r) for r in refs)


@tool("generate_quiz", parse_docstring=True)
async def generate_quiz(
    topic: str,
    runtime: ToolRuntime[TurnContext],
    n: int = 3,
    difficulty: Difficulty | None = None,
) -> str:
    """Generate a multiple-choice quiz on a course topic; it is displayed to the student automatically.

    Call it when the student asks to be tested or wants a quiz. After the
    call, answer with one short introductory sentence only — never rewrite
    the questions.

    Args:
        runtime: Injected turn context (hidden from the model).
        topic: Quiz topic, in the words of the course.
        n: Number of questions (1 to 10); use the number the student asked for.
        difficulty: "facile", "moyen" or "difficile" — only when the student asks for a level; otherwise leave it unset and the level adapted to the student's previous scores is used.
    """
    ctx = runtime.context
    difficulty = difficulty or ctx.quiz_difficulty or "moyen"
    try:
        quiz = await _generate_quiz(
            ctx.get_llm(),
            ctx.get_retriever(),
            topic,
            n=n,
            difficulty=difficulty,
            level=ctx.level,
            registry=ctx.registry,
        )
    except Exception as exc:
        logger.exception("generate_quiz failed")
        return f"La génération du quiz a échoué ({exc.__class__.__name__}). Propose à l'étudiant de réessayer."
    if quiz is None:
        return _nothing_on_topic(ctx, topic, "quiz")
    ctx.quiz = quiz
    return (
        f"Quiz de {len(quiz.questions)} questions généré sur « {topic} », difficulté « {quiz.difficulty} » "
        f"(extraits {', '.join(quiz.sources)}). "
        "Il est affiché à l'étudiant avec des boutons de réponse : réponds seulement par une courte phrase d'introduction."
    )


@tool("generate_exercise", parse_docstring=True)
async def generate_exercise(topic: str, runtime: ToolRuntime[TurnContext]) -> str:
    """Generate an exercise (statement, progressive hints, solution) on a course topic; it is displayed automatically.

    Call it when the student asks for an exercise or practice problem. After
    the call, answer with one short introductory sentence only — never
    rewrite the statement or reveal the solution.

    Args:
        runtime: Injected turn context (hidden from the model).
        topic: Exercise topic, in the words of the course.
    """
    ctx = runtime.context
    try:
        exercise = await _generate_exercise(
            ctx.get_llm(), ctx.get_retriever(), topic, level=ctx.level, registry=ctx.registry
        )
    except Exception as exc:
        logger.exception("generate_exercise failed")
        return f"La génération de l'exercice a échoué ({exc.__class__.__name__}). Propose à l'étudiant de réessayer."
    if exercise is None:
        return _nothing_on_topic(ctx, topic, "exercice")
    ctx.exercise = exercise
    return (
        f"Exercice « {exercise.title} » généré (extraits {', '.join(exercise.sources)}). "
        "Il est affiché à l'étudiant avec les boutons Indice / Solution : réponds seulement par une courte phrase d'introduction."
    )


@tool("generate_review_sheet", parse_docstring=True)
async def generate_review_sheet(course: str, chapter: str, runtime: ToolRuntime[TurnContext]) -> str:
    """Build the one-page review sheet of a course chapter; it is displayed to the student automatically.

    Call it when the student asks for a summary, a review sheet or « the
    essentials » of a chapter. Use get_course_outline first if the chapter is
    not clearly identified. After the call, answer with one short
    introductory sentence only — never rewrite the sheet.

    Args:
        runtime: Injected turn context (hidden from the model).
        course: Course name (folder, acronym such as UTS/MTS/HFE, or file name).
        chapter: Chapter title (as in the outline), number (« 2 ») or a distinctive fragment of the title.
    """
    ctx = runtime.context
    try:
        sheet = await review_sheet_for(ctx.get_llm(), ctx.level, course, chapter)
    except ReviewLookupError as exc:
        return str(exc)
    except Exception as exc:
        logger.exception("generate_review_sheet failed")
        return f"La génération de la fiche a échoué ({exc.__class__.__name__}). Propose à l'étudiant de réessayer."
    ctx.review_sheet = sheet
    return (
        f"Fiche « {sheet.title} » générée pour {sheet.course} › {sheet.chapter} ({len(sheet.slides)} slides). "
        "Elle est affichée à l'étudiant avec un fichier à télécharger : réponds seulement par une courte phrase d'introduction."
    )


@tool("run_python", parse_docstring=True)
async def run_python(code: str, runtime: ToolRuntime[TurnContext]) -> str:
    """Run a short Python script in a sandbox (numpy, pandas, scipy, statsmodels, matplotlib, arch) and return its output.

    Use it when the student asks to compute, simulate, estimate or check
    something numerically. The script must be self-contained (synthetic data
    with a fixed seed, no network, no files, no input()); print the numbers
    you need and let matplotlib figures open — they are captured and shown to
    the student. Comment on the output; do not copy it back.

    Args:
        runtime: Injected turn context (hidden from the model).
        code: Complete Python script.
    """
    ctx = runtime.context
    try:
        result = await sandbox.execute(
            code, user_id=ctx.user_id, thread_id=ctx.thread_id, origin="agent", topic=None, language="python"
        )
    except sandbox.QuotaExceededError as exc:
        return f"Exécution refusée : l'étudiant a atteint son quota de {exc.limit} exécutions par jour. Dis-le-lui et réponds sans exécuter."
    except sandbox.SandboxUnavailableError:
        return "Le bac à sable d'exécution est indisponible pour le moment. Réponds sans exécuter de code (dis-le à l'étudiant)."
    except Exception as exc:
        logger.exception("run_python failed")
        return f"L'exécution a échoué ({exc.__class__.__name__}). Réponds sans exécuter de code."
    ctx.code_runs.append((code, result))
    return sandbox.summary_for_model(result)


def _nothing_on_topic(ctx: TurnContext, topic: str, what: str) -> str:
    """Tool answer when the visible course has nothing on ``topic`` (level-aware)."""
    elsewhere = courses_elsewhere(topic, ctx.level)
    if elsewhere:
        where = ", ".join(f"{course} (niveau {level})" for course, level in elsewhere)
        return (
            f"Les cours accessibles à cet étudiant (niveau {ctx.level}) ne contiennent rien sur « {topic} » : aucun {what} généré. "
            f"Ce sujet est traité dans un cours qui n'est pas au programme de son niveau : {where}. "
            "Dis-le-lui clairement (sans affirmer que le cours n'existe pas) et propose un sujet proche parmi ses cours."
        )
    return f"Le cours ne contient rien sur « {topic} » : aucun {what} généré. Dis-le à l'étudiant et propose un sujet proche."


def build_tools(*, sandbox_enabled: bool | None = None) -> list[BaseTool]:
    """Tools of the agent, in the order shown to the model.

    Args:
        sandbox_enabled: Include ``run_python`` (defaults to ``SANDBOX_URL`` being set).
    """
    tools: list[BaseTool] = [
        search_course,
        get_course_outline,
        get_slide,
        generate_quiz,
        generate_exercise,
        generate_review_sheet,
    ]
    if get_settings().sandbox_enabled if sandbox_enabled is None else sandbox_enabled:
        tools.append(run_python)
    return tools
