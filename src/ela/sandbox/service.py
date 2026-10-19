"""Execution service: quota, persistence, result formatting, automatic fix, interpretation.

Sits between the UI / agent tool and the HTTP client: checks the student's
daily quota, runs the code, records the outcome, can ask the model to
repair a failing script once, and interprets the real output of a
successful run against the course (table of results + commentary with
citations).
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ela.agent.citations import SourceRef, SourceRegistry, verify_citations
from ela.agent.prompts import render
from ela.config import get_settings
from ela.db.code_runs import CodeRun, record_run, runs_today
from ela.mathfmt import normalize_math
from ela.pedagogy.quiz import retrieve
from ela.sandbox.client import RunResult, SandboxUnavailableError, run_code

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```[ \t]*(?P<lang>[\w+-]*)[^\n]*\n(?P<body>.*?)```", re.DOTALL)
MAX_SHOWN_CHARS = 6000
STDERR_TAIL_LINES = 25


class QuotaExceededError(RuntimeError):
    """The student reached the daily number of executions."""

    def __init__(self, limit: int):
        super().__init__(f"daily quota of {limit} executions reached")
        self.limit = limit


def extract_code(markdown: str, language: str = "python") -> str | None:
    """Return the body of the first fenced block of ``language`` (any fenced block otherwise).

    Args:
        markdown: Model answer (intro + code block + interpretation).
        language: Fence language to prefer.

    Returns:
        The code, or ``None`` when the text holds no fenced block.
    """
    blocks = list(_FENCE_RE.finditer(markdown))
    if not blocks:
        return None
    wanted = [b for b in blocks if b.group("lang").lower() in {language.lower(), "py" if language == "python" else ""}]
    block = (wanted or blocks)[0]
    body = block.group("body").strip("\n")
    return body if body.strip() else None


async def quota_remaining(user_id: str | None) -> int | None:
    """Return the executions left today for a student (``None`` = unlimited or anonymous)."""
    limit = get_settings().sandbox_daily_quota
    if not user_id or limit <= 0:
        return None
    return max(0, limit - await runs_today(user_id))


async def execute(
    code: str,
    *,
    user_id: str | None,
    thread_id: str | None = None,
    origin: str = "workshop",
    topic: str | None = None,
    language: str = "python",
) -> RunResult:
    """Run ``code`` for a student, within the daily quota, and record the outcome.

    Args:
        code: Script source.
        user_id: Student login (``None``: no quota, no persistence).
        thread_id: Chat thread id.
        origin: ``workshop`` / ``agent`` / ``fix``.
        topic: Subject of the script (reporting).
        language: Script language (only ``python`` is executed).

    Returns:
        The run result.

    Raises:
        QuotaExceededError: Daily quota reached.
        SandboxUnavailableError: Sandbox not configured or unreachable.
    """
    limit = get_settings().sandbox_daily_quota
    if user_id and limit > 0 and await runs_today(user_id) >= limit:
        raise QuotaExceededError(limit)
    try:
        result = await run_code(code, language=language)
    except SandboxUnavailableError as exc:
        if user_id:
            await record_run(
                CodeRun(user_id, "unavailable", origin, topic, language, error=str(exc), thread_id=thread_id)
            )
        raise
    if user_id:
        await record_run(
            CodeRun(
                user_id,
                result.status,
                origin,
                topic,
                language,
                elapsed_ms=result.elapsed_ms,
                n_images=len(result.images),
                error=_error_summary(result),
                thread_id=thread_id,
            )
        )
    return result


def _error_summary(result: RunResult) -> str | None:
    if result.ok:
        return None
    lines = [ln for ln in result.stderr.strip().splitlines() if ln.strip()]
    return lines[-1][:500] if lines else result.status


def stderr_tail(stderr: str, lines: int = STDERR_TAIL_LINES) -> str:
    """Last lines of a traceback (the useful part)."""
    rows = stderr.strip().splitlines()
    tail = rows[-lines:]
    prefix = "…\n" if len(rows) > lines else ""
    return prefix + "\n".join(tail)


def format_result(result: RunResult) -> str:
    """Markdown of a run for the chat (status line, stdout, stderr tail)."""
    seconds = result.elapsed_ms / 1000
    if result.ok:
        head = f"✅ **Exécution terminée** en {seconds:.1f} s"
    elif result.status == "timeout":
        head = f"⏱️ **Exécution interrompue** (limite de temps ou de mémoire) après {seconds:.1f} s"
    else:
        head = f"❌ **Le script a échoué** (code {result.exit_code}) après {seconds:.1f} s"
    parts = [head]
    if result.stdout.strip():
        parts.append("**Sortie**\n```text\n" + result.stdout.strip()[:MAX_SHOWN_CHARS] + "\n```")
    if result.stderr.strip() and not result.ok:
        parts.append("**Erreur**\n```text\n" + stderr_tail(result.stderr)[:MAX_SHOWN_CHARS] + "\n```")
    if result.images:
        parts.append(f"📊 {len(result.images)} graphique(s) ci-dessous.")
    return "\n\n".join(parts)


def summary_for_model(result: RunResult, *, max_chars: int = 3000) -> str:
    """Text returned to the agent by ``run_python`` (compact, no images)."""
    lines = [f"Statut : {result.status} ({result.elapsed_ms} ms)."]
    if result.stdout.strip():
        lines.append("SORTIE :\n" + result.stdout.strip()[:max_chars])
    if not result.ok and result.stderr.strip():
        lines.append("ERREUR :\n" + stderr_tail(result.stderr)[:max_chars])
    if result.images:
        lines.append(f"{len(result.images)} graphique(s) affiché(s) à l'étudiant.")
    lines.append("La sortie est affichée à l'étudiant : commente-la, ne la recopie pas intégralement.")
    return "\n".join(lines)


def syntax_error(code: str, language: str = "Python") -> str | None:
    """Compile-check a script without running it (``None`` when it parses).

    Catches syntax errors and truncated scripts before any execution, so
    they can be fixed without spending the student's quota.
    """
    if language.lower() != "python":
        return None
    try:
        ast.parse(code)
    except SyntaxError as exc:
        where = f" (ligne {exc.lineno})" if exc.lineno else ""
        return f"SyntaxError{where}: {exc.msg}\n{(exc.text or '').rstrip()}"
    return None


def history_block(history: list[str] | None) -> str:
    """Prompt block listing the errors already fixed in this session (empty when none)."""
    if not history:
        return ""
    lines = "\n".join(f"- {stderr_tail(e, 3)}" for e in history)
    return (
        "\nERREURS DÉJÀ RENCONTRÉES ET CORRIGÉES SUR CE SCRIPT (tentatives précédentes) — "
        "ne les réintroduis pas et garde les corrections correspondantes :\n" + lines
    )


async def fix_code(
    llm: BaseChatModel,
    code: str,
    error: str,
    *,
    topic: str = "",
    language: str = "Python",
    history: list[str] | None = None,
) -> str:
    """Ask the model to repair a failing script (one pass).

    Args:
        llm: Text model.
        code: Failing script.
        error: stderr of the run (or a hint such as :data:`NO_OUTPUT_HINT`).
        topic: Subject of the script.
        language: Script language.
        history: Errors of the previous attempts on this script (regression guard).

    Returns:
        The repaired script.

    Raises:
        ValueError: When the model returned no code block.
    """
    system = render("fix_code", topic=topic or "(non précisé)", language=language, history_block=history_block(history))
    messages = [
        SystemMessage(content=system),
        HumanMessage(
            content=f"SCRIPT :\n```{language.lower()}\n{code}\n```\n\nERREUR :\n```text\n{stderr_tail(error)}\n```"
        ),
    ]
    response = await llm.ainvoke(messages)
    content = response.content if isinstance(response.content, str) else str(response.content)
    fixed = extract_code(content, language.lower())
    if not fixed:
        raise ValueError("the model returned no code block")
    return fixed


# ---------------------------------------------------------------- interpretation of a run

_DIGITS_RE = re.compile(r"\d")
_FIG_TITLE_RE = re.compile(r"(?:set_title|plt\.title|suptitle)\(\s*(?:f?r?)(['\"])(.+?)\1", re.DOTALL)
NO_OUTPUT_HINT = (
    "Le script ne produit aucun résultat chiffré : il n'imprime rien (ou seulement du texte). "
    "Ajoute les print() nécessaires — tableau des estimations (paramètre, vraie valeur si simulée, estimation, "
    "écart-type, t-stat) et statistiques utiles — sans changer le reste du script."
)


@dataclass
class Interpretation:
    """Interpretation of a run: Markdown text and the passages it cites."""

    text: str
    cited: list[SourceRef] = field(default_factory=list)
    needs_output: bool = False  # the script printed no number: nothing to interpret


def has_numbers(stdout: str) -> bool:
    """Whether the output holds something to interpret (at least one digit)."""
    return bool(_DIGITS_RE.search(stdout))


def figure_titles(code: str, n_images: int) -> str:
    """Describe the figures for the prompt: titles found in the code, else their count."""
    if n_images == 0:
        return "aucune"
    titles = [m.group(2) for m in _FIG_TITLE_RE.finditer(code)]
    if titles:
        listed = "; ".join(f"figure {i} « {t} »" for i, t in enumerate(titles[:n_images], start=1))
        return f"{n_images} — {listed}" if len(titles) >= n_images else f"{n_images} (titres repérés : {listed})"
    return f"{n_images} (sans titre repéré)"


def _format_refs(refs: list[SourceRef]) -> str:
    return "\n\n".join(f"[{r.sid}] {r.breadcrumb}\n{r.content}" for r in refs) or "(aucun extrait)"


async def interpret_run(
    llm: BaseChatModel,
    retriever,
    code: str,
    result: RunResult,
    *,
    topic: str,
    level: str = "ALL",
    language: str = "Python",
    registry: SourceRegistry | None = None,
) -> Interpretation:
    """Interpret the real output of a run against the course (one model call).

    Args:
        llm: Text model.
        retriever: Level-scoped retriever (course passages on ``topic``).
        code: Script that ran.
        result: Its result (must be ``ok``).
        topic: Subject of the script.
        level: Student level.
        language: Script language.
        registry: Registry receiving the passages (sources panel); a private one otherwise.

    Returns:
        The interpretation; ``needs_output`` is set (and ``text`` explains it) when the
        script printed no number.
    """
    if not has_numbers(result.stdout):
        return Interpretation(
            "ℹ️ Le script n'a imprimé aucun résultat chiffré : il n'y a rien à interpréter au-delà des figures. "
            "Utilisez « 🔧 Compléter le script » pour y ajouter l'impression des estimations et statistiques.",
            needs_output=True,
        )
    registry = registry if registry is not None else SourceRegistry()
    try:
        docs = retrieve(retriever, topic) if topic else []
    except Exception:
        logger.warning("Course retrieval failed for the interpretation", exc_info=True)
        docs = []
    refs = registry.register_all(docs, origin="search", query=topic) if docs else []
    system = render(
        "interpret_run",
        level=level,
        language=language,
        language_lower=language.lower(),
        topic=topic or "(non précisé)",
        code=code[:12_000],
        stdout=result.stdout.strip()[:MAX_SHOWN_CHARS],
        figures=figure_titles(code, len(result.images)),
        context=_format_refs(refs),
    )
    messages = [SystemMessage(content=system), HumanMessage(content="Interprète cette sortie.")]
    response = await llm.ainvoke(messages)
    raw = response.content if isinstance(response.content, str) else str(response.content)
    check = verify_citations(raw, registry.ids())
    if check.invalid:
        logger.warning("Phantom citations removed from the interpretation: %s", check.invalid)
    text = normalize_math(check.text)
    return Interpretation(text, cited=registry.cited(text))
