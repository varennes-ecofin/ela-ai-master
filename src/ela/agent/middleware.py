"""Middleware of the ELA agent.

* :func:`ela_system_prompt` — system prompt rendered per request from the
  versioned template (student level, visible courses, history summary);
* :func:`build_middleware` — the ordered middleware list: prompt, model
  fallback, model-call and tool-call limits, tool error recovery.

Transient network errors are retried by the ``ChatOpenAI`` client itself
(``LLM_MAX_RETRIES``); the fallback middleware handles a provider or model
outage by switching to ``LLM_MODEL_FALLBACK``.
"""

from __future__ import annotations

import logging

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRequest,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    dynamic_prompt,
)
from langchain_core.language_models import BaseChatModel

from ela import prompts as legacy_prompts
from ela.agent.context import TurnContext
from ela.agent.outline import get_outlines
from ela.agent.prompts import render
from ela.config import Settings, get_settings
from ela.retrieval.retriever import UNFILTERED_LEVELS

logger = logging.getLogger(__name__)


def courses_block(level: str) -> str:
    """One line per course visible to the level, for the system prompt."""
    try:
        outlines = get_outlines(level)
    except Exception as exc:  # store unreachable: the agent still works, without the listing
        logger.warning("Course listing unavailable for the prompt: %s", exc)
        return "(liste des cours indisponible — utilise get_course_outline)"
    if not outlines:
        return "(aucun cours ingéré pour ce niveau)"
    lines = []
    for course, outline in outlines.items():
        chapters = "; ".join(ch.title for ch in outline.chapters[:10])
        files = ", ".join(outline.sources)
        lines.append(f"- {course} — fichiers {files} ({outline.level}, {outline.n_slides} slides) : {chapters}")
    return "\n".join(lines)


def level_block(level: str) -> str:
    """Level-specific instructions (M1 / M2), empty for unscoped levels."""
    return legacy_prompts.level_instructions(level).strip()


def access_block(level: str) -> str:
    """Describe what the student may see, for levels restricted to their own courses."""
    if level in UNFILTERED_LEVELS:
        return ""
    return (
        f"L'étudiant est en {level} : seuls les cours de ce niveau (et les documents communs) lui sont accessibles ; "
        "les autres cours du master lui sont masqués. Si un sujet du domaine (ex. GARCH, VaR pour un M1) n'apparaît "
        "pas dans ses cours, dis que ce sujet n'est pas au programme des cours accessibles à son niveau — en précisant "
        "le cours où il est traité si un outil te l'indique — et ne dis jamais que « le cours ne traite pas » ce sujet."
    )


def code_block(enabled: bool | None = None) -> str:
    """Instructions for ``run_python`` (empty when the sandbox is not configured)."""
    if not (get_settings().sandbox_enabled if enabled is None else enabled):
        return ""
    return (
        "CALCUL NUMÉRIQUE\n"
        "- `run_python(code)` exécute un script Python autonome (numpy, pandas, scipy, statsmodels, matplotlib, arch) "
        "dans un bac à sable sans réseau — un appel par réponse. Utilise-le quand l'étudiant demande de calculer, "
        "simuler, estimer ou vérifier numériquement ; fixe une graine, imprime les résultats utiles, laisse les "
        "figures ouvertes (elles sont capturées). La sortie est affichée à l'étudiant : commente-la (ce que montrent "
        "les chiffres, le lien avec le cours), ne la recopie pas et n'invente jamais un résultat non exécuté."
    )


def summary_block(summary: str | None) -> str:
    """Summary of the older part of the conversation, when any."""
    if not summary:
        return ""
    return f"\nRÉSUMÉ DES ÉCHANGES PRÉCÉDENTS (plus anciens que les messages ci-dessous)\n{summary.strip()}"


def render_system_prompt(ctx: TurnContext, *, summary: str | None = None) -> str:
    """Render the system prompt for a turn."""
    return render(
        "system",
        level_block=level_block(ctx.level),
        courses_block=courses_block(ctx.level),
        access_block=access_block(ctx.level),
        code_block=code_block(),
        summary_block=summary_block(summary if summary is not None else getattr(ctx, "summary", None)),
    )


@dynamic_prompt
def ela_system_prompt(request: ModelRequest) -> str:
    """System prompt of the agent, built from the turn context."""
    ctx: TurnContext = request.runtime.context
    return render_system_prompt(ctx)


def build_middleware(
    *,
    fallback: BaseChatModel | None = None,
    settings: Settings | None = None,
) -> list[AgentMiddleware]:
    """Ordered middleware list of the agent.

    Args:
        fallback: Fallback chat model (``None`` disables the fallback).
        settings: Settings override.

    Returns:
        Middleware instances for ``create_agent``.
    """
    s = settings or get_settings()
    middleware: list[AgentMiddleware] = [ela_system_prompt]
    if fallback is not None:
        middleware.append(ModelFallbackMiddleware(fallback))
    middleware.extend(
        [
            ModelCallLimitMiddleware(run_limit=s.agent_max_model_calls, exit_behavior="end"),
            ToolCallLimitMiddleware(run_limit=s.agent_max_tool_calls),
            ToolCallLimitMiddleware(tool_name="search_course", run_limit=s.agent_max_searches),
            ToolCallLimitMiddleware(tool_name="generate_quiz", run_limit=1),
            ToolCallLimitMiddleware(tool_name="generate_exercise", run_limit=1),
            ToolCallLimitMiddleware(tool_name="generate_review_sheet", run_limit=1),
            ToolCallLimitMiddleware(tool_name="run_python", run_limit=1),
            ToolErrorMiddleware(_tool_error_message),
        ]
    )
    return middleware


def _tool_error_message(exc: Exception, request) -> str:
    """Turn an unexpected tool exception into a message the model can act on."""
    name = getattr(request, "tool_call", {}).get("name", "outil") if request is not None else "outil"
    logger.exception("Tool %s raised", name)
    return (
        f"L'outil {name} a échoué ({exc.__class__.__name__}). "
        "Continue avec les informations déjà obtenues ou réessaie autrement."
    )
