"""Agent factory (LangChain 1.x ``create_agent`` on LangGraph).

The compiled agent is stateless: every student message is a new run whose
input is the bounded history plus the message, and whose per-turn state
lives in :class:`~ela.agent.context.TurnContext` (passed as runtime
context). One compiled graph is shared by all sessions of the process.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from ela.agent.context import TurnContext
from ela.agent.middleware import build_middleware
from ela.agent.tools import build_tools
from ela.config import Settings, get_settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_shared = None


def build_agent(
    *,
    model: BaseChatModel,
    fallback: BaseChatModel | None = None,
    tools: Sequence[BaseTool] | None = None,
    settings: Settings | None = None,
):
    """Compile the ELA agent.

    Args:
        model: Primary chat model (must support tool calling).
        fallback: Fallback model used when the primary one fails.
        tools: Tool override (defaults to :func:`ela.agent.tools.build_tools`).
        settings: Settings override.

    Returns:
        The compiled LangGraph agent.
    """
    return create_agent(
        model,
        tools=list(tools) if tools is not None else build_tools(),
        middleware=build_middleware(fallback=fallback, settings=settings),
        context_schema=TurnContext,
        name="ela",
    )


def get_agent():
    """Process-wide agent built from the configured models (first call builds it)."""
    global _shared
    if _shared is None:
        with _lock:
            if _shared is None:
                from ela.llm import get_chat_model

                settings = get_settings()
                primary = get_chat_model("chat", with_fallback=False)
                fallback = None
                if settings.fallback_model:
                    try:
                        fallback = get_chat_model("chat", with_fallback=False, settings=_fallback_settings(settings))
                    except ValueError as exc:
                        logger.warning("Agent fallback model disabled: %s", exc)
                _shared = build_agent(model=primary, fallback=fallback, settings=settings)
                logger.info("Agent ready (model=%s, fallback=%s)", settings.llm_model_chat, settings.fallback_model)
    return _shared


def _fallback_settings(settings: Settings) -> Settings:
    """Return a settings view where the chat model is the fallback model."""
    return settings.model_copy(
        update={
            "llm_model_chat": settings.fallback_model,
            "llm_provider": settings.llm_fallback_provider or settings.llm_provider,
        }
    )


def reset_agent() -> None:
    """Forget the shared agent (tests, provider change)."""
    global _shared
    with _lock:
        _shared = None
