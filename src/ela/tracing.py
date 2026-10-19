"""Optional Langfuse tracing.

Tracing is enabled only when ``LANGFUSE_PUBLIC_KEY`` and
``LANGFUSE_SECRET_KEY`` are set (``LANGFUSE_HOST`` for a self-hosted
instance); otherwise every helper is a no-op and the ``langfuse`` package
is never imported. Student identifiers are forwarded only when
``LANGFUSE_SEND_USER_ID`` is true (off by default: no student data leaves
the server unless the teacher decides so).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from ela.config import get_settings

logger = logging.getLogger(__name__)

_handler: Any = None
_checked = False


def tracing_enabled() -> bool:
    """Whether Langfuse keys are configured."""
    s = get_settings()
    return bool(s.langfuse_public_key and s.langfuse_secret_key)


def get_handler() -> Any:
    """Shared Langfuse ``CallbackHandler`` (``None`` when tracing is off or unavailable)."""
    global _handler, _checked
    if _checked:
        return _handler
    _checked = True
    if not tracing_enabled():
        return None
    try:
        import os

        s = get_settings()
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", s.langfuse_public_key.get_secret_value())
        os.environ.setdefault("LANGFUSE_SECRET_KEY", s.langfuse_secret_key.get_secret_value())
        if s.langfuse_host:
            os.environ.setdefault("LANGFUSE_HOST", s.langfuse_host)
        from langfuse.langchain import CallbackHandler

        _handler = CallbackHandler()
        logger.info("Langfuse tracing enabled (%s)", s.langfuse_host or "cloud")
    except Exception as exc:  # package missing or misconfigured: never block the chat
        logger.warning("Langfuse tracing disabled: %s", exc)
        _handler = None
    return _handler


def reset() -> None:
    """Forget the cached handler (tests)."""
    global _handler, _checked
    _handler = None
    _checked = False


def run_config(
    *,
    run_name: str,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> RunnableConfig:
    """Build the ``RunnableConfig`` of one agent run (callbacks + metadata).

    Args:
        run_name: Name of the trace (``ela.ask``, ``ela.eval``...).
        session_id: Chat thread id (Langfuse session).
        user_id: Student identifier (sent only when ``LANGFUSE_SEND_USER_ID`` is true).
        tags: Trace tags.
        metadata: Extra metadata (prompt version, level...).

    Returns:
        A config usable with ``agent.astream(..., config=...)``.
    """
    s = get_settings()
    meta: dict[str, Any] = {"prompt_version": s.prompt_version, **(metadata or {})}
    if session_id:
        meta["langfuse_session_id"] = session_id
    if user_id and s.langfuse_send_user_id:
        meta["langfuse_user_id"] = user_id
    if tags:
        meta["langfuse_tags"] = list(tags)
    config: RunnableConfig = {"run_name": run_name, "metadata": meta, "tags": list(tags or [])}
    handler = get_handler()
    if handler is not None:
        config["callbacks"] = [handler]
    return config


def flush() -> None:
    """Flush pending traces (call at shutdown / end of an evaluation)."""
    if _handler is None:
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as exc:
        logger.debug("Langfuse flush failed: %s", exc)
