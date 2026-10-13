"""Provider-agnostic chat-model factory.

Cerebras, ILaaS and Groq all expose an OpenAI-compatible
``/chat/completions`` endpoint, so a single ``ChatOpenAI`` client
configured with the right ``base_url`` and ``api_key`` is enough to talk
to any of them. Switching provider or model is therefore a matter of
environment variables only (see ``.env.example``).

Roles
-----
* ``chat``   — text-only requests (RAG answers, quiz, code workshop).
* ``vision`` — requests that carry one or more images; routed to a
  multimodal model (``gpt-oss-120b`` has no vision capability).

Usage::

    from ela.llm import get_chat_model
    llm = get_chat_model("chat")            # with fallback if configured
    async for chunk in llm.astream(messages): ...
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Literal, Optional

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ela.config import ProviderName, Settings, get_settings

logger = logging.getLogger(__name__)

Role = Literal["chat", "vision"]

# Providers whose API rejects the ``reasoning_effort`` parameter.
_NO_REASONING_PARAM: frozenset[str] = frozenset({"groq"})

# Image MIME types accepted by the vision models (OpenAI-compatible data URIs).
SUPPORTED_IMAGE_MIMES: frozenset[str] = frozenset(
    {"image/jpeg", "image/png", "image/webp", "image/gif"}
)


def build_chat_model(
    model: str,
    provider: Optional[ProviderName] = None,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    streaming: bool = True,
    settings: Optional[Settings] = None,
) -> ChatOpenAI:
    """Build a ``ChatOpenAI`` client for an OpenAI-compatible provider.

    Args:
        model: Model identifier as exposed by the provider (e.g. ``gpt-oss-120b``).
        provider: Provider name; defaults to ``settings.llm_provider``.
        temperature: Sampling temperature; defaults to ``settings.llm_temperature``.
        max_tokens: Maximum completion tokens; defaults to ``settings.llm_max_tokens``.
        reasoning_effort: Reasoning effort override; defaults to
            ``settings.reasoning_effort``. ``"omit"`` (or ``None`` in settings)
            leaves the parameter out. Ignored for providers that do not
            accept it.
        streaming: Whether to enable token streaming.
        settings: Settings instance (defaults to the cached global one).

    Returns:
        A configured ``ChatOpenAI`` instance.

    Raises:
        ValueError: If the provider has no API key configured.
    """
    settings = settings or get_settings()
    provider = provider or settings.llm_provider
    base_url, api_key = settings.provider_credentials(provider)

    if not api_key:
        raise ValueError(
            f"No API key configured for provider '{provider}'. "
            f"Set {provider.upper()}_API_KEY in your .env file."
        )

    kwargs: dict = {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": settings.llm_temperature if temperature is None else temperature,
        "max_tokens": settings.llm_max_tokens if max_tokens is None else max_tokens,
        "timeout": settings.llm_timeout_seconds,
        "max_retries": settings.llm_max_retries,
        "streaming": streaming,
    }

    effort = settings.reasoning_effort if reasoning_effort is None else reasoning_effort
    if effort and effort != "omit" and provider not in _NO_REASONING_PARAM:
        kwargs["reasoning_effort"] = effort

    logger.info("LLM client: provider=%s model=%s effort=%s", provider, model, effort)
    return ChatOpenAI(**kwargs)


def get_chat_model(
    role: Role = "chat",
    *,
    with_fallback: bool = True,
    settings: Optional[Settings] = None,
    **overrides,
) -> BaseChatModel:
    """Return the chat model for a role, optionally wrapped with a fallback.

    Args:
        role: ``"chat"`` for text-only requests, ``"vision"`` for requests
            containing images.
        with_fallback: Wrap the primary model with ``LLM_MODEL_FALLBACK``
            (chat role only; ``LLM_MODEL_FALLBACK=none`` disables it). The vision role never falls back because the
            fallback model is not guaranteed to be multimodal.
        settings: Settings instance (defaults to the cached global one).
        **overrides: Extra keyword arguments passed to :func:`build_chat_model`
            (``temperature``, ``max_tokens``, ``reasoning_effort``, ``streaming``).

    Returns:
        A runnable chat model (``ChatOpenAI`` or ``RunnableWithFallbacks``).
    """
    settings = settings or get_settings()

    if role == "vision":
        provider = settings.llm_vision_provider or settings.llm_provider
        return build_chat_model(
            settings.llm_model_vision, provider, settings=settings, **overrides
        )

    primary = build_chat_model(
        settings.llm_model_chat, settings.llm_provider, settings=settings, **overrides
    )

    fallback_model = settings.fallback_model
    fallback_provider = settings.llm_fallback_provider or settings.llm_provider
    same_as_primary = (
        fallback_model == settings.llm_model_chat
        and fallback_provider == settings.llm_provider
    )
    if not with_fallback or not fallback_model or same_as_primary:
        return primary

    try:
        fallback = build_chat_model(
            fallback_model, fallback_provider, settings=settings, **overrides
        )
    except ValueError as exc:  # fallback provider not configured: degrade gracefully
        logger.warning("Fallback model disabled: %s", exc)
        return primary

    return primary.with_fallbacks([fallback])


def encode_image_block(path: str | Path, mime: Optional[str] = None) -> dict:
    """Encode an image file as an OpenAI-style ``image_url`` content block.

    Args:
        path: Path to the image file on disk.
        mime: MIME type reported by the upload layer. Guessed from the file
            extension when missing or unknown.

    Returns:
        A dict ``{"type": "image_url", "image_url": {"url": "data:<mime>;base64,..."}}``.

    Raises:
        ValueError: If the MIME type is not a supported image type.
    """
    path = Path(path)
    mime = (mime or "").lower().strip()
    if mime not in SUPPORTED_IMAGE_MIMES:
        guessed, _ = mimetypes.guess_type(path.name)
        mime = (guessed or "").lower()
    if mime not in SUPPORTED_IMAGE_MIMES:
        raise ValueError(
            f"Unsupported image type '{mime or 'unknown'}' for {path.name}. "
            f"Accepted: {', '.join(sorted(SUPPORTED_IMAGE_MIMES))}."
        )

    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}
