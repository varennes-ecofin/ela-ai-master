"""Structured (Pydantic) outputs from any chat model, with a text fallback.

OpenAI-compatible providers (Cerebras, ILaaS) support JSON-schema outputs,
but the smaller models and the fake models used in tests do not. The helper
first tries ``with_structured_output`` on the primary model; if the provider
rejects it or the result does not validate, it falls back to asking for
strict JSON in the text and parsing it (with a repair pass for the LaTeX
backslashes models forget to escape).

The model handed over may be wrapped in a ``RunnableWithFallbacks`` (the
shared chat model is): the structured path is tried on the primary model,
the text path keeps the fallbacks.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables.fallbacks import RunnableWithFallbacks
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
# A backslash that does not start a JSON escape the model meant (quote or backslash).
_LONE_BACKSLASH_RE = re.compile(r'\\(?!["\\])')


def extract_json(text: str) -> str:
    """Return the JSON payload of a model answer (fenced block or first object/array).

    Args:
        text: Raw model text.

    Returns:
        A string that should parse as JSON.

    Raises:
        ValueError: When no JSON-looking fragment is found.
    """
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON found in model output")
    start = min(starts)
    closer = "}" if text[start] == "{" else "]"
    end = text.rfind(closer)
    if end <= start:
        raise ValueError("unterminated JSON in model output")
    return text[start : end + 1]


def repair_escapes(payload: str) -> str:
    r"""Double the backslashes a model wrote unescaped (``\\hat`` → ``\\\\hat``).

    Models asked for JSON with LaTeX inside routinely emit ``"$\\hat{\\rho}$"``,
    which is invalid JSON (``\\h``) or silently wrong (``\\r`` becomes a carriage
    return). When the strict parse fails, every backslash not followed by a
    quote or another backslash is treated as literal text.

    Args:
        payload: JSON text that failed to parse.

    Returns:
        The repaired text (parsing may still fail for other reasons).
    """
    return _LONE_BACKSLASH_RE.sub(r"\\\\", payload)


def unwrap_fallbacks(llm: Runnable) -> tuple[Runnable, list[Runnable]]:
    """Split a ``RunnableWithFallbacks`` into ``(primary, fallbacks)`` (identity otherwise)."""
    if isinstance(llm, RunnableWithFallbacks):
        return llm.runnable, list(llm.fallbacks)
    return llm, []


def with_overrides(llm: Runnable, *, max_tokens: int | None = None, reasoning_effort: str | None = None) -> Runnable:
    """Copy of ``llm`` with another completion budget and/or reasoning effort (fallbacks included).

    Fields a model does not have (fakes) are left alone; the reasoning effort
    is only changed on models that already carry one (providers rejecting the
    parameter never had it set).

    Args:
        llm: Chat model, possibly wrapped with fallbacks.
        max_tokens: Completion budget to apply.
        reasoning_effort: Effort to apply (``low`` / ``medium`` / ``high``).

    Returns:
        The model to invoke.
    """

    def _copy(model: Runnable) -> Runnable:
        if not isinstance(model, BaseChatModel):
            return model
        fields = type(model).model_fields
        update: dict = {}
        if max_tokens is not None and "max_tokens" in fields:
            update["max_tokens"] = max_tokens
        if reasoning_effort and "reasoning_effort" in fields and getattr(model, "reasoning_effort", None):
            update["reasoning_effort"] = reasoning_effort
        return model.model_copy(update=update) if update else model

    primary, fallbacks = unwrap_fallbacks(llm)
    primary = _copy(primary)
    if not fallbacks:
        return primary
    return primary.with_fallbacks([_copy(f) for f in fallbacks])


def with_max_tokens(llm: Runnable, max_tokens: int) -> Runnable:
    """Copy of ``llm`` with a larger completion budget (see :func:`with_overrides`)."""
    return with_overrides(llm, max_tokens=max_tokens)


# Models whose provider rejected the structured request format (HTTP 400): text path only.
_structured_rejected: set[str] = set()


def _model_key(model: Runnable) -> str:
    """Identify a model for the rejection memo (class and model name)."""
    name = getattr(model, "model_name", None) or getattr(model, "model", None) or ""
    return f"{type(model).__name__}:{name}"


def _text_of(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def parse_payload(schema: type[T], text: str) -> T:
    """Parse a JSON payload into ``schema``, repairing unescaped backslashes if needed.

    Args:
        schema: Pydantic model.
        text: JSON text (as extracted from the model answer).

    Returns:
        The validated object.

    Raises:
        ValueError: When the text does not parse or does not validate.
    """
    try:
        return schema.model_validate_json(text)
    except ValidationError as first:
        repaired = repair_escapes(text)
        if repaired == text:
            raise ValueError(f"model output does not match {schema.__name__}: {first}") from first
        try:
            obj = schema.model_validate_json(repaired)
        except ValidationError as second:
            raise ValueError(f"model output does not match {schema.__name__}: {second}") from second
        logger.info("JSON payload repaired (unescaped backslashes)")
        return obj


async def ainvoke_structured(
    llm: Runnable,
    schema: type[T],
    messages: list[BaseMessage],
    *,
    method: str = "json_schema",
    max_tokens: int | None = None,
) -> T:
    """Invoke ``llm`` and parse the answer into ``schema``.

    Args:
        llm: Chat model (possibly wrapped with fallbacks).
        schema: Pydantic model describing the expected output.
        messages: Prompt messages.
        method: ``with_structured_output`` method tried first.
        max_tokens: Completion budget for this call (``None`` keeps the model's).

    Returns:
        A validated ``schema`` instance.

    Raises:
        ValueError: When neither the structured nor the text path produced a
            valid object.
    """
    if max_tokens is not None:
        llm = with_max_tokens(llm, max_tokens)
    primary, _ = unwrap_fallbacks(llm)
    key = _model_key(primary)
    if key not in _structured_rejected:
        try:
            structured = primary.with_structured_output(schema, method=method)
            result = await structured.ainvoke(messages)
            if isinstance(result, schema):
                return result
            if isinstance(result, dict):
                return schema.model_validate(result)
            logger.warning("Structured output returned %s; falling back to text parsing", type(result).__name__)
        except (NotImplementedError, ValueError, TypeError, ValidationError, AttributeError) as exc:
            logger.info("Structured output unavailable (%s); falling back to JSON text", exc.__class__.__name__)
        except Exception as exc:  # provider error on the structured path
            if getattr(exc, "status_code", None) == 400:  # the provider rejects the request format: do not retry it
                _structured_rejected.add(key)
                logger.warning("Structured output rejected by the provider for %s; JSON text from now on: %s", key, exc)
            else:
                logger.warning("Structured output failed (%s); falling back to JSON text", exc)

    hint = HumanMessage(
        content=(
            "Réponds UNIQUEMENT par un objet JSON valide (RFC 8259), sans texte autour, "
            f"conforme à ce schéma JSON :\n{json.dumps(schema.model_json_schema(), ensure_ascii=False)}\n"
            "Dans les chaînes : double chaque antislash LaTeX (écris \\\\alpha, \\\\hat) "
            "et n'utilise pas de guillemets doubles."
        )
    )
    response = await llm.ainvoke([*messages, hint])
    return parse_payload(schema, extract_json(_text_of(response)))
