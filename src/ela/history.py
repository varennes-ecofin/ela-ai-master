"""Bounded conversation history.

The full thread used to be sent to the model at every turn. This module
keeps the most recent exchanges within a turn count and a character
budget. Phase 3 adds summarisation of the dropped part via middleware.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ela.config import get_settings


@dataclass(frozen=True)
class ChatTurn:
    """One message of the thread as stored by Chainlit."""

    role: str  # "user" | "assistant"
    content: str


def to_turns(messages: Iterable, *, exclude_id: str | None = None) -> list[ChatTurn]:
    """Convert Chainlit messages (``cl.chat_context.get()``) into turns.

    Args:
        messages: Chainlit ``Message`` objects (need ``id``, ``type``, ``content``).
        exclude_id: Message id to skip (the message being answered).

    Returns:
        Ordered list of user/assistant turns; other step types are ignored.
    """
    turns: list[ChatTurn] = []
    for m in messages:
        if exclude_id is not None and getattr(m, "id", None) == exclude_id:
            continue
        kind = getattr(m, "type", None)
        content = getattr(m, "content", "") or ""
        if kind == "user_message":
            turns.append(ChatTurn("user", content))
        elif kind == "assistant_message":
            turns.append(ChatTurn("assistant", content))
    return turns


def bounded_history(
    turns: list[ChatTurn],
    *,
    max_turns: int | None = None,
    max_chars: int | None = None,
) -> list[BaseMessage]:
    """Keep the most recent turns within the configured budgets.

    The window is cut on exchange boundaries (never starts with an
    assistant message) and always keeps at least the last user turn.

    Args:
        turns: Full ordered history.
        max_turns: Maximum number of *exchanges* (user + assistant pairs);
            defaults to ``HISTORY_MAX_TURNS``.
        max_chars: Character budget for the window; defaults to ``HISTORY_MAX_CHARS``.

    Returns:
        LangChain messages ready to be inserted after the system prompt.
    """
    settings = get_settings()
    max_turns = settings.history_max_turns if max_turns is None else max_turns
    max_chars = settings.history_max_chars if max_chars is None else max_chars

    if not turns:
        return []

    kept: list[ChatTurn] = []
    chars = 0
    exchanges = 0
    for turn in reversed(turns):
        if turn.role == "user":
            exchanges += 1
            if exchanges > max_turns:
                break
        if kept and chars + len(turn.content) > max_chars:
            break
        kept.append(turn)
        chars += len(turn.content)

    kept.reverse()
    # Never start the window with an assistant message
    while kept and kept[0].role != "user":
        kept.pop(0)

    return [HumanMessage(content=t.content) if t.role == "user" else AIMessage(content=t.content) for t in kept]
