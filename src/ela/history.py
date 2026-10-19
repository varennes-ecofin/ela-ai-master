"""Bounded conversation history with summarisation of the dropped part.

The full thread used to be sent to the model at every turn. The window now
keeps the most recent exchanges (turn count and character budget); the
turns that fall out of the window are summarised once by the model and the
summary is kept per session (:class:`HistorySummarizer`), so that the agent
keeps the thread of a long conversation without re-reading it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from ela.config import get_settings

logger = logging.getLogger(__name__)


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


def split_history(
    turns: list[ChatTurn],
    *,
    max_turns: int | None = None,
    max_chars: int | None = None,
) -> tuple[list[ChatTurn], list[ChatTurn]]:
    """Split the thread into ``(dropped, kept)`` according to the budgets.

    The window is cut on exchange boundaries (never starts with an
    assistant message) and always keeps at least the last user turn.

    Args:
        turns: Full ordered history.
        max_turns: Maximum number of *exchanges* (user + assistant pairs);
            defaults to ``HISTORY_MAX_TURNS``.
        max_chars: Character budget for the window; defaults to ``HISTORY_MAX_CHARS``.

    Returns:
        The turns that fall out of the window and the turns kept, both in order.
    """
    settings = get_settings()
    max_turns = settings.history_max_turns if max_turns is None else max_turns
    max_chars = settings.history_max_chars if max_chars is None else max_chars

    if not turns:
        return [], []

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
    return turns[: len(turns) - len(kept)], kept


def to_messages(turns: Iterable[ChatTurn]) -> list[BaseMessage]:
    """Convert turns into LangChain messages."""
    return [HumanMessage(content=t.content) if t.role == "user" else AIMessage(content=t.content) for t in turns]


def bounded_history(
    turns: list[ChatTurn],
    *,
    max_turns: int | None = None,
    max_chars: int | None = None,
) -> list[BaseMessage]:
    """Keep the most recent turns within the configured budgets.

    Args:
        turns: Full ordered history.
        max_turns: Maximum number of exchanges (defaults to ``HISTORY_MAX_TURNS``).
        max_chars: Character budget (defaults to ``HISTORY_MAX_CHARS``).

    Returns:
        LangChain messages ready to be inserted after the system prompt.
    """
    _, kept = split_history(turns, max_turns=max_turns, max_chars=max_chars)
    return to_messages(kept)


def format_turns(turns: Iterable[ChatTurn], *, max_chars_per_turn: int = 1500) -> str:
    """Render turns as a transcript for the summariser."""
    lines = []
    for t in turns:
        who = "Étudiant" if t.role == "user" else "ELA"
        content = t.content if len(t.content) <= max_chars_per_turn else t.content[:max_chars_per_turn] + " […]"
        lines.append(f"{who} : {content}")
    return "\n\n".join(lines)


class HistorySummarizer:
    """Summary of the turns that left the window, kept per chat session.

    The summary is recomputed only when new turns leave the window (one
    model call at most per such event), never at every message.

    Args:
        llm: Model used to summarise.
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self._llm = llm
        self.summary: str = ""
        self.covered: int = 0  # number of dropped turns the summary accounts for

    async def update(self, dropped: list[ChatTurn]) -> str:
        """Bring the summary up to date with the dropped turns.

        Args:
            dropped: Turns outside the window (full prefix of the thread).

        Returns:
            The current summary (possibly unchanged, possibly empty on failure).
        """
        if len(dropped) <= self.covered:
            return self.summary
        if self._llm is None:
            from ela.engine import chat_llm

            self._llm = chat_llm()
        from ela.agent.prompts import render

        new_turns = dropped[self.covered :]
        prompt = render("summary", previous=self.summary or "(aucun)", messages=format_turns(new_turns))
        try:
            response = await self._llm.ainvoke(
                [SystemMessage(content=prompt), HumanMessage(content="Rédige le résumé mis à jour.")]
            )
            text = response.content if isinstance(response.content, str) else str(response.content)
            self.summary = text.strip()
            self.covered = len(dropped)
            logger.info("History summary updated (%d turns covered)", self.covered)
        except Exception as exc:  # keep the previous summary rather than failing the turn
            logger.warning("History summarisation failed: %s", exc)
        return self.summary
