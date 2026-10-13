"""Tests for the bounded conversation history."""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from ela.history import ChatTurn, bounded_history, to_turns


def _thread(n: int) -> list[ChatTurn]:
    turns = []
    for i in range(n):
        turns.append(ChatTurn("user", f"q{i}"))
        turns.append(ChatTurn("assistant", f"a{i}"))
    return turns


def test_to_turns_filters_and_excludes_current():
    msgs = [
        SimpleNamespace(id="1", type="user_message", content="hello"),
        SimpleNamespace(id="2", type="assistant_message", content="hi"),
        SimpleNamespace(id="3", type="tool", content="ignored"),
        SimpleNamespace(id="4", type="user_message", content="current"),
    ]
    turns = to_turns(msgs, exclude_id="4")
    assert turns == [ChatTurn("user", "hello"), ChatTurn("assistant", "hi")]


def test_empty_history():
    assert bounded_history([]) == []


def test_turn_limit_keeps_most_recent_exchanges():
    out = bounded_history(_thread(20), max_turns=3, max_chars=10_000)
    assert len(out) == 6
    assert isinstance(out[0], HumanMessage) and out[0].content == "q17"
    assert isinstance(out[-1], AIMessage) and out[-1].content == "a19"


def test_char_budget_cuts_old_messages_and_keeps_last_user_turn():
    turns = [ChatTurn("user", "x" * 500), ChatTurn("assistant", "y" * 500), ChatTurn("user", "z" * 50)]
    out = bounded_history(turns, max_turns=10, max_chars=100)
    assert [m.content for m in out] == ["z" * 50]


def test_window_never_starts_with_assistant():
    turns = [
        ChatTurn("user", "q0"),
        ChatTurn("assistant", "a0" * 100),
        ChatTurn("user", "q1"),
        ChatTurn("assistant", "a1"),
    ]
    out = bounded_history(turns, max_turns=10, max_chars=150)
    assert isinstance(out[0], HumanMessage)
    assert [m.content for m in out] == ["q1", "a1"]


def test_defaults_come_from_settings():
    out = bounded_history(_thread(50))
    assert len(out) == 24  # HISTORY_MAX_TURNS default = 12 exchanges
