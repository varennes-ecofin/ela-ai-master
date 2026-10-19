"""Scripted chat model for agent tests (tool calls, no network)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    """Return scripted ``AIMessage`` objects in order; accepts ``bind_tools``.

    Attributes:
        script: Messages (or strings) returned one per call.
        calls: Prompts received, one list of messages per call.
        fail_first: Raise on the first N calls (fallback / retry tests).
    """

    script: list[Any]
    calls: list[list[BaseMessage]] = []
    fail_first: int = 0
    failures: int = 0

    def __init__(self, script: Iterable[Any], **kwargs):
        super().__init__(script=list(script), calls=[], **kwargs)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        raise NotImplementedError("scripted model: JSON text path only")

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(list(messages))
        if self.failures < self.fail_first:
            self.failures += 1
            raise RuntimeError("scripted failure")
        if not self.script:
            raise RuntimeError("script exhausted")
        item = self.script.pop(0)
        return item if isinstance(item, AIMessage) else AIMessage(content=str(item))

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])


def tool_call_message(*calls: tuple[str, dict], content: str = "") -> AIMessage:
    """Build an ``AIMessage`` carrying tool calls ``(name, args)``."""
    return AIMessage(
        content=content,
        tool_calls=[{"name": name, "args": args, "id": f"call_{i}"} for i, (name, args) in enumerate(calls, start=1)],
    )
