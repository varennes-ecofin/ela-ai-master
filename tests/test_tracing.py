"""Tests for the optional Langfuse tracing (no network)."""

from __future__ import annotations

import sys
import types

import pytest

from ela import tracing
from ela.config import get_settings


@pytest.fixture(autouse=True)
def _reset():
    tracing.reset()
    yield
    tracing.reset()


def test_disabled_without_keys():
    assert not tracing.tracing_enabled()
    assert tracing.get_handler() is None
    config = tracing.run_config(run_name="ela.ask", session_id="t1", user_id="alice", tags=["chat"])
    assert "callbacks" not in config
    assert config["run_name"] == "ela.ask" and config["tags"] == ["chat"]
    assert config["metadata"]["langfuse_session_id"] == "t1" and config["metadata"]["prompt_version"] == "v1"
    assert "langfuse_user_id" not in config["metadata"]  # LANGFUSE_SEND_USER_ID is off by default
    tracing.flush()  # no-op


def test_enabled_with_keys_uses_handler(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_HOST", "http://langfuse.local")
    monkeypatch.setenv("LANGFUSE_SEND_USER_ID", "true")
    get_settings.cache_clear()

    class DummyHandler:
        pass

    fake_pkg = types.ModuleType("langfuse")
    fake_langchain = types.ModuleType("langfuse.langchain")
    fake_langchain.CallbackHandler = DummyHandler
    fake_pkg.langchain = fake_langchain
    monkeypatch.setitem(sys.modules, "langfuse", fake_pkg)
    monkeypatch.setitem(sys.modules, "langfuse.langchain", fake_langchain)

    assert tracing.tracing_enabled()
    handler = tracing.get_handler()
    assert isinstance(handler, DummyHandler) and tracing.get_handler() is handler
    config = tracing.run_config(run_name="ela.eval", user_id="alice")
    assert config["callbacks"] == [handler] and config["metadata"]["langfuse_user_id"] == "alice"


def test_handler_failure_is_not_fatal(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    get_settings.cache_clear()
    broken = types.ModuleType("langfuse.langchain")

    class Boom:
        def __init__(self):
            raise RuntimeError("no server")

    broken.CallbackHandler = Boom
    monkeypatch.setitem(sys.modules, "langfuse", types.ModuleType("langfuse"))
    monkeypatch.setitem(sys.modules, "langfuse.langchain", broken)
    assert tracing.get_handler() is None
    assert "callbacks" not in tracing.run_config(run_name="x")
