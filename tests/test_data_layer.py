"""Integration test of the custom Chainlit data layer against PostgreSQL.

Exercises the real ``SQLAlchemyDataLayer`` of the installed Chainlit version
(user, thread, step, element stored on local disk, feedback, thread resume)
on a schema created by ``ela-db upgrade``. Requires ``TEST_DATABASE_URL``.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

TEST_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_URL, reason="TEST_DATABASE_URL not set")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "chat"))


@pytest.fixture
async def layer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", TEST_URL)
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "files"))
    from ela.config import get_settings
    from ela.db.migrate import upgrade

    get_settings.cache_clear()
    upgrade()

    from chainlit.context import init_http_context
    from data_layer import LocalSQLAlchemyDataLayer, LocalStorageClient

    # The data layer decorators need a Chainlit context (an HTTP one is enough)
    init_http_context()

    dl = LocalSQLAlchemyDataLayer(
        conninfo=get_settings().async_database_url,
        storage_provider=LocalStorageClient(str(tmp_path / "files")),
    )
    yield dl
    await dl.engine.dispose()
    from ela.db import engine as shared_engine

    await shared_engine.dispose()  # activity/quiz persistence may have created the shared engine on this loop


async def test_full_thread_lifecycle(layer, tmp_path: Path):
    from chainlit.element import Image
    from chainlit.types import Feedback
    from chainlit.user import User

    ident = f"pytest_dl_{uuid.uuid4().hex[:6]}"
    user = await layer.create_user(User(identifier=ident))
    assert user is not None and user.identifier == ident

    thread_id = str(uuid.uuid4())
    await layer.update_thread(thread_id, name="🎓 Quiz", user_id=user.id)

    step_id = str(uuid.uuid4())
    await layer.create_step(
        {
            "id": step_id,
            "name": "user",
            "type": "user_message",
            "threadId": thread_id,
            "output": "Qu'est-ce qu'un bruit blanc ?",
            "createdAt": "2026-10-13T20:00:00Z",
            "metadata": {},
        }
    )

    # Element with a local file → stored by LocalStorageClient, served under /local_files
    img = tmp_path / "plot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    element = Image(name="plot.png", path=str(img), display="inline")
    element.for_id = step_id
    element.thread_id = thread_id
    element.mime = "image/png"
    await layer.create_element(element)

    stored = await layer.get_element(thread_id, element.id)
    assert stored is not None and stored["url"].startswith("/local_files/")
    assert (tmp_path / "files" / stored["objectKey"]).exists()

    await layer.upsert_feedback(Feedback(forId=step_id, threadId=thread_id, value=1, comment="utile"))

    thread = await layer.get_thread(thread_id)
    assert thread is not None and thread["name"] == "🎓 Quiz"
    assert [s["output"] for s in thread["steps"]] == ["Qu'est-ce qu'un bruit blanc ?"]
    assert thread["steps"][0]["feedback"]["value"] == 1
    assert thread["elements"] and thread["elements"][0]["url"].startswith("/local_files/")

    await layer.delete_thread(thread_id)
    assert await layer.get_thread(thread_id) is None


async def test_agent_turn_is_persisted_and_resumable(layer, tmp_path: Path):
    """Steps, streamed answer and sources panel of one agent turn survive a thread resume."""
    import asyncio

    import chainlit as cl
    import chainlit.data as cld
    import ui
    from chainlit.context import init_http_context
    from chainlit.user import User

    from ela.agent.citations import SourceRef
    from ela.agent.events import FinalEvent, SourcesEvent, TokenEvent, ToolEndEvent, ToolStartEvent

    cld._data_layer, cld._data_layer_initialized = layer, True
    try:
        ident = f"pytest_turn_{uuid.uuid4().hex[:6]}"
        user = await layer.create_user(User(identifier=ident))
        thread_id = str(uuid.uuid4())
        init_http_context(thread_id=thread_id, user=user)
        await layer.update_thread(thread_id, name="agent", user_id=user.id)
        await cl.Message(content="ADF ?", type="user_message", author=ident).send()

        ref = SourceRef("S1", "c1", "UTS.tex", "UTS", "M1", "Chapitre 2", "", "Test ADF", "contenu $x$", 0.03, None)

        class FakeEngine:
            async def ask(self, question, chat_history=None, images=None, session_id=None, user_id=None, **kw):
                yield ToolStartEvent("c1", "search_course", {"query": "ADF"}, "🔎 Recherche : « ADF »")
                yield ToolEndEvent("c1", "search_course", "1 extrait(s)")
                yield TokenEvent("Le test ADF ")
                yield TokenEvent("[S1].")
                yield SourcesEvent(cited=[ref], all=[ref])
                yield FinalEvent(text="Le test ADF [S1].", citations_valid=["S1"])

        # Chainlit 2.12 runs ``on_message`` inside a ``run`` step: the answer is nested under it,
        # so that step must be persisted too (migration 0003), else the answer vanishes on resume.
        async with cl.Step(name="on_message", type="run") as run_step:
            await ui.run_turn(FakeEngine(), "ADF ?", [], [], session_id=thread_id, user_id=ident)
        await asyncio.sleep(1.0)  # persistence runs in background tasks

        thread = await layer.get_thread(thread_id)
        steps = {(s["type"], s["name"]): s for s in thread["steps"]}
        assert ("user_message", ident) in steps
        assert steps[("run", "on_message")]["id"] == run_step.id
        answer = steps[("assistant_message", "ELA AI 🤖")]
        assert answer["output"] == "Le test ADF [S1]." and answer["parentId"] == run_step.id
        tool = steps[("tool", "🔎 Recherche : « ADF »")]
        assert tool["output"] == "1 extrait(s)" and tool["end"]
        elements = {e["name"]: e for e in thread["elements"]}
        assert elements["S1"]["display"] == "side" and elements["S1"]["forId"] == answer["id"]
        stored = Path(layer.storage_provider.resolve_to_disk_path(elements["S1"]["objectKey"]))
        assert "Test ADF" in stored.read_text(encoding="utf-8")
    finally:
        cld._data_layer, cld._data_layer_initialized = None, False


async def test_late_create_does_not_erase_a_finished_step(layer):
    """send() and update() persist as concurrent tasks; the insert may land last."""
    import uuid as _uuid

    from chainlit.user import User

    user = await layer.create_user(User(identifier=f"pytest_race_{_uuid.uuid4().hex[:6]}"))
    thread_id = str(_uuid.uuid4())
    await layer.update_thread(thread_id, name="race", user_id=user.id)
    step_id = str(_uuid.uuid4())
    base = {"id": step_id, "name": "tool", "type": "tool", "threadId": thread_id, "createdAt": "2026-10-18T08:00:00Z"}
    await layer.create_step({**base, "output": "5 extrait(s)", "end": "2026-10-18T08:00:01Z", "streaming": False})
    await layer.create_step({**base, "output": "", "streaming": False})  # the late initial insert
    thread = await layer.get_thread(thread_id)
    step = next(s for s in thread["steps"] if s["id"] == step_id)
    assert step["output"] == "5 extrait(s)" and step["end"] == "2026-10-18T08:00:01Z"
