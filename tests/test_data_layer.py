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
