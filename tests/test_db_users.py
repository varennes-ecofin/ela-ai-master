"""Integration tests for ``ela.db`` (require a PostgreSQL ``TEST_DATABASE_URL``).

Run with, e.g.::

    TEST_DATABASE_URL=postgresql://user:pwd@localhost:5432/ela_test pytest tests/test_db_users.py
"""

from __future__ import annotations

import os
import uuid

import bcrypt
import pytest

TEST_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_URL, reason="TEST_DATABASE_URL not set")


@pytest.fixture
async def db(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", TEST_URL)
    from ela.config import get_settings
    from ela.db import engine

    get_settings.cache_clear()
    await engine.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY, identifier TEXT UNIQUE, "createdAt" TEXT, metadata JSONB,
            password_hash TEXT, role TEXT NOT NULL DEFAULT 'student', level TEXT NOT NULL DEFAULT 'M1',
            is_active BOOLEAN NOT NULL DEFAULT TRUE, daily_quota INTEGER DEFAULT 50, last_login TIMESTAMP)
        """
    )
    await engine.execute(
        'CREATE TABLE IF NOT EXISTS threads (id UUID PRIMARY KEY, name TEXT, "createdAt" TEXT, '
        '"userId" UUID, "userIdentifier" TEXT, tags TEXT[], metadata JSONB)'
    )
    await engine.execute(
        'CREATE TABLE IF NOT EXISTS steps (id UUID PRIMARY KEY, name TEXT, type TEXT, "threadId" UUID, '
        '"createdAt" TEXT, metadata JSONB)'
    )
    yield engine
    for stmt in ("DELETE FROM steps", "DELETE FROM threads", "DELETE FROM users WHERE identifier LIKE 'pytest_%'"):
        await engine.execute(stmt)
    await engine.dispose()  # connections are bound to this test's event loop


async def test_authenticate_profile_and_quota(db):
    from ela.db import users

    ident = f"pytest_{uuid.uuid4().hex[:8]}"
    pwd_hash = bcrypt.hashpw(b"s3cret", bcrypt.gensalt(4)).decode()
    await db.execute(
        'INSERT INTO users (id, identifier, "createdAt", password_hash, role, level, daily_quota) '
        "VALUES (gen_random_uuid(), :i, NOW()::TEXT, :h, 'student', 'M2', 2)",
        {"i": ident, "h": pwd_hash},
    )

    assert await users.authenticate(ident, "wrong") is None
    profile = await users.authenticate(ident, "s3cret")
    assert profile is not None and profile.level == "M2" and profile.daily_quota == 2
    assert (await users.get_profile("nobody")) is None

    status = await users.check_quota(profile)
    assert status.allowed and status.used == 0 and status.limit == 2

    # two messages today → quota reached
    tid = uuid.uuid4()
    await db.execute(
        'INSERT INTO threads (id, name, "createdAt", "userIdentifier") VALUES (:t, \'x\', NOW()::TEXT, :i)',
        {"t": tid, "i": ident},
    )
    for _ in range(2):
        await db.execute(
            'INSERT INTO steps (id, name, type, "threadId", "createdAt") '
            "VALUES (gen_random_uuid(), 'm', 'user_message', :t, NOW()::TEXT)",
            {"t": tid},
        )
    status = await users.check_quota(profile)
    assert not status.allowed and status.used == 2

    # inactive accounts cannot log in
    await db.execute("UPDATE users SET is_active = FALSE WHERE identifier = :i", {"i": ident})
    assert await users.authenticate(ident, "s3cret") is None


async def test_rename_thread_never_raises(db):
    from ela.db import threads

    await threads.rename_thread(str(uuid.uuid4()), "🎓 Quiz")  # unknown id → no error
