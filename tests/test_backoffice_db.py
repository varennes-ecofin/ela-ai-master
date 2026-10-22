"""Integration tests of the backoffice against PostgreSQL (``TEST_DATABASE_URL``).

Applies the Alembic migrations (idempotent), then exercises the audit log,
the dashboard queries and a real login (bcrypt) through the HTTP layer.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import bcrypt
import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

TEST_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_URL, reason="TEST_DATABASE_URL not set")

SECRET = "integration-secret-key-0123456789"
PREFIX = "/backoffice"


@pytest.fixture
async def db(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", TEST_URL)
    monkeypatch.setenv("BACKOFFICE_SECRET_KEY", SECRET)
    from ela.config import get_settings
    from ela.db import engine
    from ela.db.migrate import upgrade

    get_settings.cache_clear()
    upgrade()  # all migrations, 0009_admin_audit included
    yield engine
    for stmt in (
        "DELETE FROM admin_audit WHERE actor LIKE 'pytest_%'",
        'DELETE FROM steps WHERE "threadId" IN (SELECT id FROM threads WHERE "userIdentifier" LIKE \'pytest_%\')',
        "DELETE FROM threads WHERE \"userIdentifier\" LIKE 'pytest_%'",
        "DELETE FROM users WHERE identifier LIKE 'pytest_%'",
    ):
        await engine.execute(stmt)
    await engine.dispose()


async def _create_user(db, identifier: str, role: str, password: str = "s3cret", level: str = "ALL") -> None:
    pwd_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt(4)).decode()
    await db.execute(
        'INSERT INTO users (id, identifier, "createdAt", password_hash, role, level, daily_quota) '
        "VALUES (gen_random_uuid(), :i, NOW()::TEXT, :h, :r, :l, NULL)",
        {"i": identifier, "h": pwd_hash, "r": role, "l": level},
    )


async def test_audit_record_and_recent(db):
    from ela.db import audit

    actor = f"pytest_{uuid.uuid4().hex[:8]}"
    assert await audit.record(audit.LOGIN_OK, actor, ip="10.0.0.1", details={"role": "admin"})
    assert await audit.record("user.create", actor, target="pytest_new", details={"level": "M1"})
    rows = await audit.recent(limit=10, actor=actor)
    assert [r["action"] for r in rows] == ["user.create", "login.ok"]
    assert rows[0]["target"] == "pytest_new" and rows[0]["details"] == {"level": "M1"}
    assert rows[1]["ip"] == "10.0.0.1" and rows[1]["at"].tzinfo is not None


async def test_dashboard_overview(db):
    from ela.db import dashboard

    student = f"pytest_{uuid.uuid4().hex[:8]}"
    await _create_user(db, student, "student", level="M1")
    tid = uuid.uuid4()
    await db.execute(
        'INSERT INTO threads (id, name, "createdAt", "userIdentifier") VALUES (:t, :n, NOW()::TEXT, :i)',
        {"t": tid, "n": "🎓 Quiz : ADF", "i": student},
    )
    for _ in range(3):
        await db.execute(
            'INSERT INTO steps (id, name, type, "threadId", "createdAt") '
            "VALUES (gen_random_uuid(), 'm', 'user_message', :t, NOW()::TEXT)",
            {"t": tid},
        )
    ov = await dashboard.overview(recent_limit=5)
    assert ov.error is None
    assert ov.students >= 1 and ov.users_total >= 1 and ov.threads >= 1
    assert ov.messages_today >= 3 and ov.active_students_7d >= 1
    mine = [t for t in ov.recent_threads if t["identifier"] == student]
    assert mine and mine[0]["messages"] == 3 and mine[0]["name"] == "🎓 Quiz : ADF"


async def test_real_login_flow(db):
    from backoffice.main import create_app
    from ela.config import get_settings
    from ela.db import audit

    admin = f"pytest_{uuid.uuid4().hex[:8]}"
    student = f"pytest_{uuid.uuid4().hex[:8]}"
    await _create_user(db, admin, "admin")
    await _create_user(db, student, "student", level="M2")

    app = create_app(get_settings())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", follow_redirects=False) as client:
        r = await client.get(f"{PREFIX}/login")
        token = r.cookies["ela_bo_csrf"]

        r = await client.post(
            f"{PREFIX}/login", data={"csrf_token": token, "identifier": student, "password": "s3cret"}
        )
        assert r.status_code == 403

        r = await client.post(f"{PREFIX}/login", data={"csrf_token": token, "identifier": admin, "password": "wrong"})
        assert r.status_code == 401

        r = await client.post(f"{PREFIX}/login", data={"csrf_token": token, "identifier": admin, "password": "s3cret"})
        assert r.status_code == 303 and "ela_bo_session" in r.cookies

        r = await client.get(f"{PREFIX}/")
        assert r.status_code == 200 and admin in r.text and "login.ok" in r.text

        r = await client.get(f"{PREFIX}/healthz?deep=1")
        assert r.status_code == 200 and r.json()["database"] == "ok"

        r = await client.post(f"{PREFIX}/logout", data={"csrf_token": token})
        assert r.status_code == 303

    rows = await audit.recent(limit=10, actor=admin)
    assert [row["action"] for row in rows] == ["logout", "login.ok", "login.failed"]
    rows = await audit.recent(limit=10, actor=student)
    assert rows[0]["action"] == "login.failed" and rows[0]["details"]["reason"] == "role"

    # last_login was recorded by the local provider
    row = await db.fetch_one("SELECT last_login FROM users WHERE identifier = :i", {"i": admin})
    assert row["last_login"] is not None
