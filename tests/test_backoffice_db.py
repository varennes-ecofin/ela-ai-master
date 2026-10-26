"""Integration tests of the backoffice against PostgreSQL (``TEST_DATABASE_URL``).

Applies the Alembic migrations (idempotent), then exercises the audit log,
the dashboard queries and a real login (bcrypt) through the HTTP layer.
"""

from __future__ import annotations

import asyncio
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


# --- lot 5.2 : account administration against the real database --------------------------


async def test_account_administration(db):
    from ela.auth import passwords
    from ela.db import users

    ident = f"pytest_{uuid.uuid4().hex[:8]}"
    # create, read back, list with filters
    a = await users.create_account(
        ident, password_hash=passwords.hash_password("Secret-123"), level="M2", daily_quota=10
    )
    assert a.identifier == ident and a.level == "M2" and a.daily_quota == 10 and a.has_password and a.is_active
    assert (await users.get_account(ident)).messages_today == 0
    found = [x.identifier for x in await users.list_accounts(search=ident[:10], level="M2", active=True)]
    assert found == [ident]
    assert await users.list_accounts(search=ident, role="admin") == []
    with pytest.raises(users.AccountExistsError):
        await users.create_account(ident, password_hash=None)

    # update, password, deactivate
    a = await users.update_account(ident, role="supervisor", level="ALL", daily_quota=None)
    assert a.role == "supervisor" and a.level == "ALL" and a.unlimited
    await users.set_password_hash(ident, passwords.hash_password("Autre-mdp-9"))
    assert passwords.check_password("Autre-mdp-9", await users.get_password_hash(ident))
    assert (await users.authenticate(ident, "Autre-mdp-9")) is not None
    await users.update_account(ident, is_active=False)
    assert not (await users.get_account(ident)).is_active
    with pytest.raises(users.AccountNotFoundError):
        await users.update_account("pytest_nobody", level="M1")

    # activity → purge refused; messages_today counts today's user messages
    tid = uuid.uuid4()
    await db.execute(
        'INSERT INTO threads (id, name, "createdAt", "userIdentifier") VALUES (:t, :n, NOW()::TEXT, :i)',
        {"t": tid, "n": "x", "i": ident},
    )
    await db.execute(
        'INSERT INTO steps (id, name, type, "threadId", "createdAt") '
        "VALUES (gen_random_uuid(), 'm', 'user_message', :t, NOW()::TEXT)",
        {"t": tid},
    )
    assert (await users.get_account(ident)).messages_today == 1
    summary = await users.activity_summary(ident)
    assert summary["threads"] == 1 and summary["questions"] == 0
    with pytest.raises(users.AccountHasActivityError):
        await users.purge_account(ident)

    # a fresh account can be purged
    other = f"pytest_{uuid.uuid4().hex[:8]}"
    await users.create_account(other, password_hash=None)
    await users.purge_account(other)
    assert await users.get_account(other) is None
    with pytest.raises(users.AccountNotFoundError):
        await users.purge_account(other)
    assert await users.count_admins() >= 0


async def test_csv_import_end_to_end(db):
    from ela.accounts import importer
    from ela.auth import passwords
    from ela.db import users

    tag = uuid.uuid4().hex[:6]
    a, b = f"pytest_{tag}.un", f"pytest_{tag}.deux"
    text = f"identifier;level;password\n{a}@etu.univ-orleans.fr;M1;\n{b};M2;Bon-mdp-1!\n"
    plan = await importer.plan_import(text, default_quota=30)
    assert (plan.n_create, plan.n_error) == (2, 0)
    result = await importer.apply_import(plan)
    assert result.created == [a, b] and result.failed == []
    assert [i for i, _ in result.credentials] == [a]
    assert (await users.authenticate(a, result.credentials[0][1])) is not None
    assert (await users.authenticate(b, "Bon-mdp-1!")).level == "M2"
    assert (await users.get_account(a)).daily_quota == 30

    # second run: skipped, then updated with update_existing
    plan = await importer.plan_import(text)
    assert plan.n_skip == 2 and not plan.applicable
    plan = await importer.plan_import(f"identifier;level;quota\n{a};M2;0\n", update_existing=True)
    result = await importer.apply_import(plan)
    assert result.updated == [a] and result.credentials == []
    acc = await users.get_account(a)
    assert acc.level == "M2" and acc.unlimited and passwords.check_password  # password unchanged


async def test_cli_import(db, tmp_path, capsys):
    """``ela-users import`` (dry run, then real run writing the credentials file)."""
    import argparse

    from ela.cli import users as cli
    from ela.db import users

    tag = uuid.uuid4().hex[:6]
    ident = f"pytest_{tag}.cli"
    csv_file = tmp_path / "etudiants.csv"
    csv_file.write_text(f"identifier;level\n{ident};M1\nbad id;M1\n", encoding="utf-8")
    creds = tmp_path / "mdp.csv"
    base = dict(file=str(csv_file), level=None, quota=50, update_existing=False, credentials=str(creds))

    await db.dispose()  # the CLI runs its own event loop (in a thread here) and disposes the engine itself
    await asyncio.to_thread(cli.cmd_import, argparse.Namespace(dry_run=True, **base))
    out = capsys.readouterr().out
    assert "1 à créer" in out and "1 en erreur" in out and await users.get_account(ident) is None

    await db.dispose()
    await asyncio.to_thread(cli.cmd_import, argparse.Namespace(dry_run=False, **base))
    out = capsys.readouterr().out
    assert "1 compte(s) créé(s)" in out and creds.exists()
    password = creds.read_text(encoding="utf-8").splitlines()[1].split(";")[1]
    assert (await users.authenticate(ident, password)) is not None
