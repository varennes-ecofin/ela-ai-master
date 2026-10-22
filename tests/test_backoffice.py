"""Offline tests of the FastAPI backoffice (lot 5.1): sessions, CSRF, roles, pages.

The database layer is replaced by in-memory fakes; the integration test
with a real PostgreSQL lives in ``test_backoffice_db.py``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

from ela.config import Settings  # noqa: E402
from ela.db.dashboard import Overview  # noqa: E402
from ela.db.users import UserProfile  # noqa: E402

SECRET = "unit-test-secret-key-0123456789"
PREFIX = "/backoffice"

ACCOUNTS = {
    "admin": UserProfile(id="1", identifier="admin", role="admin", level="ALL", is_active=True, daily_quota=None),
    "prof": UserProfile(id="2", identifier="prof", role="supervisor", level="ALL", is_active=True, daily_quota=None),
    "etu1": UserProfile(id="3", identifier="etu1", role="student", level="M1", is_active=True, daily_quota=50),
}
PASSWORD = "secret123"


class FakeDb:
    """Replace the ``ela.db`` functions used by the backoffice."""

    def __init__(self) -> None:
        self.accounts = dict(ACCOUNTS)
        self.audit: list[tuple[str, str, dict]] = []
        self.overview = Overview(users_total=3, users_active=3, students=1, threads=2, messages_today=4)

    async def authenticate(self, identifier: str, password: str):
        profile = self.accounts.get(identifier)
        if profile is None or not profile.is_active or password != PASSWORD:
            return None
        return profile

    async def get_profile(self, identifier: str):
        return self.accounts.get(identifier)

    async def record(self, action: str, actor: str, *, target=None, details=None, ip=None) -> bool:
        self.audit.append((action, actor, dict(details or {})))
        return True

    async def recent(self, limit: int = 50, actor: str | None = None):
        return [
            {"id": i, "at": "2026-10-21", "actor": a, "action": act, "target": None, "details": d, "ip": "1.2.3.4"}
            for i, (act, a, d) in enumerate(reversed(self.audit))
        ][:limit]

    async def overview_fn(self, recent_limit: int = 10):
        return self.overview


@pytest.fixture
def fake_db(monkeypatch: pytest.MonkeyPatch) -> FakeDb:
    import ela.db.audit as audit_mod
    import ela.db.dashboard as dashboard_mod
    import ela.db.users as users_mod

    db = FakeDb()
    monkeypatch.setattr(users_mod, "authenticate", db.authenticate)
    monkeypatch.setattr(users_mod, "get_profile", db.get_profile)
    monkeypatch.setattr(audit_mod, "record", db.record)
    monkeypatch.setattr(audit_mod, "recent", db.recent)
    monkeypatch.setattr(dashboard_mod, "overview", db.overview_fn)
    return db


def make_settings(**overrides) -> Settings:
    values = {
        "backoffice_secret_key": SECRET,
        "database_url": "postgresql://u:p@localhost:5432/unused",
        "backoffice_login_max_failures": 3,
        "backoffice_login_lock_minutes": 10,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def app(fake_db):
    from backoffice.main import create_app

    return create_app(make_settings())


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", follow_redirects=False) as c:
        yield c


async def get_csrf(client: httpx.AsyncClient) -> str:
    """Open the login page and return the CSRF token (cookie set on the client)."""
    r = await client.get(f"{PREFIX}/login")
    assert r.status_code == 200
    return r.cookies.get("ela_bo_csrf") or client.cookies["ela_bo_csrf"]  # the cookie is only set once


async def login(
    client: httpx.AsyncClient, identifier: str, password: str = PASSWORD, next_: str = ""
) -> httpx.Response:
    token = await get_csrf(client)
    data = {"csrf_token": token, "identifier": identifier, "password": password}
    if next_:
        data["next"] = next_
    return await client.post(f"{PREFIX}/login", data=data)


# --- application factory -----------------------------------------------------------


def test_create_app_requires_secret(fake_db):
    from backoffice.main import create_app

    with pytest.raises(ValueError, match="BACKOFFICE_SECRET_KEY"):
        create_app(make_settings(backoffice_secret_key=None))
    with pytest.raises(ValueError, match="16 characters"):
        create_app(make_settings(backoffice_secret_key="short"))


def test_prefix_normalisation():
    assert make_settings(backoffice_root_path="/backoffice/").backoffice_prefix == "/backoffice"
    assert make_settings(backoffice_root_path="admin").backoffice_prefix == "/admin"
    assert make_settings(backoffice_root_path="/").backoffice_prefix == ""


# --- health and anonymous access ------------------------------------------------------


async def test_healthz(client, monkeypatch):
    r = await client.get(f"{PREFIX}/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.headers["x-content-type-options"] == "nosniff"

    import backoffice.routes.health as health_mod

    async def boom(sql, params=None):
        raise ConnectionError("db down")

    monkeypatch.setattr(health_mod, "fetch_one", boom)
    r = await client.get(f"{PREFIX}/healthz?deep=1")
    assert r.status_code == 503 and r.json()["status"] == "degraded"


async def test_anonymous_is_redirected_to_login(client):
    r = await client.get(f"{PREFIX}/")
    assert r.status_code == 303 and r.headers["location"] == f"{PREFIX}/login"
    r = await client.get(f"{PREFIX}/partials/overview")
    assert r.headers["location"] == f"{PREFIX}/login?next={PREFIX}/partials/overview"
    r = await client.get(f"{PREFIX}/partials/overview", headers={"HX-Request": "true"})
    assert r.status_code == 401 and r.headers["hx-redirect"].startswith(f"{PREFIX}/login")
    r = await client.get("/")
    assert r.status_code == 307 and r.headers["location"] == f"{PREFIX}/"


async def test_static_files_served(client):
    r = await client.get(f"{PREFIX}/static/htmx.min.js")
    assert r.status_code == 200 and "htmx" in r.text[:200]
    assert "no-store" not in r.headers.get("cache-control", "")


# --- login -----------------------------------------------------------------------


async def test_login_page_sets_csrf_cookie(client):
    r = await client.get(f"{PREFIX}/login")
    assert r.status_code == 200
    assert "ela_bo_csrf" in r.cookies
    assert 'name="csrf_token"' in r.text and "Se connecter" in r.text


async def test_login_without_csrf_is_refused(client, fake_db):
    await get_csrf(client)
    r = await client.post(f"{PREFIX}/login", data={"identifier": "admin", "password": PASSWORD})
    assert r.status_code == 403 and "Jeton de sécurité" in r.text
    assert "ela_bo_session" not in r.cookies
    assert fake_db.audit == []


async def test_login_wrong_password(client, fake_db):
    r = await login(client, "admin", "nope")
    assert r.status_code == 401 and "Identifiants invalides" in r.text
    assert fake_db.audit[-1][0] == "login.failed" and fake_db.audit[-1][2]["reason"] == "credentials"


async def test_login_student_is_forbidden(client, fake_db):
    r = await login(client, "etu1")
    assert r.status_code == 403 and "pas accès" in r.text
    assert fake_db.audit[-1] == ("login.failed", "etu1", {"reason": "role", "role": "student"})


async def test_login_missing_fields(client):
    token = await get_csrf(client)
    r = await client.post(f"{PREFIX}/login", data={"csrf_token": token, "identifier": "", "password": ""})
    assert r.status_code == 400


async def test_login_lock_after_failures(client, fake_db):
    for _ in range(3):
        r = await login(client, "admin", "nope")
    assert r.status_code == 401 and fake_db.audit[-1][2]["locked"] is True
    r = await login(client, "admin")  # right password, but locked
    assert r.status_code == 429 and "verrouillé" in r.text
    assert fake_db.audit[-1][0] == "login.locked"
    assert "ela_bo_session" not in client.cookies


async def test_login_admin_then_home_and_logout(client, fake_db):
    r = await login(client, "admin")
    assert r.status_code == 303 and r.headers["location"] == f"{PREFIX}/"
    cookie = r.cookies["ela_bo_session"]
    assert cookie and "HttpOnly" in r.headers["set-cookie"]
    assert fake_db.audit[-1][0] == "login.ok"

    r = await client.get(f"{PREFIX}/")
    assert r.status_code == 200
    assert "admin" in r.text and "Journal d'administration" in r.text and "login.ok" in r.text
    assert "Dernières conversations" in r.text and ">4<" in r.text  # messages today
    assert r.headers["cache-control"] == "no-store"

    r = await client.get(f"{PREFIX}/partials/overview", headers={"HX-Request": "true"})
    assert r.status_code == 200 and "kpi" in r.text and "<html" not in r.text

    # already signed in: the login page goes home
    r = await client.get(f"{PREFIX}/login")
    assert r.status_code == 303 and r.headers["location"] == f"{PREFIX}/"

    r = await client.post(f"{PREFIX}/logout", data={"csrf_token": client.cookies["ela_bo_csrf"]})
    assert r.status_code == 303 and r.headers["location"].endswith("/login?logged_out=1")
    assert fake_db.audit[-1][0] == "logout"
    r = await client.get(f"{PREFIX}/")
    assert r.status_code == 303  # session gone


async def test_supervisor_sees_home_without_audit(client):
    r = await login(client, "prof")
    assert r.status_code == 303
    r = await client.get(f"{PREFIX}/")
    assert r.status_code == 200 and "supervisor" in r.text
    assert "Journal d'administration" not in r.text


async def test_next_redirect_is_restricted_to_the_app(client):
    r = await login(client, "admin", next_="https://evil.example/")
    assert r.headers["location"] == f"{PREFIX}/"
    client.cookies.clear()
    r = await login(client, "admin", next_="//evil.example/")
    assert r.headers["location"] == f"{PREFIX}/"
    client.cookies.clear()
    r = await login(client, "admin", next_=f"{PREFIX}/partials/overview")
    assert r.headers["location"] == f"{PREFIX}/partials/overview"


async def test_tampered_or_foreign_cookie_is_ignored(client, app):
    client.cookies.set("ela_bo_session", "not-a-valid-signature", domain="test", path=PREFIX)
    r = await client.get(f"{PREFIX}/")
    assert r.status_code == 303
    from backoffice.security import Session, SessionSigner

    other = SessionSigner("another-secret-key-another-secret", 3600)
    client.cookies.set("ela_bo_session", other.dump_session(Session("admin", "admin")), domain="test", path=PREFIX)
    r = await client.get(f"{PREFIX}/")
    assert r.status_code == 303


async def test_deactivated_or_demoted_account_loses_access(client, fake_db):
    await login(client, "admin")
    assert (await client.get(f"{PREFIX}/")).status_code == 200
    fake_db.accounts["admin"] = UserProfile("1", "admin", "admin", "ALL", is_active=False, daily_quota=None)
    assert (await client.get(f"{PREFIX}/")).status_code == 303  # back to login
    fake_db.accounts["admin"] = UserProfile("1", "admin", "student", "M1", is_active=True, daily_quota=50)
    r = await client.get(f"{PREFIX}/")
    assert r.status_code == 403 and "Accès refusé" in r.text


async def test_root_prefix_variant(fake_db):
    from backoffice.main import create_app

    app = create_app(make_settings(backoffice_root_path="/"))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/healthz")
        assert r.status_code == 200
        r = await c.get("/", follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/login"


# --- security primitives --------------------------------------------------------------


def test_session_signer_round_trip_and_expiry():
    from backoffice.security import Session, SessionSigner

    signer = SessionSigner(SECRET, max_age_seconds=3600)
    session = Session("prof", "supervisor", level="ALL", provider="local", issued_at=time.time())
    loaded = signer.load_session(signer.dump_session(session))
    assert loaded is not None and loaded.identifier == "prof" and loaded.role == "supervisor"
    assert not loaded.is_admin
    assert signer.load_session(None) is None
    assert signer.load_session(signer.dump_session(session) + "x") is None

    expired = SessionSigner(SECRET, max_age_seconds=0)
    value = expired.dump_session(session)
    time.sleep(1.1)
    assert expired.load_session(value) is None


def test_csrf_tokens():
    from backoffice.security import SessionSigner

    signer = SessionSigner(SECRET, 3600)
    token = signer.new_csrf()
    assert signer.csrf_valid(token, token)
    assert not signer.csrf_valid(token, signer.new_csrf())
    assert not signer.csrf_valid(token, None)
    assert not signer.csrf_valid("forged", "forged")


def test_login_limiter():
    from backoffice.security import LoginLimiter

    limiter = LoginLimiter(max_failures=2, lock_seconds=60)
    assert limiter.locked_for("Admin") == 0
    assert limiter.register_failure("admin") is False
    assert limiter.register_failure("ADMIN") is True  # case-insensitive
    assert limiter.locked_for("admin") > 0
    limiter.reset("admin")
    assert limiter.locked_for("admin") == 0
    assert LoginLimiter(max_failures=0).register_failure("x") is False


# --- auth providers -----------------------------------------------------------------


async def test_local_provider_and_factory(fake_db, monkeypatch):
    from ela.auth import AuthError, Credentials, build_auth_provider, get_auth_provider

    provider = build_auth_provider(make_settings())
    assert provider.name == "local"
    identity = await provider.authenticate(Credentials(identifier=" admin ", password=PASSWORD))
    assert identity is not None and identity.identifier == "admin" and identity.attributes["role"] == "admin"
    assert await provider.authenticate(Credentials(identifier="admin", password="bad")) is None
    assert await provider.authenticate(Credentials(identifier="admin")) is None

    with pytest.raises(AuthError):
        build_auth_provider(make_settings(auth_provider="cas"))

    monkeypatch.setenv("BACKOFFICE_SECRET_KEY", SECRET)
    assert get_auth_provider().name == "local"
