"""Offline tests of the accounts pages of the backoffice (lot 5.2).

The ``ela.db`` layer is replaced by an in-memory store; the real-database
flow lives in ``test_backoffice_db.py``.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

from ela.auth import passwords  # noqa: E402
from ela.db import users as users_db  # noqa: E402
from ela.db.users import Account, UserProfile  # noqa: E402
from test_backoffice import PASSWORD, PREFIX, FakeDb, login, make_settings  # noqa: E402


def _account(identifier, role="student", level="M1", quota=50, active=True, has_password=True, today=0) -> Account:
    return Account(
        id=identifier,
        identifier=identifier,
        role=role,
        level=level,
        is_active=active,
        daily_quota=quota,
        has_password=has_password,
        last_login=None,
        created_at="2026-10-22T10:00:00",
        messages_today=today,
    )


class FakeAccounts(FakeDb):
    """FakeDb (login, audit, dashboard) + the account administration functions."""

    def __init__(self) -> None:
        super().__init__()
        self.store: dict[str, Account] = {
            "admin": _account("admin", "admin", "ALL", None),
            "prof": _account("prof", "supervisor", "ALL", None),
            "etu1": _account("etu1", "student", "M1", 50, today=50),
            "etu2": _account("etu2", "student", "M2", 20, active=False),
        }
        self.hashes: dict[str, str] = {i: passwords.hash_password(PASSWORD) for i in self.store}
        self.activity: dict[str, dict[str, int]] = {"etu1": {"threads": 3, "questions": 5}}
        # keep the login/profile fakes of FakeDb consistent with the store
        self.accounts = {}

    # --- authentication fakes (FakeDb) -------------------------------------------------
    def _profile(self, identifier):
        a = self.store.get(identifier)
        if a is None:
            return None
        return UserProfile(a.id, a.identifier, a.role, a.level, a.is_active, a.daily_quota)

    async def authenticate(self, identifier, password):
        p = self._profile(identifier)
        h = self.hashes.get(identifier)
        if p is None or not p.is_active or not h or not passwords.check_password(password, h):
            return None
        return p

    async def get_profile(self, identifier):
        return self._profile(identifier)

    # --- administration -------------------------------------------------------------------
    async def list_accounts(self, *, search=None, role=None, level=None, active=None):
        rows = list(self.store.values())
        if search:
            rows = [a for a in rows if search.lower() in a.identifier]
        if role:
            rows = [a for a in rows if a.role == role]
        if level:
            rows = [a for a in rows if a.level == level]
        if active is not None:
            rows = [a for a in rows if a.is_active == active]
        return rows

    async def get_account(self, identifier):
        return self.store.get(identifier)

    async def create_account(self, identifier, *, password_hash, role="student", level="M1", daily_quota=50):
        if identifier in self.store:
            raise users_db.AccountExistsError(identifier)
        self.store[identifier] = _account(identifier, role, level, None if role == "admin" else daily_quota)
        if password_hash:
            self.hashes[identifier] = password_hash
        return self.store[identifier]

    async def update_account(self, identifier, *, role=None, level=None, daily_quota=users_db._UNSET, is_active=None):
        a = self.store[identifier]
        changes = {}
        if role is not None:
            changes["role"] = role
        if level is not None:
            changes["level"] = level
        if daily_quota is not users_db._UNSET:
            changes["daily_quota"] = daily_quota
        if is_active is not None:
            changes["is_active"] = is_active
        self.store[identifier] = dataclasses.replace(a, **changes)
        return self.store[identifier]

    async def set_password_hash(self, identifier, password_hash):
        self.hashes[identifier] = password_hash
        self.store[identifier] = dataclasses.replace(self.store[identifier], is_active=True, has_password=True)

    async def get_password_hash(self, identifier):
        return self.hashes.get(identifier)

    async def count_admins(self, active_only=True):
        return sum(1 for a in self.store.values() if a.role == "admin" and (a.is_active or not active_only))

    async def activity_summary(self, identifier):
        base = {"threads": 0, "questions": 0, "quiz_answers": 0, "exercise_attempts": 0, "code_runs": 0}
        base.update(self.activity.get(identifier, {}))
        return base

    async def purge_account(self, identifier):
        if identifier not in self.store:
            raise users_db.AccountNotFoundError(identifier)
        if any((await self.activity_summary(identifier)).values()):
            raise users_db.AccountHasActivityError(identifier)
        del self.store[identifier]


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeAccounts:
    import ela.db.audit as audit_mod
    import ela.db.dashboard as dashboard_mod

    monkeypatch.setattr(passwords, "BCRYPT_ROUNDS", 4)
    db = FakeAccounts()
    for name in (
        "authenticate",
        "get_profile",
        "list_accounts",
        "get_account",
        "create_account",
        "update_account",
        "set_password_hash",
        "get_password_hash",
        "count_admins",
        "activity_summary",
        "purge_account",
    ):
        monkeypatch.setattr(users_db, name, getattr(db, name))
    monkeypatch.setattr(audit_mod, "record", db.record)
    monkeypatch.setattr(audit_mod, "recent", db.recent)
    monkeypatch.setattr(dashboard_mod, "overview", db.overview_fn)
    return db


@pytest.fixture
async def client(fake):
    from backoffice.main import create_app

    app = create_app(make_settings())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", follow_redirects=False) as c:
        yield c


async def _admin(client) -> str:
    r = await login(client, "admin")
    assert r.status_code == 303
    return client.cookies["ela_bo_csrf"]


def _actions(fake: FakeAccounts) -> list[str]:
    return [a for a, _, _ in fake.audit]


# --- list -------------------------------------------------------------------------------


async def test_list_and_filters(client, fake):
    await _admin(client)
    r = await client.get(f"{PREFIX}/users")
    assert r.status_code == 200
    assert "4 compte(s)" in r.text and "quota atteint" in r.text and "Nouveau compte" in r.text
    r = await client.get(f"{PREFIX}/users?q=etu&state=active", headers={"HX-Request": "true"})
    assert r.status_code == 200 and "<html" not in r.text and "1 compte(s)" in r.text and "etu2" not in r.text
    r = await client.get(f"{PREFIX}/users?role=supervisor")
    assert "prof" in r.text and "etu1" not in r.text


async def test_supervisor_is_read_only(client, fake):
    await login(client, "prof")
    r = await client.get(f"{PREFIX}/users")
    assert r.status_code == 200 and "Nouveau compte" not in r.text and "Quota" not in r.text
    r = await client.get(f"{PREFIX}/users/etu1")
    assert r.status_code == 200 and "Modifier" not in r.text and "Supprimer" not in r.text
    assert (await client.get(f"{PREFIX}/users/new")).status_code == 403
    assert (await client.get(f"{PREFIX}/users/import")).status_code == 403
    token = client.cookies["ela_bo_csrf"]
    r = await client.post(
        f"{PREFIX}/users/etu1", data={"csrf_token": token, "role": "admin", "level": "M1", "quota": "0"}
    )
    assert r.status_code == 403 and fake.store["etu1"].role == "student"


# --- create -----------------------------------------------------------------------------


async def test_create_with_typed_password(client, fake):
    token = await _admin(client)
    data = {
        "csrf_token": token,
        "identifier": "Jean.Dupont@etu.univ-orleans.fr",
        "role": "student",
        "level": "M2",
        "quota": "30",
        "password": "Mdp-initial-7",
    }
    r = await client.post(f"{PREFIX}/users/new", data=data)
    assert r.status_code == 303 and r.headers["location"] == f"{PREFIX}/users/jean.dupont?ok=created"
    a = fake.store["jean.dupont"]
    assert (
        a.level == "M2"
        and a.daily_quota == 30
        and passwords.check_password("Mdp-initial-7", fake.hashes["jean.dupont"])
    )
    assert _actions(fake)[-1] == "user.create" and fake.audit[-1][2]["generated_password"] is False
    r = await client.get(r.headers["location"])
    assert r.status_code == 200 and "Compte créé." in r.text and "Mot de passe généré" not in r.text


async def test_create_with_generated_password_shown_once(client, fake):
    token = await _admin(client)
    data = {
        "csrf_token": token,
        "identifier": "marie.martin",
        "role": "student",
        "level": "M1",
        "quota": "0",
        "generate": "1",
    }
    r = await client.post(f"{PREFIX}/users/new", data=data)
    assert r.status_code == 303 and "&secret=" in r.headers["location"]
    assert fake.store["marie.martin"].daily_quota is None
    r = await client.get(r.headers["location"])
    assert "Mot de passe généré" in r.text
    secret = r.text.split('class="secret-value">')[1].split("<")[0]
    assert passwords.is_acceptable(secret) and passwords.check_password(secret, fake.hashes["marie.martin"])
    r = await client.get(r.request.url)  # second read: gone
    assert "Mot de passe généré" not in r.text


async def test_create_validation_errors(client, fake):
    token = await _admin(client)
    base = {"csrf_token": token, "role": "student", "level": "M1", "quota": "50"}
    r = await client.post(f"{PREFIX}/users/new", data={**base, "identifier": "etu1", "password": "Mdp-initial-7"})
    assert r.status_code == 409 and "existe déjà" in r.text
    r = await client.post(f"{PREFIX}/users/new", data={**base, "identifier": "ok.name", "password": "weak"})
    assert r.status_code == 400 and "trop faible" in r.text
    r = await client.post(f"{PREFIX}/users/new", data={**base, "identifier": "Bad Name!", "password": "Mdp-initial-7"})
    assert r.status_code == 400 and "invalide" in r.text
    r = await client.post(f"{PREFIX}/users/new", data={**base, "identifier": "ok.name"})
    assert r.status_code == 400 and "Saisissez un mot de passe" in r.text
    assert "ok.name" not in fake.store and "user.create" not in _actions(fake)


# --- edit, password, state, purge ---------------------------------------------------------


async def test_update_role_level_quota(client, fake):
    token = await _admin(client)
    r = await client.post(
        f"{PREFIX}/users/etu1", data={"csrf_token": token, "role": "supervisor", "level": "ALL", "quota": "0"}
    )
    assert r.status_code == 303 and r.headers["location"].endswith("?ok=updated")
    a = fake.store["etu1"]
    assert a.role == "supervisor" and a.level == "ALL" and a.daily_quota is None
    assert fake.audit[-1][0] == "user.update" and fake.audit[-1][2] == {
        "role": "supervisor",
        "level": "ALL",
        "quota": None,
    }
    r = await client.get(f"{PREFIX}/users/nobody")
    assert r.status_code == 404


async def test_self_and_last_admin_protections(client, fake):
    token = await _admin(client)
    r = await client.post(
        f"{PREFIX}/users/admin", data={"csrf_token": token, "role": "student", "level": "ALL", "quota": "0"}
    )
    assert r.status_code == 400 and "propre rôle" in r.text and fake.store["admin"].role == "admin"
    r = await client.post(f"{PREFIX}/users/admin/active", data={"csrf_token": token, "active": "0"})
    assert r.status_code == 400 and "propre compte" in r.text and fake.store["admin"].is_active
    r = await client.post(f"{PREFIX}/users/admin/purge", data={"csrf_token": token})
    assert r.status_code == 400 and "admin" in fake.store
    # a second admin may be demoted only while another active admin remains
    fake.store["boss"] = _account("boss", "admin", "ALL", None)
    r = await client.post(
        f"{PREFIX}/users/boss", data={"csrf_token": token, "role": "student", "level": "M1", "quota": "50"}
    )
    assert r.status_code == 303 and fake.store["boss"].role == "student"
    # the acting admin is always counted, so another admin can be deactivated safely
    fake.store["boss"] = _account("boss", "admin", "ALL", None)
    r = await client.post(f"{PREFIX}/users/boss/active", data={"csrf_token": token, "active": "0"})
    assert r.status_code == 303 and not fake.store["boss"].is_active


async def test_password_reset_typed_and_generated(client, fake):
    token = await _admin(client)
    r = await client.post(f"{PREFIX}/users/etu2/password", data={"csrf_token": token, "password": "weak"})
    assert r.status_code == 400 and "trop faible" in r.text
    r = await client.post(f"{PREFIX}/users/etu2/password", data={"csrf_token": token, "password": "Nouveau-mdp-2"})
    assert r.status_code == 303 and r.headers["location"].endswith("?ok=password")
    assert passwords.check_password("Nouveau-mdp-2", fake.hashes["etu2"]) and fake.store["etu2"].is_active
    assert fake.audit[-1][0] == "user.password_reset" and fake.audit[-1][2] == {"generated": False}
    r = await client.post(f"{PREFIX}/users/etu2/password", data={"csrf_token": token, "generate": "1"})
    assert r.status_code == 303 and "secret=" in r.headers["location"]
    r = await client.get(r.headers["location"])
    secret = r.text.split('class="secret-value">')[1].split("<")[0]
    assert passwords.check_password(secret, fake.hashes["etu2"])


async def test_deactivate_activate_and_purge(client, fake):
    token = await _admin(client)
    r = await client.post(f"{PREFIX}/users/etu1/active", data={"csrf_token": token, "active": "0"})
    assert r.status_code == 303 and not fake.store["etu1"].is_active and fake.audit[-1][0] == "user.deactivate"
    r = await client.post(f"{PREFIX}/users/etu1/active", data={"csrf_token": token, "active": "1"})
    assert r.status_code == 303 and fake.store["etu1"].is_active and fake.audit[-1][0] == "user.activate"
    # purge refused with activity, accepted without
    r = await client.post(f"{PREFIX}/users/etu1/purge", data={"csrf_token": token})
    assert r.status_code == 409 and "activité" in r.text and "etu1" in fake.store
    r = await client.post(f"{PREFIX}/users/etu2/purge", data={"csrf_token": token})
    assert r.status_code == 303 and r.headers["location"].endswith("/users?ok=purged") and "etu2" not in fake.store
    assert fake.audit[-1][0] == "user.purge"
    r = await client.get(f"{PREFIX}/users/etu1")
    assert "Impossible : ce compte a une activité" in r.text


async def test_write_requires_csrf(client, fake):
    await _admin(client)
    r = await client.post(f"{PREFIX}/users/etu1/active", data={"active": "0"})
    assert r.status_code == 403 and fake.store["etu1"].is_active


# --- CSV import -------------------------------------------------------------------------


async def test_import_preview_apply_and_one_time_credentials(client, fake):
    token = await _admin(client)
    csv_bytes = (
        "\ufeffidentifier;level;password\njean.dupont;M1;\n"
        "Marie@etu.univ-orleans.fr;M2;Bon-mdp-1!\netu1;M1;\nbad id;M1;\n"
    ).encode()
    files = {"file": ("etudiants.csv", csv_bytes, "text/csv")}
    r = await client.post(f"{PREFIX}/users/import", data={"csrf_token": token, "level": "", "quota": "25"}, files=files)
    assert r.status_code == 200 and "encore été écrit" in r.text
    assert "jean.dupont" in r.text and "création" in r.text and "ignoré" in r.text and "erreur" in r.text
    assert "jean.dupont" not in fake.store
    apply_url = r.text.split('action="')[
        1 + [i for i, s in enumerate(r.text.split('action="')) if "/apply" in s][0] - 1
    ].split('"')[0]
    assert apply_url.endswith("/apply")

    r = await client.post(apply_url, data={"csrf_token": token})
    assert r.status_code == 200 and "Import terminé" in r.text
    assert fake.store["jean.dupont"].daily_quota == 25 and fake.store["marie"].level == "M2"
    assert passwords.check_password("Bon-mdp-1!", fake.hashes["marie"])
    assert (
        fake.audit[-1][0] == "users.import"
        and fake.audit[-1][2]["created"] == 2
        and fake.audit[-1][2]["generated_passwords"] == 1
    )
    assert (
        "jean.dupont" in r.text
        and "marie" not in r.text.split("Identifiants générés")[1]
        and "affichés une seule fois" in r.text
    )
    download = (
        f"{PREFIX}/users/import/"
        + r.text.split(f"{PREFIX}/users/import/")[1].split("/credentials.csv")[0]
        + "/credentials.csv"
    )

    r = await client.get(download)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    lines = r.text.splitlines()
    assert lines[0] == "identifier;password" and lines[1].startswith("jean.dupont;") and len(lines) == 2
    assert passwords.check_password(lines[1].split(";")[1], fake.hashes["jean.dupont"])
    r = await client.get(download)
    assert r.status_code == 410  # one-time link

    r = await client.post(apply_url, data={"csrf_token": token})
    assert r.status_code == 410 and "expirée" in r.text  # the preview token was consumed


async def test_import_rejects_bad_input(client, fake):
    token = await _admin(client)
    files = {"file": ("x.csv", b"nom;prenom\nDupont;Jean\n", "text/csv")}
    r = await client.post(f"{PREFIX}/users/import", data={"csrf_token": token, "quota": "50"}, files=files)
    assert r.status_code == 200 and "Colonne obligatoire absente" in r.text and "Aucune ligne exploitable" in r.text
    files = {"file": ("x.csv", b"identifier;level\na.b;M1\n", "text/csv")}
    r = await client.post(f"{PREFIX}/users/import", data={"csrf_token": token, "quota": "dix"}, files=files)
    assert r.status_code == 400 and "Quota invalide" in r.text
    r = await client.post(
        f"{PREFIX}/users/import",
        data={"csrf_token": token, "quota": "50"},
        files={"file": ("big.csv", b"x" * (2 * 1024 * 1024 + 1), "text/csv")},
    )
    assert r.status_code == 400 and "volumineux" in r.text


# --- own password -----------------------------------------------------------------------


async def test_own_password_change(client, fake):
    await login(client, "prof")
    token = client.cookies["ela_bo_csrf"]
    r = await client.get(f"{PREFIX}/account/password")
    assert r.status_code == 200 and "Mon mot de passe" in r.text
    bad = {"csrf_token": token, "current": "wrong", "password": "Nouveau-mdp-2", "confirm": "Nouveau-mdp-2"}
    r = await client.post(f"{PREFIX}/account/password", data=bad)
    assert r.status_code == 400 and "actuel incorrect" in r.text
    r = await client.post(f"{PREFIX}/account/password", data={**bad, "current": PASSWORD, "confirm": "other"})
    assert r.status_code == 400 and "diffèrent" in r.text
    r = await client.post(f"{PREFIX}/account/password", data={**bad, "current": PASSWORD})
    assert r.status_code == 303 and r.headers["location"].endswith("?ok=1")
    assert passwords.check_password("Nouveau-mdp-2", fake.hashes["prof"]) and fake.audit[-1][0] == "account.password"
    r = await client.get(r.headers["location"])
    assert "Mot de passe modifié." in r.text
