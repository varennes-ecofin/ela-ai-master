"""CSV account import (ela.accounts.importer) with a fake accounts store."""

from __future__ import annotations

import dataclasses

import pytest

from ela.accounts import importer
from ela.auth import passwords
from ela.db import users as users_db
from ela.db.users import Account


def _account(identifier: str, role: str = "student", level: str = "M1", quota: int | None = 50) -> Account:
    return Account(
        id=identifier,
        identifier=identifier,
        role=role,
        level=level,
        is_active=True,
        daily_quota=quota,
        has_password=True,
        last_login=None,
        created_at=None,
    )


class FakeStore:
    def __init__(self, *accounts: Account) -> None:
        self.accounts = {a.identifier: a for a in accounts}
        self.hashes: dict[str, str] = {}
        self.fail_on: set[str] = set()

    async def get_account(self, identifier: str):
        return self.accounts.get(identifier)

    async def create_account(self, identifier, *, password_hash, role="student", level="M1", daily_quota=50):
        if identifier in self.fail_on:
            raise RuntimeError("boom")
        if identifier in self.accounts:
            raise users_db.AccountExistsError(identifier)
        self.accounts[identifier] = _account(identifier, role, level, daily_quota)
        if password_hash:
            self.hashes[identifier] = password_hash
        return self.accounts[identifier]

    async def update_account(self, identifier, *, role=None, level=None, daily_quota=users_db._UNSET, is_active=None):
        a = self.accounts[identifier]
        changes = {}
        if role is not None:
            changes["role"] = role
        if level is not None:
            changes["level"] = level
        if daily_quota is not users_db._UNSET:
            changes["daily_quota"] = daily_quota
        if is_active is not None:
            changes["is_active"] = is_active
        self.accounts[identifier] = dataclasses.replace(a, **changes)
        return self.accounts[identifier]

    async def set_password_hash(self, identifier, password_hash):
        self.hashes[identifier] = password_hash


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    s = FakeStore(_account("etu1"), _account("prof", role="supervisor", level="ALL", quota=None))
    for name in ("get_account", "create_account", "update_account", "set_password_hash"):
        monkeypatch.setattr(users_db, name, getattr(s, name))
    monkeypatch.setattr(passwords, "BCRYPT_ROUNDS", 4)
    return s


# --- parsing --------------------------------------------------------------------------


def test_parse_csv_detects_delimiter_bom_and_aliases():
    cols, rows, errors = importer.parse_csv("﻿Email,Niveau\njean@u.fr,M1\n\nmarie@u.fr,M2\n")
    assert errors == [] and cols == ["identifier", "level"]
    assert rows == [{"identifier": "jean@u.fr", "level": "M1"}, {"identifier": "marie@u.fr", "level": "M2"}]
    cols, rows, _ = importer.parse_csv("identifier;level;password\n a ; M1 ; x \n")
    assert rows == [{"identifier": "a", "level": "M1", "password": "x"}]


def test_parse_csv_errors():
    assert importer.parse_csv("   \n")[2] == ["Fichier vide."]
    _, rows, errors = importer.parse_csv("nom;prenom\nDupont;Jean\n")
    assert rows == [] and "identifier" in errors[0]


# --- plan -----------------------------------------------------------------------------


async def test_plan_import_checks_every_line(store):
    text = (
        "identifier;level;password;role;quota\n"
        "Jean.Dupont@etu.univ-orleans.fr;M1;;;\n"
        "marie.martin;m2;Bon-mdp-1!;;20\n"
        "p.durand;M1;faible;;\n"
        "x;M1;;;\n"
        "jean.dupont;M1;;;\n"
        "etu1;M1;;;\n"
        "boss;M1;;admin;\n"
        "nolevel;;;;\n"
        "badrole;M1;;chef;\n"
        "badquota;M1;;;dix\n"
    )
    plan = await importer.plan_import(text)
    by_id = {}
    for r in plan.rows:
        by_id.setdefault(r.identifier, r)  # first occurrence
    assert plan.errors == [] and len(plan.rows) == 10
    assert by_id["jean.dupont"].action == "create" and by_id["jean.dupont"].generated_password
    assert by_id["marie.martin"].action == "create" and by_id["marie.martin"].level == "M2"
    assert by_id["marie.martin"].quota == 20 and by_id["marie.martin"].password == "Bon-mdp-1!"
    assert by_id["p.durand"].action == "error" and "trop faible" in by_id["p.durand"].message
    assert by_id["x"].action == "error" and "invalide" in by_id["x"].message
    assert [r.action for r in plan.rows if r.identifier == "jean.dupont"] == ["create", "skip"]
    assert by_id["etu1"].action == "skip" and "existant" in by_id["etu1"].message
    assert by_id["boss"].action == "create" and by_id["boss"].role == "admin" and by_id["boss"].quota is None
    assert by_id["nolevel"].action == "error" and "niveau" in by_id["nolevel"].message
    assert by_id["badrole"].action == "error" and by_id["badquota"].action == "error"
    assert (plan.n_create, plan.n_update, plan.n_skip, plan.n_error) == (3, 0, 2, 5)
    assert plan.applicable


async def test_plan_import_defaults_and_update_existing(store):
    plan = await importer.plan_import("identifier\netu1\nnew.one\n", default_level="M2", default_quota=None)
    assert [r.action for r in plan.rows] == ["skip", "create"]
    assert plan.rows[1].level == "M2" and plan.rows[1].quota is None

    plan = await importer.plan_import(
        "identifier;level;password\netu1;M2;Nouveau-1!\nprof;ALL;\n", update_existing=True
    )
    assert [r.action for r in plan.rows] == ["update", "update"]
    assert "mot de passe remplacé" in plan.rows[0].message and "remplacé" not in plan.rows[1].message
    assert not plan.rows[1].generated_password  # existing password is kept
    # empty cells keep the current values of an existing account (prof stays supervisor, unlimited)
    plan = await importer.plan_import("identifier;role;quota\nprof;;\n", update_existing=True)
    assert plan.rows[0].action == "update" and plan.rows[0].role == "supervisor"
    assert plan.rows[0].level == "ALL" and plan.rows[0].quota is None


async def test_plan_import_file_error(store):
    plan = await importer.plan_import("foo;bar\n1;2\n")
    assert plan.errors and plan.rows == [] and not plan.applicable


# --- apply ----------------------------------------------------------------------------


async def test_apply_import_creates_updates_and_reports(store):
    text = "identifier;level;password\njean.dupont;M1;\nmarie.martin;M2;Bon-mdp-1!\netu1;M2;\nfails;M1;\n"
    store.fail_on.add("fails")
    plan = await importer.plan_import(text, update_existing=True)
    result = await importer.apply_import(plan)
    assert result.created == ["jean.dupont", "marie.martin"] and result.updated == ["etu1"]
    assert result.failed == [("fails", "RuntimeError: boom")]
    assert [ident for ident, _ in result.credentials] == ["jean.dupont"]
    generated = result.credentials[0][1]
    assert passwords.is_acceptable(generated) and passwords.check_password(generated, store.hashes["jean.dupont"])
    assert passwords.check_password("Bon-mdp-1!", store.hashes["marie.martin"])
    assert store.accounts["etu1"].level == "M2" and "etu1" not in store.hashes
    csv_text = result.credentials_csv()
    assert csv_text.splitlines()[0] == "identifier;password" and f"jean.dupont;{generated}" in csv_text


async def test_apply_import_handles_race(store):
    plan = await importer.plan_import("identifier;level\nnew.one;M1\n")
    await store.create_account("new.one", password_hash="x")  # created meanwhile
    result = await importer.apply_import(plan)
    assert result.created == [] and result.failed == [("new.one", "compte créé entre-temps")]
