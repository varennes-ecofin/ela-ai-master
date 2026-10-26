"""Bulk account creation from a CSV file (backoffice « Importer » and ``ela-users import``).

Expected file: a header line, then one account per line. Columns (any order,
case-insensitive, separator ``;`` or ``,`` detected, UTF-8 with or without BOM):

* ``identifier`` (required) — login; a full e-mail address is cut before ``@``
  (``jean.dupont@etu.univ-orleans.fr`` → ``jean.dupont``);
* ``level`` (required unless a default is given) — ``M1``, ``M2`` or ``ALL``;
* ``password`` (optional) — initial password; generated when empty;
* ``role`` (optional, default ``student``) and ``quota`` (optional, default from the form).

Two steps: :func:`plan_import` parses and checks everything without writing
(the backoffice shows the result as a preview), then :func:`apply_import`
creates or updates the accounts and returns the generated credentials,
which are shown once and never stored.
"""

from __future__ import annotations

import asyncio
import csv
import io
from dataclasses import dataclass, field

from ela.auth import passwords
from ela.db import users as users_db

REQUIRED_COLUMNS = ("identifier",)
KNOWN_COLUMNS = ("identifier", "level", "password", "role", "quota")
COLUMN_ALIASES = {
    "login": "identifier",
    "identifiant": "identifier",
    "email": "identifier",
    "e-mail": "identifier",
    "mail": "identifier",
    "niveau": "level",
    "mot_de_passe": "password",
    "mdp": "password",
    "mot de passe": "password",
    "rôle": "role",
    "role": "role",
    "quota_journalier": "quota",
}

CREATE = "create"
UPDATE = "update"
SKIP = "skip"
ERROR = "error"


@dataclass
class ImportRow:
    """One line of the file after validation."""

    line: int
    identifier: str
    level: str | None = None
    role: str = "student"
    quota: int | None = None
    password: str | None = None
    action: str = ERROR
    message: str = ""
    raw_identifier: str = ""

    @property
    def generated_password(self) -> bool:
        """Whether a password will be generated at apply time (new accounts without one)."""
        return self.action == CREATE and not self.password


@dataclass
class ImportPlan:
    """Outcome of :func:`plan_import`: what would happen, line by line."""

    rows: list[ImportRow] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # file-level errors (no usable header...)
    update_existing: bool = False

    def count(self, action: str) -> int:
        """Count the rows with the given action."""
        return sum(1 for r in self.rows if r.action == action)

    @property
    def n_create(self) -> int:
        """Rows that will create an account."""
        return self.count(CREATE)

    @property
    def n_update(self) -> int:
        """Rows that will update an existing account."""
        return self.count(UPDATE)

    @property
    def n_skip(self) -> int:
        """Rows ignored (existing account, duplicates)."""
        return self.count(SKIP)

    @property
    def n_error(self) -> int:
        """Rows rejected."""
        return self.count(ERROR)

    @property
    def applicable(self) -> bool:
        """Whether there is at least one row to create or update."""
        return self.n_create + self.n_update > 0


@dataclass
class ImportResult:
    """Outcome of :func:`apply_import`."""

    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    credentials: list[tuple[str, str]] = field(default_factory=list)  # (identifier, generated password)

    def credentials_csv(self) -> str:
        """Render the generated credentials as a CSV text (``;`` separator, UTF-8)."""
        buffer = io.StringIO()
        writer = csv.writer(buffer, delimiter=";", lineterminator="\n")
        writer.writerow(["identifier", "password"])
        writer.writerows(self.credentials)
        return buffer.getvalue()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _normalise_header(name: str) -> str:
    key = (name or "").strip().lstrip("﻿").lower()
    return COLUMN_ALIASES.get(key, key)


def parse_csv(text: str) -> tuple[list[str], list[dict[str, str]], list[str]]:
    """Parse the CSV text.

    Returns:
        ``(columns, records, errors)`` — normalised column names, one dict per
        data line (keys = columns, values stripped) and file-level errors.
    """
    text = text.lstrip("﻿")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return [], [], ["Fichier vide."]
    try:
        dialect = csv.Sniffer().sniff(lines[0], delimiters=";,\t")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ";" if lines[0].count(";") >= lines[0].count(",") else ","
    reader = csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter)
    header = [_normalise_header(h) for h in next(reader)]
    errors: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        errors.append(
            "Colonne obligatoire absente : " + ", ".join(missing) + f" (colonnes lues : {', '.join(header) or '—'})."
        )
        return header, [], errors
    records: list[dict[str, str]] = []
    for values in reader:
        record = {col: (values[i].strip() if i < len(values) else "") for i, col in enumerate(header) if col}
        records.append(record)
    return header, records, errors


def _parse_quota(value: str, default: int | None) -> tuple[int | None, str | None]:
    if value == "":
        return default, None
    low = value.lower()
    if low in {"0", "illimité", "illimite", "unlimited", "∞", "none"}:
        return None, None
    try:
        quota = int(value)
    except ValueError:
        return default, f"quota invalide « {value} »"
    if quota < 0 or quota > 10_000:
        return default, f"quota hors limites « {value} »"
    return quota, None


# ---------------------------------------------------------------------------
# Plan and apply
# ---------------------------------------------------------------------------


async def plan_import(
    text: str,
    *,
    default_level: str | None = None,
    default_quota: int | None = 50,
    default_role: str = "student",
    update_existing: bool = False,
) -> ImportPlan:
    """Validate the file against the database without writing anything.

    Args:
        text: Decoded CSV content.
        default_level: Level used when the column is absent or empty (``None`` = required).
        default_quota: Daily quota for rows without a ``quota`` column/value.
        default_role: Role for rows without a ``role`` value.
        update_existing: Update role/level/quota (and password when given) of existing accounts
            instead of skipping them.

    Returns:
        An :class:`ImportPlan` with one :class:`ImportRow` per data line.
    """
    columns, records, errors = parse_csv(text)
    plan = ImportPlan(columns=columns, errors=errors, update_existing=update_existing)
    if errors:
        return plan
    seen: set[str] = set()
    for index, record in enumerate(records, start=2):
        raw = record.get("identifier", "")
        identifier = users_db.normalize_identifier(raw)
        row = ImportRow(line=index, identifier=identifier, raw_identifier=raw)
        problem = users_db.identifier_problem(identifier)
        if problem:
            row.message = problem
            plan.rows.append(row)
            continue
        if identifier in seen:
            row.action, row.message = SKIP, "doublon dans le fichier"
            plan.rows.append(row)
            continue
        seen.add(identifier)

        existing = await users_db.get_account(identifier)
        updating = existing is not None and update_existing

        # For an existing account (update mode) an empty cell keeps the current value.
        level = (
            (record.get("level") or "").strip().upper() or (default_level or "") or (existing.level if updating else "")
        )
        if level not in users_db.LEVELS:
            row.message = "niveau manquant ou invalide (M1, M2 ou ALL)" if level else "niveau manquant (M1 ou M2)"
            plan.rows.append(row)
            continue
        row.level = level

        role = (record.get("role") or "").strip().lower() or (existing.role if updating else default_role)
        if role not in users_db.ROLES:
            row.message = f"rôle invalide « {role} »"
            plan.rows.append(row)
            continue
        row.role = role

        quota_cell = record.get("quota", "")
        if updating and quota_cell == "":
            row.quota, quota_error = existing.daily_quota, None
        else:
            row.quota, quota_error = _parse_quota(quota_cell, default_quota)
        if quota_error:
            row.message = quota_error
            plan.rows.append(row)
            continue
        if role == "admin":
            row.quota = None

        password = record.get("password", "") or None
        if password and not passwords.is_acceptable(password):
            row.message = "mot de passe trop faible : " + ", ".join(passwords.problems(password))
            plan.rows.append(row)
            continue
        row.password = password

        if existing is None:
            row.action, row.message = CREATE, "nouveau compte"
        elif update_existing:
            row.action = UPDATE
            row.message = "compte existant : rôle, niveau et quota mis à jour" + (
                ", mot de passe remplacé" if password else ""
            )
        else:
            row.action, row.message = SKIP, "compte existant (ignoré)"
        plan.rows.append(row)
    return plan


async def apply_import(plan: ImportPlan) -> ImportResult:
    """Create and update the accounts of an :class:`ImportPlan`.

    Passwords are generated for rows without one; bcrypt hashing runs in a
    worker thread. A row that fails (race with another administrator, for
    instance) is reported and the others are still processed.
    """
    result = ImportResult()
    for row in plan.rows:
        if row.action not in (CREATE, UPDATE):
            continue
        password = row.password
        generated = False
        if row.action == CREATE and not password:
            password, generated = passwords.generate(), True
        password_hash = await asyncio.to_thread(passwords.hash_password, password) if password else None
        try:
            if row.action == CREATE:
                await users_db.create_account(
                    row.identifier,
                    password_hash=password_hash,
                    role=row.role,
                    level=row.level or "M1",
                    daily_quota=row.quota,
                )
                result.created.append(row.identifier)
            else:
                await users_db.update_account(
                    row.identifier, role=row.role, level=row.level, daily_quota=row.quota, is_active=True
                )
                if password_hash:
                    await users_db.set_password_hash(row.identifier, password_hash)
                result.updated.append(row.identifier)
            if generated and password:
                result.credentials.append((row.identifier, password))
        except users_db.AccountExistsError:
            result.failed.append((row.identifier, "compte créé entre-temps"))
        except Exception as exc:  # keep going with the other rows
            result.failed.append((row.identifier, f"{exc.__class__.__name__}: {exc}"))
    return result
