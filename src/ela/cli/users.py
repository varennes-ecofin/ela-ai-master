"""Command-line user management for ELA AI (``ela-users``).

Accounts live in the PostgreSQL ``users`` table (bcrypt hashes); this script
is the CLI counterpart of the backoffice "Utilisateurs" page and is handy
when no admin account exists yet or when the backoffice is unreachable.

Usage (virtualenv activated, ``.env`` in the current directory)::

    ela-users check                       # DB reachable? schema migrated?
    ela-users migrate                     # add account columns if missing
    ela-users list                        # list accounts
    ela-users create admin --role admin --level ALL
    ela-users create etudiant1 --role student --level M1 --quota 50
    ela-users set-password etudiant1
    ela-users deactivate etudiant1

The password is asked interactively (never passed on the command line).
The connection string comes from ``DATABASE_URL`` (settings); the driver
suffix is adjusted automatically.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys

import bcrypt
import psycopg2
import psycopg2.extras

from ela.config import get_settings

ROLES = ("student", "supervisor", "admin")
LEVELS = ("M1", "M2", "ALL")
REQUIRED_COLUMNS = {"password_hash", "role", "level", "is_active", "daily_quota", "last_login"}


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def _database_url() -> str:
    """Return a psycopg2-compatible connection string (from ``DATABASE_URL``).

    Raises:
        SystemExit: If ``DATABASE_URL`` is not configured.
    """
    try:
        return get_settings().libpq_database_url
    except ValueError as exc:
        sys.exit(f"❌ {exc}")


def _connect():
    """Open a psycopg2 connection (autocommit off).

    ``PGCLIENTENCODING`` is forced to UTF-8 so that server error messages
    (localised in French on Windows installations) are transcoded before
    psycopg2 decodes them; otherwise a wrong password surfaces as an
    opaque ``UnicodeDecodeError``.
    """
    os.environ.setdefault("PGCLIENTENCODING", "UTF8")
    try:
        return psycopg2.connect(_database_url())
    except psycopg2.OperationalError as exc:
        sys.exit(
            "❌ Connexion PostgreSQL impossible. Vérifiez que le serveur tourne et que "
            f"DATABASE_URL / ADMIN_DATABASE_URL sont corrects.\n   Détail : {exc}"
        )
    except UnicodeDecodeError:
        sys.exit(
            "❌ Connexion PostgreSQL refusée (le serveur a renvoyé une erreur dans un "
            "encodage non UTF-8, typiquement « authentification par mot de passe échouée »).\n"
            "   Vérifiez l'utilisateur, le mot de passe et le nom de base dans DATABASE_URL, "
            "puis rouvrez le terminal pour recharger le .env."
        )


def _hash_password(password: str) -> str:
    """Return the bcrypt hash (cost 12) of a plaintext password."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def _ask_password(confirm: bool = True) -> str:
    """Prompt for a password on the terminal, optionally twice.

    Args:
        confirm: Ask a second time and require both entries to match.

    Returns:
        The password typed by the user.

    Raises:
        SystemExit: If the password is empty or the confirmation differs.
    """
    pwd = getpass.getpass("Mot de passe : ")
    if not pwd:
        sys.exit("❌ Mot de passe vide.")
    if confirm and getpass.getpass("Confirmez     : ") != pwd:
        sys.exit("❌ Les deux saisies diffèrent.")
    return pwd


def _schema_columns(cur) -> set[str]:
    """Return the set of column names of the ``users`` table."""
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'users'")
    rows = cur.fetchall()
    # Works with both tuple cursors and RealDictCursor
    return {row["column_name"] if isinstance(row, dict) else row[0] for row in rows}


def _ensure_migrated(cur) -> None:
    """Abort with a clear message if the Phase-1 columns are missing."""
    missing = REQUIRED_COLUMNS - _schema_columns(cur)
    if missing:
        sys.exit(
            "❌ La table users n'a pas les colonnes "
            + ", ".join(sorted(missing))
            + ".\n   Appliquez d'abord la migration : `ela-users migrate`."
        )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_check(_args) -> None:
    """Report connectivity and schema status."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT version()")
        print("✅ Connexion OK —", cur.fetchone()[0].split(",")[0])
        cols = _schema_columns(cur)
        if not cols:
            sys.exit("❌ Table users absente : la base n'a jamais été initialisée (voir README).")
        missing = REQUIRED_COLUMNS - cols
        if missing:
            print("⚠️  Colonnes manquantes :", ", ".join(sorted(missing)), "→ migration à appliquer.")
            print("   Lancez : ela-db upgrade (ou ela-users migrate)")
            cur.execute("SELECT COUNT(*) FROM users")
            print(f"ℹ️  {cur.fetchone()[0]} compte(s) (aucun mot de passe possible avant migration).")
            return
        print("✅ Schéma users migré (password_hash, role, level, is_active, daily_quota).")
        cur.execute("SELECT COUNT(*), COUNT(password_hash) FROM users")
        total, with_pwd = cur.fetchone()
        print(f"ℹ️  {total} compte(s), dont {with_pwd} avec mot de passe défini.")


def cmd_migrate(_args) -> None:
    """Apply the database migrations (same as ``ela-db upgrade``)."""
    from ela.db.migrate import upgrade

    upgrade()
    print("✅ Migrations appliquées (tables Chainlit + colonnes de comptes).")


def cmd_list(_args) -> None:
    """Print all accounts."""
    with _connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        _ensure_migrated(cur)
        cur.execute(
            """
            SELECT identifier, role, level, is_active, daily_quota,
                   password_hash IS NOT NULL AS has_password, last_login
            FROM users
            ORDER BY CASE role WHEN 'admin' THEN 0 WHEN 'supervisor' THEN 1 ELSE 2 END, identifier
            """
        )
        rows = cur.fetchall()
    if not rows:
        print("Aucun compte.")
        return
    print(f"{'identifiant':<20}{'rôle':<12}{'niveau':<8}{'actif':<7}{'quota':<8}{'mdp':<5}dernière connexion")
    for r in rows:
        quota = "∞" if r["daily_quota"] is None else str(r["daily_quota"])
        print(
            f"{r['identifier']:<20}{r['role']:<12}{r['level']:<8}"
            f"{'oui' if r['is_active'] else 'non':<7}{quota:<8}"
            f"{'oui' if r['has_password'] else 'NON':<5}{r['last_login'] or '-'}"
        )


def cmd_create(args) -> None:
    """Create an account, or update it if the identifier already exists."""
    password = _ask_password()
    quota = None if args.quota == 0 or args.role == "admin" else args.quota
    with _connect() as conn, conn.cursor() as cur:
        _ensure_migrated(cur)
        cur.execute(
            """
            INSERT INTO users (id, identifier, "createdAt", password_hash, role, level, is_active, daily_quota)
            VALUES (gen_random_uuid(), %(identifier)s, NOW()::TEXT, %(hash)s, %(role)s, %(level)s, TRUE, %(quota)s)
            ON CONFLICT (identifier) DO UPDATE
               SET password_hash = EXCLUDED.password_hash,
                   role = EXCLUDED.role,
                   level = EXCLUDED.level,
                   is_active = TRUE,
                   daily_quota = EXCLUDED.daily_quota
            RETURNING (xmax = 0) AS inserted
            """,
            {
                "identifier": args.identifier,
                "hash": _hash_password(password),
                "role": args.role,
                "level": args.level,
                "quota": quota,
            },
        )
        inserted = cur.fetchone()[0]
    verb = "créé" if inserted else "mis à jour"
    print(f"✅ Compte '{args.identifier}' {verb} (rôle={args.role}, niveau={args.level}, quota={quota or '∞'}).")


def cmd_set_password(args) -> None:
    """Reset the password of an existing account."""
    password = _ask_password()
    with _connect() as conn, conn.cursor() as cur:
        _ensure_migrated(cur)
        cur.execute(
            "UPDATE users SET password_hash = %(hash)s, is_active = TRUE WHERE identifier = %(id)s",
            {"hash": _hash_password(password), "id": args.identifier},
        )
        if cur.rowcount == 0:
            sys.exit(f"❌ Compte '{args.identifier}' introuvable (utilisez `ela-users create`).")
    print(f"✅ Mot de passe de '{args.identifier}' mis à jour.")


def cmd_deactivate(args) -> None:
    """Soft-delete an account (login refused, history kept)."""
    with _connect() as conn, conn.cursor() as cur:
        _ensure_migrated(cur)
        cur.execute("UPDATE users SET is_active = FALSE WHERE identifier = %(id)s", {"id": args.identifier})
        if cur.rowcount == 0:
            sys.exit(f"❌ Compte '{args.identifier}' introuvable.")
    print(f"✅ Compte '{args.identifier}' désactivé.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse the command line and dispatch."""
    parser = argparse.ArgumentParser(description="Gestion des comptes ELA AI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Vérifier la connexion et le schéma").set_defaults(func=cmd_check)
    sub.add_parser("list", help="Lister les comptes").set_defaults(func=cmd_list)
    sub.add_parser("migrate", help="Ajouter les colonnes de comptes à la table users (idempotent)").set_defaults(
        func=cmd_migrate
    )

    p_create = sub.add_parser("create", help="Créer (ou mettre à jour) un compte")
    p_create.add_argument("identifier")
    p_create.add_argument("--role", choices=ROLES, default="student")
    p_create.add_argument("--level", choices=LEVELS, default="M1")
    p_create.add_argument(
        "--quota", type=int, default=50, help="messages/jour ; 0 = illimité (toujours illimité pour admin)"
    )
    p_create.set_defaults(func=cmd_create)

    p_pwd = sub.add_parser("set-password", help="Changer le mot de passe d'un compte")
    p_pwd.add_argument("identifier")
    p_pwd.set_defaults(func=cmd_set_password)

    p_off = sub.add_parser("deactivate", help="Désactiver un compte")
    p_off.add_argument("identifier")
    p_off.set_defaults(func=cmd_deactivate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
