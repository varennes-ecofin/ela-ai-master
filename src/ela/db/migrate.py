"""Programmatic access to the Alembic migrations (``ela-db``).

The migration scripts live inside the package (``ela/db/migrations``) so
that ``ela-db upgrade`` works from any directory, in Docker included.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from alembic import command
from alembic.config import Config

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def alembic_config() -> Config:
    """Build the Alembic configuration pointing at the packaged scripts."""
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("prepend_sys_path", ".")
    return cfg


def upgrade(revision: str = "head") -> None:
    """Apply migrations up to ``revision`` (default: latest)."""
    command.upgrade(alembic_config(), revision)


def downgrade(revision: str) -> None:
    """Revert migrations down to ``revision`` (e.g. ``-1`` or ``base``)."""
    command.downgrade(alembic_config(), revision)


def current() -> None:
    """Print the current revision of the database."""
    command.current(alembic_config(), verbose=True)


def history() -> None:
    """Print the migration history."""
    command.history(alembic_config(), verbose=False)


def main() -> None:
    """Console entry point (``ela-db``)."""
    parser = argparse.ArgumentParser(description="ELA AI — migrations de base de données (Alembic)")
    sub = parser.add_subparsers(dest="command", required=True)
    p_up = sub.add_parser("upgrade", help="Appliquer les migrations (défaut : jusqu'à la dernière)")
    p_up.add_argument("revision", nargs="?", default="head")
    p_down = sub.add_parser("downgrade", help="Revenir à une révision (ex : -1, base)")
    p_down.add_argument("revision")
    sub.add_parser("current", help="Révision actuelle de la base")
    sub.add_parser("history", help="Historique des migrations")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    try:
        if args.command == "upgrade":
            upgrade(args.revision)
            print("✅ Base de données à jour.")
        elif args.command == "downgrade":
            downgrade(args.revision)
        elif args.command == "current":
            current()
        elif args.command == "history":
            history()
    except ValueError as exc:  # e.g. DATABASE_URL missing
        sys.exit(f"❌ {exc}")


if __name__ == "__main__":
    main()
