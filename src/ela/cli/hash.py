"""Generate bcrypt password hashes (``ela-hash``): ELA accounts or Caddy basic auth.

Usage:
    ela-hash <password> [<password> ...]
    ela-hash --caddy [<password>]

Without ``--caddy`` the script prints, for each password, the SQL statement
to paste into pgAdmin (ELA user accounts). With ``--caddy`` it prints the
line to put in ``.env`` for ``ADMIN_BASIC_AUTH_HASH`` (backoffice protection
by the reverse proxy). Caddy accepts the ``$2b$`` bcrypt variant produced
by the Python ``bcrypt`` package, so Docker is not required.

The password is never echoed when ``--caddy`` is used; prefer typing it
interactively (``ela-hash --caddy``) so it does not end up
in your shell history.
"""

from __future__ import annotations

import getpass
import sys

try:
    import bcrypt
except ImportError:  # pragma: no cover
    print("Install bcrypt first: pip install bcrypt")
    sys.exit(1)


def generate_hash(password: str, rounds: int = 12) -> str:
    """Generate a bcrypt hash from a plaintext password.

    Args:
        password: The plaintext password to hash.
        rounds: bcrypt cost factor (12 is a sensible default).

    Returns:
        The bcrypt hash string (starts with ``$2b$12$``).
    """
    salt = bcrypt.gensalt(rounds=rounds)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def _print_caddy(password: str) -> None:
    """Print the ``.env`` line for Caddy basic auth (single quotes keep the ``$``)."""
    hashed = generate_hash(password)
    print("\nAjoutez cette ligne dans .env (garder les guillemets simples) :")
    print(f"ADMIN_BASIC_AUTH_HASH='{hashed}'")


def _print_sql(password: str) -> None:
    """Print the SQL statement updating an ELA user's password hash."""
    hashed = generate_hash(password)
    print(f"\nPassword : {password}")
    print(f"Hash     : {hashed}")
    print("\nSQL (copie dans PgAdmin4) :")
    print(f"UPDATE users SET password_hash = '{hashed}' WHERE identifier = '<username>';")


def main() -> None:
    """Console entry point (``ela-hash``)."""
    args = sys.argv[1:]

    if args and args[0] == "--caddy":
        password = args[1] if len(args) > 1 else getpass.getpass("Mot de passe backoffice : ")
        if not password:
            print("Mot de passe vide, abandon.")
            sys.exit(1)
        _print_caddy(password)
        sys.exit(0)

    if not args:
        print("Usage: ela-hash <password> [<password> ...]")
        print("       ela-hash --caddy [<password>]")
        sys.exit(1)

    for pwd in args:
        _print_sql(pwd)


if __name__ == "__main__":
    main()
