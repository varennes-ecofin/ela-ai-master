"""Generate bcrypt password hashes (``ela-hash``) for ELA accounts.

Usage:
    ela-hash <password> [<password> ...]

For each password the script prints the hash and the SQL statement updating
an account (``ela-users set-password`` is the preferred way; this remains for
manual repairs through pgAdmin). The Caddy ``--caddy`` option disappeared with
the Streamlit backoffice (lot 5.4b): the FastAPI backoffice authenticates its
own users, no basic auth is needed in the reverse proxy.
"""

from __future__ import annotations

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
    if not args or args[0].startswith("-"):
        print("Usage: ela-hash <password> [<password> ...]")
        sys.exit(1)
    for pwd in args:
        _print_sql(pwd)


if __name__ == "__main__":
    main()
