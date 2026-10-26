"""Password policy, hashing and generation (local accounts).

Policy (decided 2026-10-22): at least 8 characters, at least one digit and
at least one special character. Hashes are bcrypt (cost 12), the format
already used by the ``users`` table and by Chainlit's login.
"""

from __future__ import annotations

import secrets
import string

import bcrypt

MIN_LENGTH = 8
SPECIAL_CHARACTERS = string.punctuation  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
BCRYPT_ROUNDS = 12

# Generated passwords: letters without look-alikes, digits without 0/1, a few safe specials.
_GEN_LETTERS = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ"
_GEN_DIGITS = "23456789"
_GEN_SPECIALS = "!#%+-=?@"


def problems(password: str) -> list[str]:
    """Return the policy violations of ``password`` (empty list when it is acceptable)."""
    issues: list[str] = []
    if len(password) < MIN_LENGTH:
        issues.append(f"au moins {MIN_LENGTH} caractères")
    if not any(c.isdigit() for c in password):
        issues.append("au moins un chiffre")
    if not any(c in SPECIAL_CHARACTERS for c in password):
        issues.append("au moins un caractère spécial (par ex. ! ? # % + - = @)")
    return issues


def is_acceptable(password: str) -> bool:
    """Whether ``password`` satisfies the policy."""
    return not problems(password)


def describe_policy() -> str:
    """One-line description of the policy, for forms and error messages."""
    return f"{MIN_LENGTH} caractères minimum, dont un chiffre et un caractère spécial"


def hash_password(password: str) -> str:
    """Return the bcrypt hash of ``password`` (does not check the policy)."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("utf-8")


def check_password(password: str, password_hash: str) -> bool:
    """Verify ``password`` against a bcrypt hash (``False`` on a malformed hash)."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def generate(length: int = 12) -> str:
    """Generate a random password that satisfies the policy and is easy to type.

    Args:
        length: Total length (at least ``MIN_LENGTH``).

    Returns:
        A password with letters, at least two digits and one special character, shuffled.
    """
    length = max(length, MIN_LENGTH)
    rng = secrets.SystemRandom()
    chars = [rng.choice(_GEN_DIGITS), rng.choice(_GEN_DIGITS), rng.choice(_GEN_SPECIALS)]
    chars += [rng.choice(_GEN_LETTERS) for _ in range(length - len(chars))]
    rng.shuffle(chars)
    return "".join(chars)
