"""Password policy, hashing and generation (ela.auth.passwords)."""

from __future__ import annotations

from ela.auth import passwords


def test_policy_problems():
    assert passwords.problems("Abcdef1!") == []
    assert passwords.is_acceptable("longer-password-9")
    issues = passwords.problems("abc")
    assert len(issues) == 3 and "8 caractères" in issues[0]
    assert passwords.problems("abcdefgh!") == ["au moins un chiffre"]
    assert passwords.problems("abcdefg1")[0].startswith("au moins un caractère spécial")
    assert "8 caractères" in passwords.describe_policy()


def test_hash_and_check():
    h = passwords.hash_password("Secret-123")
    assert h.startswith("$2b$12$")
    assert passwords.check_password("Secret-123", h)
    assert not passwords.check_password("secret-123", h)
    assert not passwords.check_password("x", "not-a-hash")


def test_generate_satisfies_policy_and_varies():
    seen = {passwords.generate() for _ in range(20)}
    assert len(seen) == 20
    for pwd in seen:
        assert len(pwd) == 12 and passwords.is_acceptable(pwd)
        assert not any(c in "0O1lI" for c in pwd)
    assert len(passwords.generate(4)) == passwords.MIN_LENGTH
