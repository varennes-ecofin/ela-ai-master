"""Shared pytest fixtures.

The settings tests must not depend on the developer's real environment:
VS Code injects the workspace ``.env`` into its integrated terminal, and
process variables take precedence over any env file passed to
``Settings``. This autouse fixture removes every ELA-related variable
before each test.
"""

from __future__ import annotations

import os

import pytest

_PREFIXES = ("LLM_", "CEREBRAS_", "ILAAS_", "GROQ_", "DATABASE_", "ELA_", "EMBEDDING_", "CHROMA_", "LATEX_")


@pytest.fixture(autouse=True)
def _clean_ela_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove ELA configuration variables from the process environment."""
    for key in list(os.environ):
        if key.upper().startswith(_PREFIXES):
            monkeypatch.delenv(key, raising=False)
