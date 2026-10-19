"""Shared pytest fixtures.

* The settings tests must not depend on the developer's real environment:
  VS Code injects the workspace ``.env`` into its integrated terminal, and
  process variables take precedence over any env file passed to
  ``Settings``. An autouse fixture removes every ELA-related variable.
* Process-wide caches (settings, retrieval resources, LLM clients) are
  reset between tests.
"""

from __future__ import annotations

import os

import pytest

_PREFIXES = (
    "LLM_",
    "CEREBRAS_",
    "ILAAS_",
    "GROQ_",
    "DATABASE_",
    "ADMIN_DATABASE",
    "ELA_",
    "EMBEDDING_",
    "CHROMA_",
    "LATEX_",
    "HISTORY_",
    "RETRIEVAL_",
    "RERANK_",
    "WARMUP_",
    "CHUNK_",
    "INGEST_",
    "LOG_",
    "AGENT_",
    "PROMPT_",
    "LANGFUSE_",
    "EVAL_",
    "SANDBOX_",
    "CODE_",
)


@pytest.fixture(autouse=True)
def _clean_ela_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove ELA configuration variables and reset process caches."""
    for key in list(os.environ):
        if key.upper().startswith(_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    # Ignore any .env lying around (tests are hermetic)
    monkeypatch.setenv("ELA_ENV_FILE", os.devnull)

    from ela import engine as engine_mod
    from ela.config import get_settings
    from ela.retrieval import resources

    get_settings.cache_clear()
    resources.reset()
    engine_mod.reset_llms()
    yield
    get_settings.cache_clear()
    resources.reset()
    engine_mod.reset_llms()
