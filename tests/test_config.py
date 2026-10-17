"""Tests for ``ela.config`` helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from ela.config import Settings, find_env_file


def test_database_url_drivers():
    s = Settings(_env_file=None, database_url="postgresql://u:p@h:5432/db")
    assert s.async_database_url == "postgresql+asyncpg://u:p@h:5432/db"
    assert s.sync_database_url == "postgresql+psycopg2://u:p@h:5432/db"
    assert s.libpq_database_url == "postgresql://u:p@h:5432/db"


def test_database_url_accepts_any_driver_prefix():
    for url in ("postgresql+asyncpg://u:p@h/db", "postgresql+psycopg2://u:p@h/db"):
        s = Settings(_env_file=None, database_url=url)
        assert s.async_database_url == "postgresql+asyncpg://u:p@h/db"
        assert s.libpq_database_url == "postgresql://u:p@h/db"


def test_admin_database_url_override():
    s = Settings(
        _env_file=None,
        database_url="postgresql://u:p@localhost/db",
        admin_database_url="postgresql+asyncpg://u:p@db/db",
    )
    assert s.libpq_database_url == "postgresql://u:p@db/db"


def test_missing_database_url_is_explicit():
    s = Settings(_env_file=None)
    with pytest.raises(ValueError, match="DATABASE_URL"):
        _ = s.async_database_url


def test_find_env_file_walks_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ELA_ENV_FILE", raising=False)
    (tmp_path / ".env").write_text("X=1\n")
    nested = tmp_path / "apps" / "admin"
    nested.mkdir(parents=True)
    assert find_env_file(nested) == tmp_path / ".env"
    assert find_env_file(nested, max_depth=1) is None


def test_find_env_file_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("ELA_ENV_FILE", str(tmp_path / "custom.env"))
    assert find_env_file(tmp_path) == tmp_path / "custom.env"


def test_retrieval_defaults_and_helpers():
    s = Settings(_env_file=None)
    assert s.retrieval_backend == "pgvector" and s.embedding_provider == "local"
    assert s.embedding_model == "BAAI/bge-m3" and s.embedding_dim == 1024
    assert s.rerank_provider == "none"
    assert Settings(_env_file=None, rerank_provider="flashrank").rerank_model_name == "ms-marco-MultiBERT-L-12"
    assert Settings(_env_file=None, rerank_provider="local").rerank_model_name == "BAAI/bge-reranker-v2-m3"
    assert Settings(_env_file=None, rerank_model="custom/x").rerank_model_name == "custom/x"
    with pytest.raises(ValueError):
        s.embedding_endpoint()


def test_embedding_endpoint_per_provider():
    s = Settings(_env_file=None, embedding_provider="ilaas", ilaas_api_key="k1")
    assert s.embedding_endpoint() == ("https://rag-api.ilaas.fr/v1", "k1")
    s = Settings(_env_file=None, embedding_provider="ilaas", ilaas_api_key="k1", embedding_api_key="k2")
    assert s.embedding_endpoint()[1] == "k2"
    s = Settings(_env_file=None, embedding_provider="openai", embedding_base_url="https://e/v1", embedding_api_key="k")
    assert s.embedding_endpoint() == ("https://e/v1", "k")
    with pytest.raises(ValueError):
        Settings(_env_file=None, embedding_provider="openai").embedding_endpoint()
