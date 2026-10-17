"""Tests of the embedding factory helpers (no model download)."""

from __future__ import annotations

import pytest

from ela.config import Settings
from ela.retrieval.embeddings import FallbackEmbeddings, build_embeddings


class Flaky:
    def __init__(self, fail: bool):
        self.fail = fail
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        if self.fail:
            raise ConnectionError("api down")
        return [[1.0] for _ in texts]

    def embed_query(self, text):
        self.calls += 1
        if self.fail:
            raise ConnectionError("api down")
        return [1.0]


def test_fallback_embeddings_switch_once_to_local():
    primary, local = Flaky(fail=True), Flaky(fail=False)
    built = {"n": 0}

    def factory():
        built["n"] += 1
        return local

    emb = FallbackEmbeddings(primary, factory)
    assert emb.embed_query("x") == [1.0] and emb.using_fallback
    assert emb.embed_documents(["a", "b"]) == [[1.0], [1.0]]
    assert primary.calls == 1 and local.calls == 2 and built["n"] == 1  # primary not retried


def test_fallback_embeddings_use_primary_when_healthy():
    primary = Flaky(fail=False)
    emb = FallbackEmbeddings(primary, lambda: pytest.fail("fallback must not be built"))
    assert emb.embed_query("x") == [1.0] and not emb.using_fallback


def test_build_embeddings_requires_api_key_for_api_providers():
    with pytest.raises(RuntimeError, match="API key"):
        build_embeddings(Settings(_env_file=None, embedding_provider="ilaas"))


def test_build_reranker_uses_provider_default_model(monkeypatch):
    """RERANK_MODEL empty → the provider default is passed to the library (regression)."""
    from ela.retrieval import rerank

    seen = {}

    class FakeFlash:
        def __init__(self, model, cache_dir):
            seen["model"], seen["cache"] = model, cache_dir
            self.name = model

    monkeypatch.setattr(rerank, "FlashRankReranker", FakeFlash)
    r = rerank.build_reranker(Settings(_env_file=None, rerank_provider="flashrank", rerank_cache_dir="x"))
    assert seen == {"model": "ms-marco-MultiBERT-L-12", "cache": "x"} and r.name == "ms-marco-MultiBERT-L-12"
    assert rerank.build_reranker(Settings(_env_file=None, rerank_provider="none")) is None
