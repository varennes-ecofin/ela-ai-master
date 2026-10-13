"""Tests for the retrieval layer with fake components (no models, no Chroma)."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ela.config import get_settings
from ela.retrieval import resources
from ela.retrieval.retriever import (
    ScopedRetriever,
    build_chroma_filter,
    filter_docs_by_level,
    get_retriever,
)


class _FakeVectorRetriever(BaseRetriever):
    docs: list[Document]
    filter: dict | None = None

    def _get_relevant_documents(self, query: str, *, run_manager=None):  # noqa: D401
        docs = self.docs
        if self.filter:
            allowed = set(self.filter["level"]["$in"])
            docs = [d for d in docs if d.metadata.get("level") in allowed]
        return [d for d in docs if query.split()[0] in d.page_content][:5] or docs[:5]


class FakeStore:
    """Mimics the two methods of a LangChain vector store used by ELA."""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.filters: list[dict | None] = []

    def as_retriever(self, search_kwargs=None):
        f = (search_kwargs or {}).get("filter")
        self.filters.append(f)
        return _FakeVectorRetriever(docs=self.docs, filter=f)

    def get(self):
        return {"documents": [d.page_content for d in self.docs], "metadatas": [d.metadata for d in self.docs]}


CORPUS = [
    Document(page_content="stationnarité racine unitaire test ADF", metadata={"level": "M1", "source": "UTS_C2.tex"}),
    Document(page_content="GARCH volatilité rendements", metadata={"level": "M2", "source": "HFE_C1.tex"}),
    Document(page_content="algèbre matricielle rappels", metadata={"level": "Commun", "source": "bioGDT.tex"}),
    Document(page_content="cointégration VECM Johansen", metadata={"level": "M1", "source": "MTS_C2.tex"}),
]


def test_level_filters():
    assert build_chroma_filter("ALL") is None
    assert build_chroma_filter("M1") == {"level": {"$in": ["M1", "Commun", "legacy"]}}
    assert [d.metadata["source"] for d in filter_docs_by_level(CORPUS, "M2")] == ["HFE_C1.tex", "bioGDT.tex"]
    assert filter_docs_by_level(CORPUS, "admin") == CORPUS


def test_scoped_retriever_is_scoped(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RERANK_TOP_N", "0")
    store = FakeStore(CORPUS)
    r = ScopedRetriever("M2", store=store, corpus=CORPUS, reranker=None)
    assert r.doc_count == 2
    sources = {d.metadata["source"] for d in r.invoke("volatilité GARCH")}
    assert "HFE_C1.tex" in sources
    assert "UTS_C2.tex" not in sources and "MTS_C2.tex" not in sources


def test_scoped_retriever_unfiltered_for_admin():
    store = FakeStore(CORPUS)
    r = ScopedRetriever("admin", store=store, corpus=CORPUS, reranker=None)
    assert r.doc_count == 4 and store.filters == [None]


def test_reranker_truncates_to_top_n(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RERANK_TOP_N", "2")

    class FakeRanker:
        def rerank(self, request):
            # reverse order, highest score first
            return [{"id": p["id"], "score": 1.0} for p in reversed(request.passages)]

    # flashrank.RerankRequest is imported lazily: provide a stand-in module
    import sys
    import types

    mod = types.ModuleType("flashrank")

    class RerankRequest:
        def __init__(self, query, passages):
            self.query, self.passages = query, passages

    mod.RerankRequest = RerankRequest
    monkeypatch.setitem(sys.modules, "flashrank", mod)

    r = ScopedRetriever("ALL", store=FakeStore(CORPUS), corpus=CORPUS, reranker=FakeRanker())
    out = r.invoke("rappels")
    assert len(out) == 2


def test_resources_are_shared_and_follow_store_fingerprint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    sqlite = chroma_dir / "chroma.sqlite3"
    sqlite.write_bytes(b"v1")
    monkeypatch.setenv("CHROMA_DB_PATH", str(chroma_dir))
    monkeypatch.setenv("RERANK_TOP_N", "0")
    get_settings.cache_clear()

    builds = {"emb": 0, "store": 0}

    def fake_embeddings():
        builds["emb"] += 1
        return object()

    def fake_store(_emb):
        builds["store"] += 1
        return FakeStore(CORPUS)

    monkeypatch.setattr(resources, "embeddings_builder", fake_embeddings)
    monkeypatch.setattr(resources, "store_builder", fake_store)
    resources.reset()

    r1 = get_retriever("M1")
    r2 = get_retriever("M1")
    r3 = get_retriever("M2")
    assert r1 is r2 and r1 is not r3
    assert builds == {"emb": 1, "store": 1}
    assert resources.status().corpus_size == 4

    # Simulate a rebuild of the knowledge base (new file size/mtime)
    sqlite.write_bytes(b"v2-bigger")
    r4 = get_retriever("M1")
    assert r4 is not r1
    assert builds["store"] == 2 and builds["emb"] == 1  # embeddings reused, store reloaded
