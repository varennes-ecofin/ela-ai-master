"""Tests for the retrieval layer with fake store/embeddings/reranker (no models)."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from ela.config import get_settings
from ela.retrieval import resources
from ela.retrieval.retriever import (
    ScopedRetriever,
    allowed_levels,
    build_chroma_filter,
    filter_docs_by_level,
    get_retriever,
)
from ela.retrieval.store import Hit


def _doc(
    cid: str, level: str, kind: str = "slides", ctype: str = "slide", parent: str = "", text: str = ""
) -> Document:
    return Document(
        page_content=text or f"contenu {cid}",
        metadata={
            "chunk_id": cid,
            "level": level,
            "doc_kind": kind,
            "chunk_type": ctype,
            "parent_id": parent,
            "source": f"{cid}.tex",
            "slide_title": f"Slide {cid}",
            "element_title": "Déf" if ctype != "slide" else "",
        },
    )


CORPUS = [
    _doc("uts1", "M1", text="stationnarité racine unitaire test ADF"),
    _doc("uts1-def", "M1", ctype="definition", parent="uts1", text="définition stationnarité"),
    _doc("hfe1", "M2", text="GARCH volatilité rendements"),
    _doc("com1", "Commun", text="algèbre matricielle rappels"),
    _doc("mts1", "M1", text="cointégration VECM Johansen"),
    _doc("hdr1", "M2", kind="manuscript", ctype="text", text="manuscrit HDR stationnarité GARCH"),
]


class FakeStore:
    """In-memory :class:`ChunkStore`: lexical overlap ranking, honours filters."""

    backend = "fake"

    def __init__(self, docs: list[Document], fp: str = "v1"):
        self.docs = docs
        self.fp = fp
        self.calls: list[dict] = []

    def fingerprint(self):
        return self.fp

    def count(self):
        return len(self.docs)

    def get(self, ids):
        return [d for d in self.docs if d.metadata["chunk_id"] in ids]

    def search(self, query, query_vector, *, k, levels=None, doc_kinds=None, courses=None):
        call = {"k": k, "levels": levels, "doc_kinds": doc_kinds}
        if courses:
            call["courses"] = courses
        self.calls.append(call)
        words = set(query.lower().split())
        out = []
        for d in self.docs:
            if levels and d.metadata["level"] not in levels:
                continue
            if doc_kinds and d.metadata["doc_kind"] not in doc_kinds:
                continue
            if courses and d.metadata.get("course") not in courses:
                continue
            overlap = len(words & set(d.page_content.lower().split()))
            if overlap:
                out.append(Hit(Document(page_content=d.page_content, metadata=dict(d.metadata)), float(overlap)))
        return sorted(out, key=lambda h: h.score, reverse=True)[:k]


class FakeEmb:
    def embed_query(self, text):
        return [1.0, 0.0]


class ReverseReranker:
    name = "reverse"

    def rerank(self, query, docs):
        return [(d, float(len(docs) - i)) for i, d in enumerate(reversed(docs))]


def test_level_helpers():
    assert allowed_levels("ALL") is None and allowed_levels("admin") is None
    assert allowed_levels("M1") == ["Commun", "M1", "legacy"]
    assert build_chroma_filter("M1") == {"level": {"$in": ["Commun", "M1", "legacy"]}}
    assert [d.metadata["chunk_id"] for d in filter_docs_by_level(CORPUS, "M2")] == ["hfe1", "com1", "hdr1"]


def test_scoped_retriever_is_scoped_and_queries_manuscripts_separately(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RERANK_TOP_N", "0")
    monkeypatch.setenv("RETRIEVAL_MANUSCRIPT_K", "2")
    get_settings.cache_clear()
    store = FakeStore(CORPUS)
    r = ScopedRetriever("M2", store=store, embeddings=FakeEmb(), reranker=None)
    ids = [d.metadata["chunk_id"] for d in r.invoke("volatilité GARCH stationnarité")]
    assert ids == ["hfe1", "hdr1"]  # slides first, then the manuscript quota; M1 docs never visible
    assert store.calls[0]["doc_kinds"] == ["slides"] and store.calls[1] == {
        "k": 2,
        "levels": ["Commun", "M2", "legacy"],
        "doc_kinds": ["manuscript"],
    }
    assert all("score" in d.metadata for d in r.invoke("GARCH"))


def test_scoped_retriever_unfiltered_for_admin(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RERANK_TOP_N", "0")
    get_settings.cache_clear()
    store = FakeStore(CORPUS)
    r = ScopedRetriever("admin", store=store, embeddings=FakeEmb(), reranker=None)
    assert r.levels is None
    ids = {d.metadata["chunk_id"] for d in r.invoke("stationnarité GARCH rappels")}
    assert ids == {"uts1", "hfe1", "com1", "hdr1"}  # uts1-def expanded into uts1 and deduplicated
    assert store.calls[0]["levels"] is None


def test_elements_are_expanded_to_their_parent_slide(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RERANK_TOP_N", "0")
    get_settings.cache_clear()
    r = ScopedRetriever("M1", store=FakeStore(CORPUS), embeddings=FakeEmb(), reranker=None)
    docs = r.invoke("définition stationnarité")
    assert [d.metadata["chunk_id"] for d in docs] == ["uts1"]
    assert docs[0].metadata["matched_element"] == "Déf" and docs[0].page_content.startswith("stationnarité")

    monkeypatch.setenv("RETRIEVAL_EXPAND_ELEMENTS", "false")
    get_settings.cache_clear()
    r = ScopedRetriever("M1", store=FakeStore(CORPUS), embeddings=FakeEmb(), reranker=None)
    assert {d.metadata["chunk_id"] for d in r.invoke("définition stationnarité")} == {"uts1", "uts1-def"}


def test_reranker_truncates_and_annotates(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RERANK_TOP_N", "2")
    monkeypatch.setenv("RETRIEVAL_EXPAND_ELEMENTS", "false")
    get_settings.cache_clear()
    r = ScopedRetriever("ALL", store=FakeStore(CORPUS), embeddings=FakeEmb(), reranker=ReverseReranker())
    out = r.invoke("stationnarité GARCH rappels")
    assert len(out) == 2 and all("rerank_score" in d.metadata for d in out)
    assert out[0].metadata["rerank_score"] > out[1].metadata["rerank_score"]


def test_resources_are_shared_and_follow_store_fingerprint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RERANK_TOP_N", "0")
    get_settings.cache_clear()
    builds = {"emb": 0, "store": 0}
    store = FakeStore(CORPUS, fp="gen-1")

    def fake_embeddings():
        builds["emb"] += 1
        return FakeEmb()

    def fake_store():
        builds["store"] += 1
        return store

    monkeypatch.setattr(resources, "embeddings_builder", fake_embeddings)
    monkeypatch.setattr(resources, "store_builder", fake_store)
    resources.reset()

    r1 = get_retriever("M1")
    r2 = get_retriever("M1")
    r3 = get_retriever("M2")
    assert r1 is r2 and r1 is not r3
    assert builds == {"emb": 1, "store": 1}
    st = resources.status()
    assert st.store_loaded and st.corpus_size == 6 and st.backend == "fake" and not st.reranker_loaded

    store.fp = "gen-2"  # knowledge base re-ingested
    r4 = get_retriever("M1")
    assert r4 is not r1 and builds == {"emb": 1, "store": 1}
    resources.reset()


def test_selection_caps_manuscripts_and_refills_after_dedupe(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RERANK_TOP_N", "3")
    monkeypatch.setenv("RETRIEVAL_MANUSCRIPT_MAX", "1")
    get_settings.cache_clear()
    corpus = CORPUS + [
        _doc("hdr2", "M2", kind="manuscript", ctype="text", text="manuscrit stationnarité GARCH rappels"),
        _doc("hdr3", "M2", kind="manuscript", ctype="text", text="manuscrit stationnarité GARCH rappels bis"),
    ]

    class ManuscriptsFirst:
        name = "mf"

        def rerank(self, query, docs):
            order = sorted(docs, key=lambda d: d.metadata["doc_kind"] != "manuscript")
            return [(d, float(len(order) - i)) for i, d in enumerate(order)]

    r = ScopedRetriever("ALL", store=FakeStore(corpus), embeddings=FakeEmb(), reranker=ManuscriptsFirst())
    out = r.invoke("stationnarité GARCH rappels")
    kinds = [d.metadata["doc_kind"] for d in out]
    assert len(out) == 3 and kinds.count("manuscript") == 1 and kinds[0] == "manuscript"
    # element + its parent in the ranked list count once; the slot is refilled with another slide
    ids = [d.metadata["chunk_id"] for d in out]
    assert len(set(ids)) == 3 and "uts1-def" not in ids


def test_invoke_course_chapter_and_top_n_filters(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RERANK_TOP_N", "5")
    get_settings.cache_clear()
    corpus = [
        _doc("a", "M1", text="stationnarité ADF"),
        _doc("b", "M1", text="stationnarité KPSS"),
        _doc("c", "M1", text="stationnarité PP"),
    ]
    corpus[0].metadata.update(course="UTS", chapter="Chapitre 2 : racines unitaires")
    corpus[1].metadata.update(course="UTS", chapter="Chapitre 3 : tests")
    corpus[2].metadata.update(course="MTS", chapter="Chapitre 1")
    store = FakeStore(corpus)
    r = ScopedRetriever("M1", store=store, embeddings=FakeEmb(), reranker=None)
    assert {d.metadata["chunk_id"] for d in r.invoke("stationnarité")} == {"a", "b", "c"}
    assert {d.metadata["chunk_id"] for d in r.invoke("stationnarité", courses=["UTS"])} == {"a", "b"}
    assert store.calls[-1]["courses"] == ["UTS"]
    assert [d.metadata["chunk_id"] for d in r.invoke("stationnarité", chapter="racines")] == ["a"]
    # unknown chapter: filter ignored rather than returning nothing
    assert len(r.invoke("stationnarité", chapter="inexistant")) == 3
    assert len(r.invoke("stationnarité", top_n=1)) == 1
