"""Tests of the chunk stores (RRF, pgvector integration, Chroma)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from ela.ingestion.latex import LatexIngestor
from ela.retrieval.store import ChromaStore, PgVectorStore, rrf_fuse
from hashemb import HashEmb
from latex_fixture import make_course_tree

TEST_URL = os.getenv("TEST_DATABASE_URL")
DIM = 32


def test_rrf_fuse_prefers_ids_present_in_both_legs():
    scores = rrf_fuse([["a", "b", "c"], ["c", "d"]], [0.5, 0.5], k=60)
    assert scores["c"] > scores["a"] > scores["b"]
    assert set(scores) == {"a", "b", "c", "d"}


@pytest.fixture(scope="module")
def chunks(tmp_path_factory):
    latex = make_course_tree(tmp_path_factory.mktemp("corpus"))
    return LatexIngestor(str(latex), log=lambda m: None).scan().chunks


def _exercise_store(store, chunks):
    emb = HashEmb(DIM)
    vectors = emb.embed_documents([c.page_content() for c in chunks])
    store.upsert(chunks, vectors)
    assert store.count() == len(chunks)
    assert store.existing() == {c.chunk_id: c.content_hash for c in chunks}
    fp1 = store.fingerprint()

    q = "processus stochastique série temporelle"
    hits = store.search(q, emb.embed_query(q), k=5, levels=["M1", "Commun"], doc_kinds=["slides"])
    assert hits and all(h.doc.metadata["level"] in {"M1", "Commun"} for h in hits)
    assert all(h.doc.metadata["doc_kind"] == "slides" for h in hits)
    assert hits[0].doc.metadata["slide_title"] == "Les séries temporelles"
    assert hits[0].lexical_rank is not None and hits[0].vector_rank is not None
    assert hits[0].score > hits[-1].score

    # M2 only sees the manuscript + Commun
    hits_m2 = store.search(q, emb.embed_query(q), k=5, levels=["M2", "Commun"])
    assert {h.doc.metadata["source"] for h in hits_m2} <= {"hdr_main.tex", "algebra.tex"}

    got = store.get([chunks[0].chunk_id, "missing"])
    assert len(got) == 1 and got[0].metadata["chunk_id"] == chunks[0].chunk_id
    assert len(store.all_documents(levels=["Commun"])) == 1

    store.delete([chunks[0].chunk_id])
    assert store.count() == len(chunks) - 1 and store.fingerprint() != fp1
    store.clear()
    assert store.count() == 0


@pytest.mark.skipif(not TEST_URL, reason="TEST_DATABASE_URL not set")
def test_pgvector_store_roundtrip(chunks):
    engine = create_engine(TEST_URL.replace("postgresql://", "postgresql+psycopg2://"))
    with engine.connect() as conn:
        has_ext = conn.execute(text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")).scalar()
    if not has_ext:
        pytest.skip("pgvector not available on the test server")
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS rag_chunks"))
    store = PgVectorStore(engine, dim=DIM)
    store.ensure_schema()
    _exercise_store(store, chunks)

    # Dimension change: table is cleared and the column retyped
    store.upsert(chunks[:2], HashEmb(DIM).embed_documents([c.page_content() for c in chunks[:2]]))
    other = PgVectorStore(engine, dim=DIM * 2)
    other.ensure_schema()
    assert other.count() == 0
    with engine.connect() as conn:
        mod = conn.execute(
            text("SELECT atttypmod FROM pg_attribute WHERE attrelid = 'rag_chunks'::regclass AND attname='embedding'")
        ).scalar()
    assert mod == DIM * 2
    engine.dispose()


def test_chroma_store_roundtrip(chunks, tmp_path: Path):
    pytest.importorskip("chromadb")
    store = ChromaStore(str(tmp_path / "chroma"))
    _exercise_store(store, chunks)
