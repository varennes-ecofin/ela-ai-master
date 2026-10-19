"""Chunk stores: pgvector (hybrid vector + French full-text) and Chroma.

Both backends implement :class:`ChunkStore`:

* :class:`PgVectorStore` keeps chunks in the ``rag_chunks`` PostgreSQL table
  (``vector`` column with an HNSW index, ``tsvector`` column with a GIN index)
  and runs the hybrid search in **one SQL query** fused by Reciprocal Rank
  Fusion. It is the production backend (same database as the accounts).
* :class:`ChromaStore` wraps the Phase-1 Chroma directory and adds an
  in-memory BM25 index for the lexical leg; it needs no PostgreSQL extension
  and remains available for machines without pgvector.

Vectors are computed by the caller (ingestion pipeline / retriever), so a
store never depends on the embedding model.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from langchain_core.documents import Document
from sqlalchemy import text
from sqlalchemy.engine import Engine

if TYPE_CHECKING:
    from ela.ingestion.latex import Chunk

logger = logging.getLogger(__name__)

RAG_TABLE = "rag_chunks"

#: Columns persisted for every chunk (same names as ``Chunk.metadata()``).
_COLUMNS = (
    "chunk_id",
    "parent_id",
    "content_hash",
    "source",
    "level",
    "course",
    "doc_kind",
    "chunk_type",
    "chapter",
    "section",
    "subsection",
    "slide_title",
    "element_title",
    "ordinal",
    "part",
    "n_chars",
)


@dataclass
class Hit:
    """One search result.

    Attributes:
        doc: The chunk as a LangChain document (metadata = chunk metadata).
        score: Fused score (RRF), higher is better.
        vector_rank: Rank in the vector leg (``None`` when absent).
        lexical_rank: Rank in the lexical leg (``None`` when absent).
    """

    doc: Document
    score: float
    vector_rank: int | None = None
    lexical_rank: int | None = None


class ChunkStore(Protocol):
    """Storage and hybrid search of chunks."""

    backend: str

    def fingerprint(self) -> str:
        """Cheap token that changes whenever the content changes."""
        ...

    def count(self) -> int: ...

    def existing(self) -> dict[str, str]:
        """``chunk_id → content_hash`` of every stored chunk."""
        ...

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def delete(self, ids: list[str]) -> None: ...

    def clear(self) -> None: ...

    def get(self, ids: list[str]) -> list[Document]: ...

    def all_documents(self, levels: list[str] | None = None) -> list[Document]: ...

    def search(
        self,
        query: str,
        query_vector: list[float],
        *,
        k: int,
        levels: list[str] | None = None,
        doc_kinds: list[str] | None = None,
        courses: list[str] | None = None,
    ) -> list[Hit]:
        """Hybrid search (vector + lexical, RRF-fused) restricted to ``levels``.

        Each leg returns its ``k`` best; the union (up to ``2k`` candidates) is
        fused so that a document found by only one leg still reaches the reranker.
        ``courses`` restricts the search to the given course folders.
        """
        ...


def rrf_fuse(rankings: list[list[str]], weights: list[float] | None = None, k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion of several ranked id lists.

    Args:
        rankings: One ordered list of ids per leg.
        weights: Optional weight per leg (default 1.0).
        k: RRF constant (60 in the original paper).

    Returns:
        ``id → fused score``.
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
    return scores


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in vector) + "]"


# ---------------------------------------------------------------------------
# pgvector
# ---------------------------------------------------------------------------


class PgVectorStore:
    """Chunks in PostgreSQL with pgvector + French full-text search.

    Args:
        engine: Sync SQLAlchemy engine (psycopg2).
        dim: Expected embedding dimension (checked/applied by :meth:`ensure_schema`).
        lexical_weight: Weight of the full-text leg in the fusion (vector = 1 - w).
        rrf_k: RFF constant.
        fts_config: PostgreSQL text-search configuration.
    """

    backend = "pgvector"

    def __init__(
        self,
        engine: Engine,
        *,
        dim: int = 1024,
        lexical_weight: float = 0.5,
        rrf_k: int = 60,
        fts_config: str = "french",
    ) -> None:
        self._engine = engine
        self.dim = dim
        self.lexical_weight = lexical_weight
        self.rrf_k = rrf_k
        self.fts_config = fts_config

    # ---- schema ----

    def ensure_schema(self) -> None:
        """Create the extension/table if needed and align the vector dimension.

        Raises:
            RuntimeError: When the ``vector`` extension is not installed on the
                server or cannot be created with the current role.
        """
        with self._engine.begin() as conn:
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            except Exception as exc:  # permission / not installed
                raise RuntimeError(
                    "L'extension PostgreSQL 'vector' (pgvector) est indisponible : "
                    f"{exc.__class__.__name__}. Installez pgvector puis exécutez en superuser "
                    "`CREATE EXTENSION vector;` dans la base, ou utilisez RETRIEVAL_BACKEND=chroma."
                ) from exc
            conn.execute(text(_CREATE_TABLE_SQL.format(table=RAG_TABLE, dim=self.dim, fts=self.fts_config)))
            current = conn.execute(
                text(
                    "SELECT atttypmod FROM pg_attribute WHERE attrelid = CAST(:t AS regclass) AND attname = 'embedding'"
                ),
                {"t": RAG_TABLE},
            ).scalar()
            if current is not None and current != self.dim:
                logger.warning("Embedding dimension changed (%s → %s): clearing %s.", current, self.dim, RAG_TABLE)
                conn.execute(text(f"TRUNCATE {RAG_TABLE}"))
                conn.execute(text(f"DROP INDEX IF EXISTS {RAG_TABLE}_embedding_idx"))
                conn.execute(text(f"ALTER TABLE {RAG_TABLE} ALTER COLUMN embedding TYPE vector({self.dim})"))
            conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS {RAG_TABLE}_embedding_idx ON {RAG_TABLE} "
                    "USING hnsw (embedding vector_cosine_ops)"
                )
            )

    # ---- bookkeeping ----

    def fingerprint(self) -> str:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT count(*), coalesce(max(extract(epoch from updated_at)), 0) FROM {RAG_TABLE}")
            ).one()
        return f"pg:{row[0]}:{int(row[1])}"

    def count(self) -> int:
        with self._engine.connect() as conn:
            return int(conn.execute(text(f"SELECT count(*) FROM {RAG_TABLE}")).scalar() or 0)

    def existing(self) -> dict[str, str]:
        with self._engine.connect() as conn:
            rows = conn.execute(text(f"SELECT chunk_id, content_hash FROM {RAG_TABLE}")).all()
        return {r[0]: r[1] for r in rows}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        rows = []
        for chunk, vec in zip(chunks, vectors, strict=True):
            meta = chunk.metadata()
            row = {c: meta[c] for c in _COLUMNS}
            row["content"] = chunk.page_content()
            row["embedding"] = _vector_literal(vec)
            rows.append(row)
        cols = ", ".join(_COLUMNS) + ", content, embedding"
        vals = ", ".join(f":{c}" for c in _COLUMNS) + ", :content, CAST(:embedding AS vector)"
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in (*_COLUMNS[1:], "content", "embedding"))
        sql = text(
            f"INSERT INTO {RAG_TABLE} ({cols}) VALUES ({vals}) "
            f"ON CONFLICT (chunk_id) DO UPDATE SET {updates}, updated_at = now()"
        )
        with self._engine.begin() as conn:
            for start in range(0, len(rows), 200):
                conn.execute(sql, rows[start : start + 200])

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        with self._engine.begin() as conn:
            conn.execute(text(f"DELETE FROM {RAG_TABLE} WHERE chunk_id = ANY(:ids)"), {"ids": list(ids)})

    def clear(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(text(f"DELETE FROM {RAG_TABLE}"))

    # ---- reads ----

    @staticmethod
    def _to_document(row: Any) -> Document:
        meta = {c: row._mapping[c] for c in _COLUMNS}
        meta["type"] = (
            "slide" if meta["chunk_type"] == "slide" else ("text_block" if meta["chunk_type"] == "text" else "element")
        )
        return Document(page_content=row._mapping["content"], metadata=meta)

    def get(self, ids: list[str]) -> list[Document]:
        if not ids:
            return []
        cols = ", ".join(_COLUMNS)
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT {cols}, content FROM {RAG_TABLE} WHERE chunk_id = ANY(:ids)"), {"ids": list(ids)}
            ).all()
        by_id = {r._mapping["chunk_id"]: self._to_document(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def all_documents(self, levels: list[str] | None = None) -> list[Document]:
        cols = ", ".join(_COLUMNS)
        where = "WHERE level = ANY(:levels)" if levels else ""
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT {cols}, content FROM {RAG_TABLE} {where} ORDER BY source, ordinal, part"),
                {"levels": levels} if levels else {},
            ).all()
        return [self._to_document(r) for r in rows]

    def search(
        self,
        query: str,
        query_vector: list[float],
        *,
        k: int,
        levels: list[str] | None = None,
        doc_kinds: list[str] | None = None,
        courses: list[str] | None = None,
    ) -> list[Hit]:
        filters = []
        params: dict[str, Any] = {
            "qvec": _vector_literal(query_vector),
            "qtext": query,
            "k": k,
            "k2": 2 * k,
            "rrf": self.rrf_k,
            "w_lex": self.lexical_weight,
            "w_vec": 1.0 - self.lexical_weight,
        }
        if levels:
            filters.append("level = ANY(:levels)")
            params["levels"] = list(levels)
        if doc_kinds:
            filters.append("doc_kind = ANY(:kinds)")
            params["kinds"] = list(doc_kinds)
        if courses:
            filters.append("course = ANY(:courses)")
            params["courses"] = list(courses)
        where = ("WHERE " + " AND ".join(filters)) if filters else ""
        cols = ", ".join(f"c.{c}" for c in _COLUMNS)
        sql = text(
            f"""
            WITH vec AS (
                SELECT chunk_id, row_number() OVER (ORDER BY embedding <=> CAST(:qvec AS vector)) AS r
                FROM {RAG_TABLE} {where}
                ORDER BY embedding <=> CAST(:qvec AS vector) LIMIT :k
            ),
            lex AS (
                SELECT chunk_id, row_number() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS r
                FROM {RAG_TABLE},
                     to_tsquery('{self.fts_config}',
                                replace(CAST(plainto_tsquery('{self.fts_config}', :qtext) AS text), '&', '|')) q
                {where + " AND" if where else "WHERE"} tsv @@ q
                ORDER BY ts_rank_cd(tsv, q) DESC LIMIT :k
            )
            SELECT {cols}, c.content, vec.r AS vr, lex.r AS lr,
                   coalesce(:w_vec / (:rrf + vec.r), 0) + coalesce(:w_lex / (:rrf + lex.r), 0) AS score
            FROM {RAG_TABLE} c
            LEFT JOIN vec ON vec.chunk_id = c.chunk_id
            LEFT JOIN lex ON lex.chunk_id = c.chunk_id
            WHERE vec.r IS NOT NULL OR lex.r IS NOT NULL
            ORDER BY score DESC, vr NULLS LAST LIMIT :k2
            """
        )
        with self._engine.connect() as conn:
            rows = conn.execute(sql, params).all()
        return [Hit(self._to_document(r), float(r._mapping["score"]), r._mapping["vr"], r._mapping["lr"]) for r in rows]


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    chunk_id      TEXT PRIMARY KEY,
    parent_id     TEXT NOT NULL DEFAULT '',
    content_hash  TEXT NOT NULL,
    source        TEXT NOT NULL,
    level         TEXT NOT NULL,
    course        TEXT NOT NULL,
    doc_kind      TEXT NOT NULL,
    chunk_type    TEXT NOT NULL,
    chapter       TEXT NOT NULL DEFAULT '',
    section       TEXT NOT NULL DEFAULT '',
    subsection    TEXT NOT NULL DEFAULT '',
    slide_title   TEXT NOT NULL DEFAULT '',
    element_title TEXT NOT NULL DEFAULT '',
    ordinal       INTEGER NOT NULL DEFAULT 0,
    part          INTEGER NOT NULL DEFAULT 0,
    n_chars       INTEGER NOT NULL DEFAULT 0,
    content       TEXT NOT NULL,
    embedding     vector({dim}),
    tsv           tsvector GENERATED ALWAYS AS (to_tsvector('{fts}', content)) STORED,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS {table}_level_idx ON {table} (level, doc_kind);
CREATE INDEX IF NOT EXISTS {table}_tsv_idx ON {table} USING gin (tsv);
"""


# ---------------------------------------------------------------------------
# Chroma (+ in-memory BM25)
# ---------------------------------------------------------------------------


class ChromaStore:
    """Chroma directory + in-memory BM25 lexical index.

    Args:
        path: Chroma persistence directory.
        lexical_weight: Weight of BM25 in the fusion.
        rrf_k: RRF constant.
        collection_name: Chroma collection.
    """

    backend = "chroma"

    def __init__(
        self,
        path: str,
        *,
        lexical_weight: float = 0.5,
        rrf_k: int = 60,
        collection_name: str = "ela",
    ) -> None:
        import chromadb

        self.path = path
        self.lexical_weight = lexical_weight
        self.rrf_k = rrf_k
        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})
        self._lock = threading.Lock()
        self._bm25_cache: dict[tuple, Any] = {}

    # ---- bookkeeping ----

    def fingerprint(self) -> str:
        sqlite = Path(self.path) / "chroma.sqlite3"
        try:
            st = sqlite.stat()
        except FileNotFoundError:
            return "chroma:missing"
        return f"chroma:{st.st_size}:{int(st.st_mtime)}:{self._collection.count()}"

    def count(self) -> int:
        return int(self._collection.count())

    def existing(self) -> dict[str, str]:
        data = self._collection.get(include=["metadatas"])
        return {i: (m or {}).get("content_hash", "") for i, m in zip(data["ids"], data["metadatas"], strict=False)}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        for start in range(0, len(chunks), 500):
            batch = chunks[start : start + 500]
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=vectors[start : start + 500],
                documents=[c.page_content() for c in batch],
                metadatas=[c.metadata() for c in batch],
            )
        self._bm25_cache.clear()

    def delete(self, ids: list[str]) -> None:
        if ids:
            self._collection.delete(ids=list(ids))
            self._bm25_cache.clear()

    def clear(self) -> None:
        data = self._collection.get(include=[])
        if data["ids"]:
            self._collection.delete(ids=data["ids"])
        self._bm25_cache.clear()

    # ---- reads ----

    @staticmethod
    def _where(levels: list[str] | None, doc_kinds: list[str] | None, courses: list[str] | None = None) -> dict | None:
        clauses = []
        if levels:
            clauses.append({"level": {"$in": list(levels)}})
        if doc_kinds:
            clauses.append({"doc_kind": {"$in": list(doc_kinds)}})
        if courses:
            clauses.append({"course": {"$in": list(courses)}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def get(self, ids: list[str]) -> list[Document]:
        if not ids:
            return []
        data = self._collection.get(ids=list(ids), include=["documents", "metadatas"])
        by_id = {
            i: Document(page_content=d, metadata=m or {})
            for i, d, m in zip(data["ids"], data["documents"], data["metadatas"], strict=False)
        }
        return [by_id[i] for i in ids if i in by_id]

    def all_documents(self, levels: list[str] | None = None) -> list[Document]:
        kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
        where = self._where(levels, None)
        if where:
            kwargs["where"] = where
        data = self._collection.get(**kwargs)
        return [
            Document(page_content=d, metadata=m or {})
            for d, m in zip(data["documents"], data["metadatas"], strict=False)
        ]

    def _bm25(self, levels: list[str] | None, doc_kinds: list[str] | None, courses: list[str] | None = None):
        key = (tuple(levels or ()), tuple(doc_kinds or ()), tuple(courses or ()))
        if key not in self._bm25_cache:
            with self._lock:
                if key not in self._bm25_cache:
                    from rank_bm25 import BM25Okapi

                    docs = self.all_documents(levels)
                    if doc_kinds:
                        docs = [d for d in docs if d.metadata.get("doc_kind") in set(doc_kinds)]
                    if courses:
                        docs = [d for d in docs if d.metadata.get("course") in set(courses)]
                    tokens = [_tokenize(d.page_content) for d in docs]
                    self._bm25_cache[key] = (BM25Okapi(tokens) if tokens else None, docs)
        return self._bm25_cache[key]

    def search(
        self,
        query: str,
        query_vector: list[float],
        *,
        k: int,
        levels: list[str] | None = None,
        doc_kinds: list[str] | None = None,
        courses: list[str] | None = None,
    ) -> list[Hit]:
        if self.count() == 0:
            return []
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": min(k, self.count()),
            "include": ["documents", "metadatas"],
        }
        where = self._where(levels, doc_kinds, courses)
        if where:
            kwargs["where"] = where
        res = self._collection.query(**kwargs)
        docs_by_id: dict[str, Document] = {}
        vec_ids: list[str] = []
        for i, d, m in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], strict=False):
            docs_by_id[i] = Document(page_content=d, metadata=m or {})
            vec_ids.append(i)

        lex_ids: list[str] = []
        bm25, corpus = self._bm25(levels, doc_kinds, courses)
        if bm25 is not None:
            scores = bm25.get_scores(_tokenize(query))
            order = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)[:k]
            for i in order:
                if scores[i] <= 0:
                    break
                cid = corpus[i].metadata.get("chunk_id") or f"idx:{i}"
                docs_by_id.setdefault(cid, corpus[i])
                lex_ids.append(cid)

        fused = rrf_fuse([vec_ids, lex_ids], [1.0 - self.lexical_weight, self.lexical_weight], self.rrf_k)
        vrank = {cid: r for r, cid in enumerate(vec_ids, start=1)}
        lrank = {cid: r for r, cid in enumerate(lex_ids, start=1)}
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[: 2 * k]
        return [Hit(docs_by_id[cid], s, vrank.get(cid), lrank.get(cid)) for cid, s in ranked]


def _tokenize(text_: str) -> list[str]:
    """Lower-case alphanumeric tokenizer for BM25 (accents kept)."""
    import re

    return re.findall(r"[a-zà-ÿ0-9]+", text_.lower())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_store(settings=None) -> ChunkStore:
    """Build the store selected by ``RETRIEVAL_BACKEND``.

    Args:
        settings: Settings override.

    Returns:
        A :class:`PgVectorStore` or a :class:`ChromaStore`.
    """
    from ela.config import get_settings

    s = settings or get_settings()
    if s.retrieval_backend == "chroma":
        if not os.path.isdir(s.chroma_db_path):
            logger.warning("Chroma directory '%s' not found — run `ela-ingest` first.", s.chroma_db_path)
        return ChromaStore(s.chroma_db_path, lexical_weight=s.retrieval_lexical_weight, rrf_k=s.retrieval_rrf_k)
    from ela.db.engine import get_sync_engine

    return PgVectorStore(
        get_sync_engine(),
        dim=s.embedding_dim,
        lexical_weight=s.retrieval_lexical_weight,
        rrf_k=s.retrieval_rrf_k,
        fts_config=s.retrieval_fts_config,
    )


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity (used by tests and diagnostics)."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0
