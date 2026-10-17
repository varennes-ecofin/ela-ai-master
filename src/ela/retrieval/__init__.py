"""Retrieval layer: embeddings, chunk store, reranker, level-scoped retriever.

Phase 2: ``bge-m3`` embeddings, hybrid search (vector + French full text,
RRF) in PostgreSQL/pgvector (Chroma + BM25 as fallback), multilingual
reranking, typed chunks (slides, definitions, manuscripts). Every heavy
component is loaded **once per process** (:mod:`ela.retrieval.resources`)
and chat sessions share it through ``get_retriever(level)``.
"""

from ela.retrieval.retriever import ScopedRetriever, get_retriever

__all__ = ["ScopedRetriever", "get_retriever"]
