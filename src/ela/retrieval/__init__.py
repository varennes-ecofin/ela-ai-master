"""Retrieval layer: shared resources, level-scoped hybrid retriever.

Phase 1 keeps the Phase-0 algorithm (Chroma + BM25 ensemble + FlashRank)
but loads every heavy component **once per process** instead of once per
chat session. Phase 2 replaces the store, embeddings and reranker behind
the same ``get_retriever(level)`` entry point.
"""

from ela.retrieval.retriever import ScopedRetriever, get_retriever

__all__ = ["ScopedRetriever", "get_retriever"]
