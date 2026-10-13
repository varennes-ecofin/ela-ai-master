"""Level-scoped hybrid retriever (vector + BM25, optional reranking).

One :class:`ScopedRetriever` is built per academic level and cached for
the current store generation, so a chat session only looks up an
existing object instead of re-indexing the whole corpus.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from langchain_core.documents import Document

from ela.config import get_settings
from ela.retrieval import resources

logger = logging.getLogger(__name__)

# Levels that grant access to ALL documents (no filtering)
UNFILTERED_LEVELS = frozenset({"ALL", "admin"})
# Levels always included regardless of user level
UNIVERSAL_LEVELS = frozenset({"Commun", "legacy"})

_cache: dict[tuple[str, str], ScopedRetriever] = {}
_lock = threading.Lock()


def build_chroma_filter(user_level: str) -> dict | None:
    """Build a Chroma ``where`` filter for level-scoped retrieval.

    Args:
        user_level: The user's academic level (``M1``, ``M2``, ``ALL``...).

    Returns:
        A filter dict, or ``None`` when the level sees everything.
    """
    if user_level in UNFILTERED_LEVELS:
        return None
    return {"level": {"$in": [user_level, *sorted(UNIVERSAL_LEVELS)]}}


def filter_docs_by_level(docs: list[Document], user_level: str) -> list[Document]:
    """Keep the documents visible to a level (post-hoc filter for BM25).

    Args:
        docs: Candidate documents.
        user_level: The user's academic level.

    Returns:
        Filtered list (unchanged for unfiltered levels).
    """
    if user_level in UNFILTERED_LEVELS:
        return docs
    allowed = {user_level} | UNIVERSAL_LEVELS
    return [d for d in docs if d.metadata.get("level", "legacy") in allowed]


def _tokenize(text: str) -> list[str]:
    """Case-insensitive whitespace tokenizer (Phase 2 brings a French-aware one)."""
    return text.lower().split()


class ScopedRetriever:
    """Hybrid retriever restricted to one academic level.

    Args:
        user_level: Academic level used for scoping.
        store: Vector store (LangChain ``VectorStore``); defaults to the shared one.
        corpus: Full corpus for BM25; defaults to the shared one.
        reranker: FlashRank ``Ranker`` or ``None``; defaults to the shared one.
    """

    def __init__(
        self,
        user_level: str,
        *,
        store: Any = None,
        corpus: list[Document] | None = None,
        reranker: Any = None,
    ) -> None:
        settings = get_settings()
        self.user_level = user_level
        self.k = settings.retrieval_k
        self.top_n = settings.rerank_top_n

        self._store = store if store is not None else resources.get_store()
        corpus = corpus if corpus is not None else resources.get_corpus()
        self._reranker = reranker if reranker is not None else resources.get_reranker()

        # 1. Vector retriever with native metadata filter
        search_kwargs: dict[str, Any] = {"k": self.k}
        chroma_filter = build_chroma_filter(user_level)
        if chroma_filter:
            search_kwargs["filter"] = chroma_filter
        self._vector_retriever = self._store.as_retriever(search_kwargs=search_kwargs)

        # 2. BM25 over the level-scoped corpus
        scoped_docs = filter_docs_by_level(corpus, user_level)
        self.doc_count = len(scoped_docs)
        self._ensemble = None
        if scoped_docs:
            from langchain_classic.retrievers import EnsembleRetriever
            from langchain_community.retrievers import BM25Retriever

            bm25 = BM25Retriever.from_documents(scoped_docs, preprocess_func=_tokenize)
            bm25.k = self.k
            w = settings.retrieval_bm25_weight
            self._ensemble = EnsembleRetriever(retrievers=[bm25, self._vector_retriever], weights=[w, 1.0 - w])

        logger.info(
            "ScopedRetriever ready: level=%s docs=%d rerank=%s",
            user_level if chroma_filter else "ALL",
            self.doc_count,
            "on" if self._reranker is not None and self.top_n > 0 else "off",
        )

    # ------------------------------------------------------------------

    def _rerank(self, docs: list[Document], query: str) -> list[Document]:
        if self._reranker is None or self.top_n <= 0 or not docs:
            return docs[: self.top_n] if self.top_n > 0 else docs
        from flashrank import RerankRequest

        passages = [{"id": i, "text": d.page_content, "meta": d.metadata} for i, d in enumerate(docs)]
        results = self._reranker.rerank(RerankRequest(query=query, passages=passages))
        return [docs[r["id"]] for r in results[: self.top_n]]

    def invoke(self, query: str) -> list[Document]:
        """Retrieve (and rerank) the documents relevant to ``query``.

        Args:
            query: Search query.

        Returns:
            Top documents, scoped to the user's level.
        """
        base = self._ensemble or self._vector_retriever
        docs = base.invoke(query)
        return self._rerank(docs, query)


def get_retriever(user_level: str) -> ScopedRetriever:
    """Return the cached retriever for a level (rebuilt when the store changes).

    Args:
        user_level: Academic level (``M1``, ``M2``, ``ALL``, ``admin``...).

    Returns:
        A shared :class:`ScopedRetriever` instance.
    """
    resources.get_store()  # refreshes the fingerprint if the base was rebuilt
    key = (user_level, resources.store_fingerprint())
    if key not in _cache:
        with _lock:
            if key not in _cache:
                # Drop retrievers of previous store generations
                for old in [k for k in _cache if k[1] != key[1]]:
                    _cache.pop(old, None)
                _cache[key] = ScopedRetriever(user_level)
    return _cache[key]


def reset_cache() -> None:
    """Forget every cached retriever (tests, store swap)."""
    with _lock:
        _cache.clear()
