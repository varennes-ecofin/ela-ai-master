"""Level-scoped hybrid retriever (vector + lexical, reranked).

Pipeline for one query:

1. **Hybrid search** in the chunk store (RRF of the vector and full-text
   legs), restricted to the levels the user may see. Course *slides* are
   searched first; *manuscripts* (HDR, articles) contribute a small extra
   quota so that long prose never floods the candidate list.
2. **Reranking** of every candidate with a cross-encoder.
3. **Selection** of ``top_n``: a definition/theorem chunk is replaced by its
   parent slide (fuller context for the LLM), duplicates are removed and
   refilled, and manuscripts are capped (``RETRIEVAL_MANUSCRIPT_MAX``).

One :class:`ScopedRetriever` is built per academic level and cached for the
current store generation, so a chat session only looks up an existing
object.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from langchain_core.documents import Document

from ela.config import Settings, get_settings
from ela.retrieval import resources
from ela.retrieval.store import Hit

logger = logging.getLogger(__name__)

# Levels that grant access to ALL documents (no filtering)
UNFILTERED_LEVELS = frozenset({"ALL", "admin"})
# Levels always included regardless of user level
UNIVERSAL_LEVELS = frozenset({"Commun", "legacy"})

_cache: dict[tuple[str, str], ScopedRetriever] = {}
_lock = threading.Lock()


def allowed_levels(user_level: str) -> list[str] | None:
    """Levels visible to a user (``None`` = everything).

    Args:
        user_level: The user's academic level (``M1``, ``M2``, ``ALL``...).

    Returns:
        Sorted list of levels, or ``None`` for unfiltered levels.
    """
    if user_level in UNFILTERED_LEVELS:
        return None
    return sorted({user_level, *UNIVERSAL_LEVELS})


def build_chroma_filter(user_level: str) -> dict | None:
    """Chroma ``where`` filter for a level (kept for the admin tools)."""
    levels = allowed_levels(user_level)
    return None if levels is None else {"level": {"$in": levels}}


def filter_docs_by_level(docs: list[Document], user_level: str) -> list[Document]:
    """Keep the documents visible to a level (post-hoc filter)."""
    levels = allowed_levels(user_level)
    if levels is None:
        return docs
    allowed = set(levels)
    return [d for d in docs if d.metadata.get("level", "legacy") in allowed]


class ScopedRetriever:
    """Hybrid retriever restricted to one academic level.

    Args:
        user_level: Academic level used for scoping.
        store: Chunk store (defaults to the shared one).
        embeddings: Embedding model (defaults to the shared one).
        reranker: Reranker or ``None`` (defaults to the shared one).
        settings: Settings override.
    """

    def __init__(
        self,
        user_level: str,
        *,
        store: Any = None,
        embeddings: Any = None,
        reranker: Any = None,
        settings: Settings | None = None,
    ) -> None:
        s = settings or get_settings()
        self.user_level = user_level
        self.levels = allowed_levels(user_level)
        self.k = s.retrieval_k
        self.manuscript_k = s.retrieval_manuscript_k
        self.manuscript_max = s.retrieval_manuscript_max
        self.top_n = s.rerank_top_n
        self.expand_elements = s.retrieval_expand_elements
        self._store = store if store is not None else resources.get_store()
        self._embeddings = embeddings if embeddings is not None else resources.get_embeddings()
        self._reranker = reranker if reranker is not None else resources.get_reranker()
        self.last_hits: list[Hit] = []
        logger.info(
            "ScopedRetriever ready: level=%s rerank=%s",
            user_level if self.levels else "ALL",
            getattr(self._reranker, "name", "off") if self._reranker is not None and self.top_n > 0 else "off",
        )

    # ------------------------------------------------------------------

    def search(self, query: str) -> list[Hit]:
        """Hybrid candidates (before reranking), slides first then manuscripts.

        Args:
            query: Search query.

        Returns:
            Fused candidates with their scores and leg ranks.
        """
        vector = self._embeddings.embed_query(query)
        hits = self._store.search(query, vector, k=self.k, levels=self.levels, doc_kinds=["slides"])
        if not hits:  # legacy store without doc_kind, or level without slides
            hits = self._store.search(query, vector, k=self.k, levels=self.levels)
        elif self.manuscript_k > 0:
            extra = self._store.search(query, vector, k=self.manuscript_k, levels=self.levels, doc_kinds=["manuscript"])
            seen = {h.doc.metadata.get("chunk_id") for h in hits}
            hits.extend(h for h in extra if h.doc.metadata.get("chunk_id") not in seen)
        return hits

    def _rerank(self, query: str, docs: list[Document]) -> list[Document]:
        """Order candidates by relevance (cross-encoder when enabled, fusion order otherwise)."""
        if self._reranker is None or self.top_n <= 0 or not docs:
            return docs
        ranked = self._reranker.rerank(query, docs)
        out = []
        for doc, score in ranked:
            doc.metadata = {**doc.metadata, "rerank_score": round(score, 4)}
            out.append(doc)
        return out

    def _select(self, ranked: list[Document]) -> list[Document]:
        """Pick the final ``top_n``: slides first, manuscripts capped, no duplicates.

        Typed elements are replaced by their parent slide (when enabled); a
        duplicate freed slot is filled by the next candidate, so the LLM always
        receives ``top_n`` distinct documents when enough candidates exist.
        """
        if self.top_n <= 0:
            return self._expand(ranked)
        expanded = self._expand(ranked)
        out: list[Document] = []
        manuscripts = 0
        for doc in expanded:
            if doc.metadata.get("doc_kind") == "manuscript":
                if manuscripts >= self.manuscript_max:
                    continue
                manuscripts += 1
            out.append(doc)
            if len(out) == self.top_n:
                break
        if len(out) < self.top_n:  # not enough slides: let manuscripts fill the remaining slots
            seen = {d.metadata.get("chunk_id") for d in out}
            out.extend(d for d in expanded if d.metadata.get("chunk_id") not in seen)
            out = out[: self.top_n]
        return out

    def _expand(self, docs: list[Document]) -> list[Document]:
        """Replace typed elements by their parent slide and drop duplicates."""
        parent_ids = (
            [d.metadata.get("parent_id") for d in docs if d.metadata.get("parent_id")] if self.expand_elements else []
        )
        parents = {p.metadata.get("chunk_id"): p for p in self._store.get(list(dict.fromkeys(parent_ids)))}
        out: list[Document] = []
        seen: set[str] = set()
        for doc in docs:
            pid = doc.metadata.get("parent_id")
            target = parents.get(pid) if pid else None
            if target is not None:
                target = Document(
                    page_content=target.page_content,
                    metadata={**target.metadata, "matched_element": doc.metadata.get("element_title", "")},
                )
                for key in ("rerank_score", "score"):
                    if key in doc.metadata:
                        target.metadata[key] = doc.metadata[key]
            else:
                target = doc
            cid = target.metadata.get("chunk_id") or id(target)
            if cid in seen:
                continue
            seen.add(cid)
            out.append(target)
        return out

    def invoke(self, query: str) -> list[Document]:
        """Retrieve, rerank and select the documents relevant to ``query``.

        Args:
            query: Search query.

        Returns:
            Top documents, scoped to the user's level. Each document carries
            ``score`` (fusion), ``rerank_score`` (when reranked) in its metadata.
        """
        hits = self.search(query)
        self.last_hits = hits
        docs = []
        for h in hits:
            h.doc.metadata = {**h.doc.metadata, "score": round(h.score, 5)}
            docs.append(h.doc)
        return self._select(self._rerank(query, docs))


def get_retriever(user_level: str) -> ScopedRetriever:
    """Return the cached retriever for a level (rebuilt when the store changes).

    Args:
        user_level: Academic level (``M1``, ``M2``, ``ALL``, ``admin``...).

    Returns:
        A shared :class:`ScopedRetriever` instance.
    """
    key = (user_level, resources.store_fingerprint())
    if key not in _cache:
        with _lock:
            if key not in _cache:
                for old in [k for k in _cache if k[1] != key[1]]:  # previous store generations
                    _cache.pop(old, None)
                _cache[key] = ScopedRetriever(user_level)
    return _cache[key]


def reset_cache() -> None:
    """Forget every cached retriever (tests, store swap)."""
    with _lock:
        _cache.clear()
