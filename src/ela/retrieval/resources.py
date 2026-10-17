"""Process-wide retrieval resources (embeddings, chunk store, reranker).

Each resource is built lazily, once, behind a lock. The store exposes a
*fingerprint* that changes whenever the knowledge base is re-ingested, so
level-scoped retrievers are rebuilt without restarting the process.

Nothing heavy is imported at module import time: ``sentence-transformers``,
``chromadb`` and ``flashrank`` are only imported inside the builders, which
keeps tests and CLI tools fast.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ela.config import get_settings

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_embeddings: Any = None
_reranker: Any = None
_store: Any = None

# Builders are module attributes so tests can swap them for fakes.
EmbeddingsBuilder = Callable[[], Any]
StoreBuilder = Callable[[], Any]
RerankerBuilder = Callable[[], Any]


def _build_embeddings() -> Any:
    from ela.retrieval.embeddings import build_embeddings

    return build_embeddings(get_settings())


def _build_store() -> Any:
    from ela.retrieval.store import build_store

    return build_store(get_settings())


def _build_reranker() -> Any:
    from ela.retrieval.rerank import build_reranker

    return build_reranker(get_settings())


embeddings_builder: EmbeddingsBuilder = _build_embeddings
store_builder: StoreBuilder = _build_store
reranker_builder: RerankerBuilder = _build_reranker


class _NoReranker:
    """Sentinel: reranking disabled (avoids rebuilding on every call)."""


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def get_embeddings() -> Any:
    """Return the shared embedding model (loaded on first call)."""
    global _embeddings
    if _embeddings is None:
        with _lock:
            if _embeddings is None:
                _embeddings = embeddings_builder()
    return _embeddings


def get_store() -> Any:
    """Return the shared chunk store (:class:`ela.retrieval.store.ChunkStore`)."""
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = store_builder()
                logger.info("Chunk store ready (backend=%s)", getattr(_store, "backend", "?"))
    return _store


def store_fingerprint() -> str:
    """Fingerprint of the current knowledge base (changes after an ingestion)."""
    try:
        return str(get_store().fingerprint())
    except Exception as exc:  # store unreachable: keep serving with the last retrievers
        logger.warning("Store fingerprint unavailable: %s", exc)
        return "unavailable"


def get_reranker() -> Any:
    """Return the shared reranker, or ``None`` when disabled/unavailable."""
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                _reranker = reranker_builder() or _NoReranker()
    return None if isinstance(_reranker, _NoReranker) else _reranker


def warmup(block: bool = False) -> threading.Thread | None:
    """Load embeddings, store and reranker ahead of the first request.

    Args:
        block: Load synchronously when ``True``; otherwise in a daemon thread.

    Returns:
        The background thread (``None`` when blocking).
    """

    def _run() -> None:
        try:
            get_embeddings()
            store = get_store()
            logger.info("Knowledge base: %d chunks", store.count())
            get_reranker()
            logger.info("Retrieval resources ready.")
        except Exception:
            logger.exception("Warmup failed")

    if block:
        _run()
        return None
    thread = threading.Thread(target=_run, name="ela-warmup", daemon=True)
    thread.start()
    return thread


def reset(for_tests: bool = True) -> None:
    """Drop every cached resource (tests, or after swapping builders)."""
    global _embeddings, _reranker, _store
    with _lock:
        _embeddings = None
        _reranker = None
        _store = None
        from ela.retrieval import retriever as _r

        _r.reset_cache()


@dataclass(frozen=True)
class ResourceStatus:
    """Snapshot of what is loaded (for health checks / admin page)."""

    embeddings_loaded: bool
    store_loaded: bool
    reranker_loaded: bool
    backend: str | None
    corpus_size: int | None


def status() -> ResourceStatus:
    """Return the current load state without triggering any load."""
    size = None
    if _store is not None:
        try:
            size = int(_store.count())
        except Exception:
            size = None
    return ResourceStatus(
        embeddings_loaded=_embeddings is not None,
        store_loaded=_store is not None,
        reranker_loaded=_reranker is not None and not isinstance(_reranker, _NoReranker),
        backend=getattr(_store, "backend", None) if _store is not None else None,
        corpus_size=size,
    )
