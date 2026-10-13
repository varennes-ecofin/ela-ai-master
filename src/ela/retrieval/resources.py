"""Process-wide retrieval resources (embeddings, vector store, reranker).

Each resource is built lazily, once, behind a lock. The vector store is
keyed on a *fingerprint* of the Chroma directory so that a rebuild of the
knowledge base (backoffice ingestion) is picked up by new sessions
without restarting the process.

Nothing heavy is imported at module import time: ``sentence-transformers``
and ``chromadb`` are only imported inside the builders, which keeps tests
and CLI tools fast.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from ela.config import get_settings

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_embeddings: Any = None
_reranker: Any = None
_store: Any = None
_store_fingerprint: str | None = None
_corpus_cache: dict[str, list[Document]] = {}

# Builders are module attributes so tests can swap them for fakes.
EmbeddingsBuilder = Callable[[], Any]
StoreBuilder = Callable[[Any], Any]
RerankerBuilder = Callable[[], Any]


# ---------------------------------------------------------------------------
# Default builders
# ---------------------------------------------------------------------------


def _build_embeddings() -> Any:
    from langchain_huggingface import HuggingFaceEmbeddings

    name = get_settings().embedding_model_name
    logger.info("Loading embedding model %s ...", name)
    return HuggingFaceEmbeddings(model_name=name)


def _build_store(embeddings: Any) -> Any:
    from langchain_chroma import Chroma

    path = get_settings().chroma_db_path
    if not os.path.isdir(path):
        logger.error("Chroma directory '%s' not found — run `ela-ingest` first.", path)
    return Chroma(persist_directory=path, embedding_function=embeddings)


def _build_reranker() -> Any:
    settings = get_settings()
    if settings.rerank_top_n <= 0:
        return None
    try:
        from flashrank import Ranker
    except ImportError:
        logger.warning("flashrank not installed — reranking disabled.")
        return None
    Path(settings.rerank_cache_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Loading reranker %s ...", settings.rerank_model_name)
    return Ranker(model_name=settings.rerank_model_name, cache_dir=settings.rerank_cache_dir)


embeddings_builder: EmbeddingsBuilder = _build_embeddings
store_builder: StoreBuilder = _build_store
reranker_builder: RerankerBuilder = _build_reranker


# ---------------------------------------------------------------------------
# Fingerprint of the on-disk store
# ---------------------------------------------------------------------------


def store_fingerprint() -> str:
    """Return a cheap fingerprint of the Chroma directory.

    Based on the size and mtime of ``chroma.sqlite3``; changes whenever the
    knowledge base is rebuilt (atomic swap in the backoffice).

    Returns:
        A string fingerprint (``"missing"`` if the store does not exist).
    """
    sqlite = Path(get_settings().chroma_db_path) / "chroma.sqlite3"
    try:
        st = sqlite.stat()
    except FileNotFoundError:
        return "missing"
    return f"{st.st_size}:{int(st.st_mtime)}"


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


def get_reranker() -> Any:
    """Return the shared reranker, or ``None`` when disabled/unavailable."""
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                _reranker = reranker_builder() or _NoReranker()
    return None if isinstance(_reranker, _NoReranker) else _reranker


class _NoReranker:
    """Sentinel: reranking disabled (avoids rebuilding on every call)."""


def get_store() -> Any:
    """Return the shared vector store, rebuilt if the on-disk base changed."""
    global _store, _store_fingerprint
    fp = store_fingerprint()
    if _store is None or fp != _store_fingerprint:
        with _lock:
            if _store is None or fp != _store_fingerprint:
                if _store is not None:
                    logger.info("Knowledge base changed on disk — reloading vector store.")
                _store = store_builder(get_embeddings())
                _store_fingerprint = fp
                _corpus_cache.clear()
    return _store


def get_corpus() -> list[Document]:
    """Return every document of the store as LangChain ``Document`` objects.

    Loaded once per store generation; used to build the BM25 indexes.

    Returns:
        List of documents (empty if the store is empty).
    """
    store = get_store()
    key = _store_fingerprint or "none"
    if key not in _corpus_cache:
        with _lock:
            if key not in _corpus_cache:
                data = store.get()
                docs = [
                    Document(page_content=txt, metadata=meta or {})
                    for txt, meta in zip(data.get("documents") or [], data.get("metadatas") or [], strict=False)
                ]
                _corpus_cache[key] = docs
                logger.info("Corpus loaded: %d chunks", len(docs))
    return _corpus_cache[key]


def warmup(block: bool = False) -> threading.Thread | None:
    """Load embeddings, store, corpus and reranker ahead of the first request.

    Args:
        block: Load synchronously when ``True``; otherwise in a daemon thread.

    Returns:
        The background thread (``None`` when blocking).
    """

    def _run() -> None:
        try:
            get_embeddings()
            get_corpus()
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
    global _embeddings, _reranker, _store, _store_fingerprint
    with _lock:
        _embeddings = None
        _reranker = None
        _store = None
        _store_fingerprint = None
        _corpus_cache.clear()
        from ela.retrieval import retriever as _r

        _r.reset_cache()


@dataclass(frozen=True)
class ResourceStatus:
    """Snapshot of what is loaded (for health checks / admin page)."""

    embeddings_loaded: bool
    store_loaded: bool
    reranker_loaded: bool
    corpus_size: int | None
    fingerprint: str | None


def status() -> ResourceStatus:
    """Return the current load state without triggering any load."""
    key = _store_fingerprint or "none"
    return ResourceStatus(
        embeddings_loaded=_embeddings is not None,
        store_loaded=_store is not None,
        reranker_loaded=_reranker is not None and not isinstance(_reranker, _NoReranker),
        corpus_size=len(_corpus_cache[key]) if key in _corpus_cache else None,
        fingerprint=_store_fingerprint,
    )
