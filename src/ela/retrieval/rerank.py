"""Cross-encoder rerankers behind one small interface.

* ``local`` — ``sentence-transformers`` ``CrossEncoder`` (default
  ``BAAI/bge-reranker-v2-m3``: multilingual, best quality, ~2-4 s on CPU for
  25 passages);
* ``flashrank`` — ONNX-quantised models (``ms-marco-MultiBERT-L-12`` is the
  multilingual one, a few hundred ms on CPU);
* ``none`` — candidates are returned in fusion order.

Every reranker exposes :meth:`Reranker.rerank` returning ``(document, score)``
pairs sorted by decreasing relevance; scores are model-specific and only
comparable within one call.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document

from ela.config import Settings, get_settings

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    """Scores (query, passage) pairs."""

    name: str

    def rerank(self, query: str, docs: list[Document]) -> list[tuple[Document, float]]:
        """Return ``docs`` sorted by relevance to ``query`` with a score."""
        ...


class CrossEncoderReranker:
    """``sentence-transformers`` cross-encoder (``BAAI/bge-reranker-v2-m3``...).

    Args:
        model: Model name or Hugging Face repository.
        max_length: Token limit for each (query, passage) pair.
    """

    def __init__(self, model: str, *, max_length: int = 1024) -> None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker %s ...", model)
        self.name = model
        self._model = CrossEncoder(model, max_length=max_length, device="cpu")

    def rerank(self, query: str, docs: list[Document]) -> list[tuple[Document, float]]:
        if not docs:
            return []
        scores = self._model.predict([(query, d.page_content) for d in docs], batch_size=8)
        ranked = sorted(zip(docs, (float(s) for s in scores), strict=True), key=lambda x: x[1], reverse=True)
        return ranked


class FlashRankReranker:
    """FlashRank ONNX reranker.

    Args:
        model: FlashRank model name (``ms-marco-MultiBERT-L-12`` for French).
        cache_dir: Where FlashRank downloads the model.
    """

    def __init__(self, model: str, cache_dir: str) -> None:
        from flashrank import Ranker

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Loading FlashRank reranker %s ...", model)
        self.name = model
        self._ranker = Ranker(model_name=model, cache_dir=cache_dir)

    def rerank(self, query: str, docs: list[Document]) -> list[tuple[Document, float]]:
        if not docs:
            return []
        from flashrank import RerankRequest

        passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
        results = self._ranker.rerank(RerankRequest(query=query, passages=passages))
        return [(docs[r["id"]], float(r["score"])) for r in results]


def build_reranker(settings: Settings | None = None) -> Reranker | None:
    """Build the reranker selected by ``RERANK_PROVIDER`` (``None`` when disabled).

    Args:
        settings: Settings override.

    Returns:
        A reranker, or ``None`` when ``RERANK_PROVIDER=none`` / ``RERANK_TOP_N=0``
        or when the required library is missing (logged as a warning).
    """
    s = settings or get_settings()
    if s.rerank_provider == "none" or s.rerank_top_n <= 0:
        return None
    try:
        if s.rerank_provider == "flashrank":
            return FlashRankReranker(s.rerank_model_name, s.rerank_cache_dir)
        return CrossEncoderReranker(s.rerank_model_name, max_length=s.rerank_max_length)
    except ImportError as exc:
        logger.warning("Reranker library missing (%s) — reranking disabled.", exc)
        return None
