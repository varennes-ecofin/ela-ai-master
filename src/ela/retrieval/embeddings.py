"""Embedding model factory (``bge-m3`` through an API or locally).

Three providers are supported, all returning a LangChain ``Embeddings``:

* ``local`` — ``sentence-transformers`` (``BAAI/bge-m3`` by default, 1024 dims,
  8192-token context; multilingual, strong on French and on formulas);
* ``ilaas`` — the ILaaS OpenAI-compatible embedding service
  (``ILAAS_EMBED_BASE_URL``), same ``bge-m3`` model served remotely;
* ``openai`` — any other OpenAI-compatible endpoint (``EMBEDDING_BASE_URL``).

When ``EMBEDDING_LOCAL_FALLBACK`` is on, an API failure switches to the local
model **of the same name**: vectors are only interchangeable when the model is
identical, so the fallback is refused for a different model name.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from langchain_core.embeddings import Embeddings

from ela.config import Settings, get_settings

logger = logging.getLogger(__name__)

#: Query instruction recommended for bge-m3 is empty (symmetric model); kept
#: here so that a future model with an instruction prefix is a one-line change.
QUERY_PREFIX = ""


def _local_model_name(name: str) -> str:
    """Map an API model alias (``bge-m3``) to its Hugging Face repository."""
    aliases = {"bge-m3": "BAAI/bge-m3", "baai/bge-m3": "BAAI/bge-m3"}
    return aliases.get(name.lower(), name)


def build_local_embeddings(model: str, *, batch_size: int = 16) -> Embeddings:
    """Load a sentence-transformers model on CPU (normalised embeddings).

    Args:
        model: Model name or Hugging Face repository.
        batch_size: Encoding batch size.

    Returns:
        A ``HuggingFaceEmbeddings`` instance.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    name = _local_model_name(model)
    logger.info("Loading local embedding model %s ...", name)
    return HuggingFaceEmbeddings(
        model_name=name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
    )


def build_api_embeddings(model: str, base_url: str, api_key: str, *, batch_size: int = 16) -> Embeddings:
    """Build an ``OpenAIEmbeddings`` client for an OpenAI-compatible endpoint.

    ``check_embedding_ctx_length`` is disabled so that raw strings (not
    tiktoken token arrays) are sent, which non-OpenAI servers require.

    Args:
        model: Model name as exposed by the endpoint.
        base_url: Endpoint base URL (``.../v1``).
        api_key: Bearer token.
        batch_size: Inputs per request.

    Returns:
        An ``OpenAIEmbeddings`` instance.
    """
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=model,
        base_url=base_url,
        api_key=api_key,
        check_embedding_ctx_length=False,
        tiktoken_enabled=False,
        chunk_size=batch_size,
        max_retries=2,
        timeout=60,
    )


class FallbackEmbeddings(Embeddings):
    """Primary embeddings with a lazily-built local fallback of the same model.

    Args:
        primary: API-backed embeddings.
        fallback_factory: Callable building the local embeddings on demand.
    """

    def __init__(self, primary: Embeddings, fallback_factory) -> None:
        self._primary = primary
        self._factory = fallback_factory
        self._fallback: Embeddings | None = None
        self._lock = threading.Lock()
        self.using_fallback = False

    def _get_fallback(self) -> Embeddings:
        if self._fallback is None:
            with self._lock:
                if self._fallback is None:
                    self._fallback = self._factory()
        return self._fallback

    def _run(self, method: str, arg: Any) -> Any:
        if not self.using_fallback:
            try:
                return getattr(self._primary, method)(arg)
            except Exception as exc:  # network, auth, quota...
                logger.warning("Embedding API failed (%s) — switching to the local model.", exc)
                self.using_fallback = True
        return getattr(self._get_fallback(), method)(arg)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._run("embed_documents", texts)

    def embed_query(self, text: str) -> list[float]:
        return self._run("embed_query", text)


def build_embeddings(settings: Settings | None = None) -> Embeddings:
    """Build the embeddings configured by ``EMBEDDING_*`` settings.

    Args:
        settings: Settings override (defaults to the shared settings).

    Returns:
        A LangChain ``Embeddings``.

    Raises:
        RuntimeError: When an API provider is selected without an API key.
    """
    s = settings or get_settings()
    provider = s.embedding_provider
    if provider == "local":
        return build_local_embeddings(s.embedding_model, batch_size=s.embedding_batch_size)

    base_url, api_key = s.embedding_endpoint()
    if not api_key:
        raise RuntimeError(f"EMBEDDING_PROVIDER={provider} requires an API key (EMBEDDING_API_KEY or ILAAS_API_KEY).")
    primary = build_api_embeddings(s.embedding_model, base_url, api_key, batch_size=s.embedding_batch_size)
    if not s.embedding_local_fallback:
        return primary
    return FallbackEmbeddings(
        primary, lambda: build_local_embeddings(s.embedding_model, batch_size=s.embedding_batch_size)
    )
