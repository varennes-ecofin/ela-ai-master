"""Centralised configuration for ELA AI.

All runtime settings are read from environment variables (and from a
``.env`` file in the working directory, if present) through
``pydantic-settings``. Nothing else in the code base should call
``os.getenv`` for these values.

Usage::

    from ela.config import get_settings
    settings = get_settings()
    settings.llm_model_chat  # -> "gpt-oss-120b"

Environment variable names are the upper-cased field names
(``LLM_MODEL_CHAT``, ``CEREBRAS_API_KEY``, ...). See ``.env.example``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_core import PydanticUseDefault
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["cerebras", "ilaas", "groq"]


def find_env_file(start: Path | None = None, max_depth: int = 4) -> Path | None:
    """Locate the ``.env`` file.

    Uses ``ELA_ENV_FILE`` when set, otherwise the first ``.env`` found walking
    up from the current directory.

    Lets the admin app (run from ``apps/admin``) and CLI tools share the
    project-root ``.env`` without copying it around.

    Args:
        start: Directory to start from (defaults to the current directory).
        max_depth: Number of parent directories to inspect.

    Returns:
        Path of the ``.env`` file, or ``None`` if not found.
    """
    override = os.getenv("ELA_ENV_FILE")
    if override:
        return Path(override)
    current = (start or Path.cwd()).resolve()
    for _ in range(max_depth + 1):
        candidate = current / ".env"
        if candidate.is_file():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


ReasoningEffort = Literal["omit", "none", "minimal", "low", "medium", "high"]


class Settings(BaseSettings):
    """Typed view of the ELA environment.

    Attributes are grouped by concern; every attribute has a sensible
    default so that a minimal ``.env`` (API key + database) is enough
    to start the application.
    """

    model_config = SettingsConfigDict(
        env_file=find_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    ela_env: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = Field(default=False, description="Emit JSON log lines (recommended in Docker).")
    warmup_on_start: bool = Field(
        default=True,
        description="Load embeddings, vector store and reranker in the background at startup.",
    )
    storage_dir: str = Field(default=".files_ela", description="Local storage for uploaded files.")
    max_images_per_message: int = 2

    # ------------------------------------------------------------------
    # Database (Chainlit data layer + ELA tables)
    # ------------------------------------------------------------------
    database_url: str | None = Field(
        default=None,
        description=(
            "PostgreSQL URL. Any of postgresql://, postgresql+asyncpg:// or "
            "postgresql+psycopg2:// is accepted; the driver is adjusted per use."
        ),
    )
    admin_database_url: str | None = Field(
        default=None,
        description="Optional separate URL for the backoffice; defaults to database_url.",
    )
    db_pool_size: int = 5
    db_max_overflow: int = 5

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------
    history_max_turns: int = Field(default=12, description="Max user/assistant exchanges sent to the LLM.")
    history_max_chars: int = Field(default=24_000, description="Character budget for the history window.")

    # ------------------------------------------------------------------
    # LLM providers (all OpenAI-compatible)
    # ------------------------------------------------------------------
    llm_provider: ProviderName = "cerebras"
    llm_vision_provider: ProviderName | None = Field(
        default=None,
        description="Provider used when the message contains images. Defaults to llm_provider.",
    )
    llm_fallback_provider: ProviderName | None = Field(
        default=None,
        description="Provider used for the fallback model. Defaults to llm_provider.",
    )

    llm_model_chat: str = "gpt-oss-120b"
    llm_model_vision: str = "qwen-3.8-27b"
    llm_model_fallback: str | None = Field(
        default="qwen-3.8-27b",
        description="Model tried when the chat model fails. Set to 'none' to disable the fallback.",
    )

    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048
    llm_timeout_seconds: float = 90.0
    llm_max_retries: int = 2
    llm_reasoning_effort: ReasoningEffort = Field(
        default="low",
        description=(
            "Reasoning effort sent to reasoning-capable models (gpt-oss, qwen). "
            "'none' disables thinking on Qwen; 'omit' does not send the parameter at all."
        ),
    )

    # Cerebras
    cerebras_api_key: SecretStr | None = None
    cerebras_base_url: str = "https://api.cerebras.ai/v1"

    # ILaaS (French higher-education inference platform)
    ilaas_api_key: SecretStr | None = None
    ilaas_chat_base_url: str | None = Field(
        default=None,
        description="Base URL of the ILaaS chat-completions service (to be provided by the consortium).",
    )
    ilaas_embed_base_url: str = "https://rag-api.ilaas.fr/v1"

    # Groq (legacy / optional)
    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # ------------------------------------------------------------------
    # Retrieval (phase 2 will extend this block)
    # ------------------------------------------------------------------
    chroma_db_path: str = "./data/chroma_db"
    latex_source_dir: str = "./data/latex"
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    retrieval_k: int = Field(default=20, description="Candidates fetched by each retriever (BM25, vector).")
    retrieval_bm25_weight: float = Field(default=0.5, description="Weight of BM25 in the ensemble (vector = 1 - w).")
    rerank_top_n: int = Field(default=5, description="Documents kept after reranking (0 disables reranking).")
    rerank_model_name: str = "ms-marco-MiniLM-L-12-v2"
    rerank_cache_dir: str = Field(default="./data/flashrank", description="Where FlashRank stores its model.")
    chunk_size: int = 2000
    chunk_overlap: int = 200

    # ------------------------------------------------------------------
    # Normalisation of raw environment values
    # ------------------------------------------------------------------
    @field_validator("*", mode="before")
    @classmethod
    def _normalise_env_value(cls, value):
        """Make raw ``.env`` values robust before type validation.

        * Strips an inline ``# comment`` (python-dotenv keeps it when the
          value before it is empty).
        * Treats an empty/blank value (``VAR=``) as "not set": the field
          falls back to its default (``None`` for optional fields).

        Args:
            value: Raw value from the environment or the ``.env`` file.

        Returns:
            The cleaned value.

        Raises:
            PydanticUseDefault: When the value is blank, so pydantic applies
                the field default instead of validating an empty string.
        """
        if not isinstance(value, str):
            return value
        cleaned = value.strip()
        if " #" in cleaned:
            cleaned = cleaned.split(" #", 1)[0].rstrip()
        if not cleaned or cleaned.startswith("#"):
            raise PydanticUseDefault()
        return cleaned

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _with_driver(url: str, driver: str | None) -> str:
        """Rewrite the scheme of a PostgreSQL URL to use ``driver``.

        Args:
            url: A ``postgresql[+driver]://`` URL.
            driver: ``"asyncpg"``, ``"psycopg2"`` or ``None`` for the plain libpq scheme.

        Returns:
            The URL with the requested driver.
        """
        scheme, sep, rest = url.partition("://")
        if not sep or not scheme.startswith("postgres"):
            return url
        return (f"postgresql+{driver}" if driver else "postgresql") + "://" + rest

    def _require_database_url(self) -> str:
        if not self.database_url:
            raise ValueError("DATABASE_URL is not set (see .env.example).")
        return self.database_url

    @property
    def async_database_url(self) -> str:
        """``DATABASE_URL`` with the ``asyncpg`` driver (Chainlit + chat app)."""
        return self._with_driver(self._require_database_url(), "asyncpg")

    @property
    def sync_database_url(self) -> str:
        """``DATABASE_URL`` with the ``psycopg2`` driver (CLI, migrations)."""
        return self._with_driver(self._require_database_url(), "psycopg2")

    @property
    def libpq_database_url(self) -> str:
        """Plain ``postgresql://`` URL for the backoffice (psycopg2 DSN)."""
        return self._with_driver(self.admin_database_url or self._require_database_url(), None)

    @property
    def fallback_model(self) -> str | None:
        """Fallback model name, or ``None`` when disabled (``LLM_MODEL_FALLBACK=none``)."""
        value = (self.llm_model_fallback or "").strip()
        return None if value.lower() in {"", "none", "off", "false"} else value

    @property
    def reasoning_effort(self) -> str | None:
        """Reasoning effort to send, or ``None`` when ``LLM_REASONING_EFFORT=omit``."""
        return None if self.llm_reasoning_effort == "omit" else self.llm_reasoning_effort

    def provider_credentials(self, provider: ProviderName) -> tuple[str, str | None]:
        """Return ``(base_url, api_key)`` for a provider.

        Args:
            provider: One of ``cerebras``, ``ilaas``, ``groq``.

        Returns:
            Tuple of base URL and plain-text API key (``None`` if unset).

        Raises:
            ValueError: If the provider is unknown or its base URL is missing.
        """
        if provider == "cerebras":
            key = self.cerebras_api_key
            return self.cerebras_base_url, key.get_secret_value() if key else None
        if provider == "ilaas":
            if not self.ilaas_chat_base_url:
                raise ValueError(
                    "ILAAS_CHAT_BASE_URL is not set: the ILaaS inference endpoint is required to use provider 'ilaas'."
                )
            key = self.ilaas_api_key
            return self.ilaas_chat_base_url, key.get_secret_value() if key else None
        if provider == "groq":
            key = self.groq_api_key
            return self.groq_base_url, key.get_secret_value() if key else None
        raise ValueError(f"Unknown LLM provider: {provider!r}")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide ``Settings`` instance (cached).

    Returns:
        The validated settings object.
    """
    return Settings()
