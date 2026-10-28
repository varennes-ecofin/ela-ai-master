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

    Lets the backoffice (run from ``apps/backoffice``) and CLI tools share the
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

    # The ``.env`` file is NOT fixed here: it is located when :func:`get_settings`
    # builds the instance (``_env_file=find_env_file()``), so that importing this
    # module never binds the process to a file (tests stay hermetic, ``ELA_ENV_FILE``
    # can be changed before the first ``get_settings()`` call).
    model_config = SettingsConfigDict(
        env_file=None,
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
    # Knowledge base: ingestion, embeddings, store, retrieval, reranking
    # ------------------------------------------------------------------
    latex_source_dir: str = "./data/latex"
    annales_source_dir: str = Field(
        default="./data/annales",
        description="Past exams and corrections (same {level}/{course} layout); stored now, ingested by a later lot.",
    )
    chunk_target_chars: int = Field(default=1200, description="Target size of manuscript text chunks.")
    chunk_max_chars: int = Field(default=2500, description="Hard limit above which a slide/section is split.")
    ingest_include_notes: bool = Field(default=True, description="Attach beamer \\note{} speaker notes to slides.")

    embedding_provider: Literal["local", "ilaas", "openai"] = Field(
        default="local",
        description="local = sentence-transformers on CPU; ilaas/openai = OpenAI-compatible embedding API.",
    )
    embedding_model: str = Field(default="BAAI/bge-m3", description="Model name (local repo or API model id).")
    embedding_dim: int = Field(default=1024, description="Vector dimension of the embedding model.")
    embedding_base_url: str | None = Field(default=None, description="API base URL (provider 'openai').")
    embedding_api_key: SecretStr | None = Field(default=None, description="API key (defaults to ILAAS_API_KEY).")
    embedding_batch_size: int = 16
    embedding_local_fallback: bool = Field(
        default=True,
        description="When the embedding API fails, load the same model locally (vectors stay compatible).",
    )

    retrieval_backend: Literal["pgvector", "chroma"] = Field(
        default="pgvector",
        description="pgvector = rag_chunks table in DATABASE_URL (hybrid SQL); chroma = local directory + BM25.",
    )
    chroma_db_path: str = "./data/chroma_db"
    retrieval_k: int = Field(default=20, description="Candidates fetched from the store before reranking.")
    retrieval_manuscript_k: int = Field(
        default=4,
        description="Extra candidates taken from manuscripts (HDR, articles); slides are searched first.",
    )
    retrieval_manuscript_max: int = Field(
        default=1,
        description="Max manuscript chunks kept in the final top-n after reranking (slides come first).",
    )
    retrieval_lexical_weight: float = Field(default=0.5, description="Weight of the full-text leg (vector = 1 - w).")
    retrieval_rrf_k: int = Field(default=60, description="Reciprocal Rank Fusion constant.")
    retrieval_fts_config: str = Field(default="french", description="PostgreSQL text-search configuration.")
    retrieval_expand_elements: bool = Field(
        default=True,
        description="Replace a retrieved definition/theorem chunk by its parent slide (fuller context).",
    )

    rerank_provider: Literal["local", "flashrank", "none"] = Field(
        default="none",
        description=(
            "none = fusion order (best measured configuration with bge-m3, 0.25 s/question); "
            "flashrank = ONNX cross-encoder; local = sentence-transformers CrossEncoder."
        ),
    )
    rerank_model: str | None = Field(
        default=None,
        description="Reranker model; default ms-marco-MultiBERT-L-12 (flashrank) or BAAI/bge-reranker-v2-m3 (local).",
    )
    rerank_top_n: int = Field(default=5, description="Documents kept after reranking (0 disables reranking).")
    rerank_max_length: int = Field(default=1024, description="Token limit per (query, passage) pair (local).")
    rerank_cache_dir: str = Field(default="./data/flashrank", description="Where FlashRank stores its model.")

    # ------------------------------------------------------------------
    # Agent (Phase 3)
    # ------------------------------------------------------------------
    prompt_version: str = Field(default="v1", description="Version of the prompt files in ela/agent/prompts/.")
    agent_max_model_calls: int = Field(default=8, description="Model calls allowed per student message.")
    agent_max_tool_calls: int = Field(default=8, description="Tool calls allowed per student message (all tools).")
    agent_max_searches: int = Field(default=4, description="search_course calls allowed per student message.")
    agent_search_k: int = Field(default=5, description="Default number of passages returned by search_course.")
    agent_search_k_max: int = Field(default=8, description="Upper bound of the k argument of search_course.")
    agent_outline_max_chars: int = Field(default=6000, description="Character budget of get_course_outline.")
    agent_summary_enabled: bool = Field(
        default=True, description="Summarise the turns that leave the history window (one model call, cached)."
    )

    # ------------------------------------------------------------------
    # Tracing (Langfuse, optional)
    # ------------------------------------------------------------------
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str | None = Field(default=None, description="Langfuse base URL (self-hosted); cloud when empty.")
    langfuse_send_user_id: bool = Field(
        default=False, description="Attach the student identifier to traces (off: anonymous traces)."
    )

    # ------------------------------------------------------------------
    # Code sandbox (Atelier Code 2.0, optional)
    # ------------------------------------------------------------------
    sandbox_url: str | None = Field(
        default=None,
        description="Base URL of the sandbox service (apps/sandbox); empty = code is generated but never executed.",
    )
    sandbox_timeout_seconds: float = Field(default=30.0, description="Wall-clock limit of one execution.")
    sandbox_daily_quota: int = Field(default=10, description="Executions per student and per day (0 = unlimited).")
    code_max_tokens: int = Field(
        default=4096, description="Completion budget for script generation, fix and interpretation."
    )
    code_reasoning_effort: ReasoningEffort = Field(
        default="medium", description="Reasoning effort for the Atelier Code (generation, fix, interpretation)."
    )
    code_auto_fix: bool = Field(
        default=True, description="On a failed run, fix the script once and re-run before showing the error."
    )

    # ------------------------------------------------------------------
    # Authentication and backoffice (Phase 5)
    # ------------------------------------------------------------------
    auth_provider: Literal["local", "cas"] = Field(
        default="local",
        description="local = password accounts of the users table; cas = university CAS (not implemented yet).",
    )
    backoffice_secret_key: SecretStr | None = Field(
        default=None,
        description="Secret used to sign the backoffice session cookie (required to start the backoffice).",
    )
    backoffice_root_path: str = Field(
        default="/backoffice",
        description="URL prefix under which the backoffice is served (same locally and behind the reverse proxy).",
    )
    backoffice_session_hours: float = Field(default=8.0, description="Lifetime of a backoffice session.")
    backoffice_timezone: str = Field(default="Europe/Paris", description="Time zone used to display timestamps.")
    backoffice_login_max_failures: int = Field(
        default=5, description="Failed logins per identifier before a temporary lock (0 = never lock)."
    )
    backoffice_login_lock_minutes: float = Field(default=15.0, description="Duration of the login lock.")

    # ------------------------------------------------------------------
    # Evaluation (ela-eval answer)
    # ------------------------------------------------------------------
    eval_judge_model: str | None = Field(
        default=None,
        description="Model used as LLM judge by `ela-eval answer` (defaults to LLM_MODEL_CHAT).",
    )
    eval_judge_provider: ProviderName | None = Field(
        default=None,
        description="Provider of the judge model (defaults to LLM_PROVIDER). Kept fixed when the backoffice "
        "evaluates another model so that fidelities stay comparable.",
    )
    eval_judge_reasoning_effort: ReasoningEffort = Field(
        default="medium", description="Reasoning effort of the judge (accuracy matters more than latency here)."
    )
    eval_dataset: str = Field(
        default="eval/questions.yaml", description="Reference question set (relative to the working directory)."
    )
    eval_reports_dir: str = Field(
        default="eval/reports",
        description="Where `ela-eval` writes its JSON reports; listed by the backoffice (shared volume in Docker).",
    )

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
    def backoffice_prefix(self) -> str:
        """``BACKOFFICE_ROOT_PATH`` normalised: leading slash, no trailing slash, empty for the root."""
        value = (self.backoffice_root_path or "").strip().strip("/")
        return f"/{value}" if value else ""

    @property
    def sandbox_enabled(self) -> bool:
        """Whether student code may be executed (``SANDBOX_URL`` set)."""
        return bool(self.sandbox_url and self.sandbox_url.strip())

    @property
    def reasoning_effort(self) -> str | None:
        """Reasoning effort to send, or ``None`` when ``LLM_REASONING_EFFORT=omit``."""
        return None if self.llm_reasoning_effort == "omit" else self.llm_reasoning_effort

    @property
    def rerank_model_name(self) -> str:
        """Reranker model: ``RERANK_MODEL`` or the default of the selected provider."""
        if self.rerank_model:
            return self.rerank_model
        return "ms-marco-MultiBERT-L-12" if self.rerank_provider == "flashrank" else "BAAI/bge-reranker-v2-m3"

    def embedding_endpoint(self) -> tuple[str, str | None]:
        """Return ``(base_url, api_key)`` of the embedding API.

        Provider ``ilaas`` uses ``ILAAS_EMBED_BASE_URL`` / ``ILAAS_API_KEY``;
        provider ``openai`` uses ``EMBEDDING_BASE_URL`` / ``EMBEDDING_API_KEY``.
        ``EMBEDDING_API_KEY`` overrides the ILaaS key when both are set.

        Raises:
            ValueError: For the local provider, or when the base URL is missing.
        """
        if self.embedding_provider == "local":
            raise ValueError("The local embedding provider has no API endpoint.")
        key = self.embedding_api_key or (self.ilaas_api_key if self.embedding_provider == "ilaas" else None)
        base_url = self.ilaas_embed_base_url if self.embedding_provider == "ilaas" else self.embedding_base_url
        if not base_url:
            raise ValueError("EMBEDDING_BASE_URL is not set for EMBEDDING_PROVIDER=openai.")
        return base_url, key.get_secret_value() if key else None

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

    The ``.env`` file is located now (``ELA_ENV_FILE`` or the first ``.env``
    walking up from the current directory); process variables keep priority.

    Returns:
        The validated settings object.
    """
    return Settings(_env_file=find_env_file())
