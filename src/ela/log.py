"""Logging configuration for ELA AI.

Call :func:`configure_logging` once at process start (chat app, admin app,
CLI). Modules then simply use ``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

from ela.config import get_settings

_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    """Minimal JSON line formatter (one object per record)."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - stdlib API
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str | None = None, json_lines: bool | None = None) -> None:
    """Configure the root logger once.

    Args:
        level: Log level name; defaults to ``LOG_LEVEL`` from the settings.
        json_lines: Emit JSON lines instead of text; defaults to ``LOG_JSON``.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    level = (level or settings.log_level).upper()
    json_lines = settings.log_json if json_lines is None else json_lines

    handler = logging.StreamHandler(sys.stdout)
    if json_lines:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s — %(message)s", "%H:%M:%S"))

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)

    # Quieten chatty third-party loggers
    for name in ("httpx", "httpcore", "openai", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(name).setLevel(max(logging.WARNING, root.level))

    _CONFIGURED = True
