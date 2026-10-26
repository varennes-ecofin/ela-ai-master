"""In-memory one-shot store: short-lived payloads read once (pending imports, generated passwords).

Generated credentials must be shown to the administrator exactly once and
never persisted; the preview of a CSV import must survive between the
upload and the confirmation. Both live here, keyed by a random token, for a
few minutes, in the process memory (a single backoffice instance).
"""

from __future__ import annotations

import secrets
import threading
import time
from typing import Any


class OneShotStore:
    """Token → payload map with expiry; a payload is removed when read.

    Args:
        ttl_seconds: Default lifetime of a payload.
        max_items: Oldest payloads are dropped beyond this size.
    """

    def __init__(self, ttl_seconds: float = 900.0, max_items: int = 200) -> None:
        self.ttl = float(ttl_seconds)
        self.max_items = int(max_items)
        self._items: dict[str, tuple[float, str, Any]] = {}
        self._lock = threading.Lock()

    def _purge(self) -> None:
        now = time.monotonic()
        for token in [t for t, (exp, _, _) in self._items.items() if exp <= now]:
            self._items.pop(token, None)
        while len(self._items) > self.max_items:
            self._items.pop(next(iter(self._items)), None)

    def put(self, payload: Any, *, owner: str, ttl_seconds: float | None = None) -> str:
        """Store ``payload`` for ``owner`` and return its token."""
        token = secrets.token_urlsafe(24)
        with self._lock:
            self._purge()
            self._items[token] = (time.monotonic() + (ttl_seconds or self.ttl), owner, payload)
        return token

    def pop(self, token: str, *, owner: str) -> Any | None:
        """Return and remove the payload, or ``None`` if unknown, expired or owned by someone else."""
        with self._lock:
            self._purge()
            item = self._items.get(token)
            if item is None or item[1] != owner:
                return None
            del self._items[token]
            return item[2]

    def peek(self, token: str, *, owner: str) -> Any | None:
        """Return the payload without removing it (same rules as :meth:`pop`)."""
        with self._lock:
            self._purge()
            item = self._items.get(token)
            return None if item is None or item[1] != owner else item[2]
