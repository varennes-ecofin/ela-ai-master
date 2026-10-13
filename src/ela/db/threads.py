"""Helpers on Chainlit's ``threads`` table."""

from __future__ import annotations

import logging

from ela.db.engine import execute

logger = logging.getLogger(__name__)


async def rename_thread(thread_id: str, name: str) -> None:
    """Rename a conversation (used by quiz / code-workshop modes).

    Args:
        thread_id: Chainlit thread id.
        name: New display name.
    """
    try:
        await execute("UPDATE threads SET name = :name WHERE id = :id", {"name": name, "id": thread_id})
    except Exception:  # never break the chat flow for a cosmetic rename
        logger.warning("Could not rename thread %s", thread_id, exc_info=True)
