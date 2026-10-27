"""Helpers on Chainlit's ``threads`` table: renaming and classification of conversations.

The chat renames a thread when a mode is started (``🎓 Quiz : …``,
``✏️ Exercice : …``, ``📄 Fiche de révision``, ``💻 Atelier Code``); the
backoffice derives the *kind* of a conversation from that prefix, in SQL
(:data:`THREAD_KIND_SQL`) as well as in Python (:func:`classify_thread`).
"""

from __future__ import annotations

import logging

from ela.db.engine import execute

logger = logging.getLogger(__name__)

# kind → (name prefix, French label); the order matters for the classification.
THREAD_KINDS: dict[str, tuple[str, str]] = {
    "quiz": ("🎓", "Quiz"),
    "exercise": ("✏", "Exercice"),  # U+270F, with or without the variation selector
    "sheet": ("📄", "Fiche"),
    "code": ("💻", "Code"),
    "chat": ("", "Chat"),
}
KIND_LABELS: dict[str, str] = {kind: label for kind, (_, label) in THREAD_KINDS.items()}

# SQL expression (``t`` = threads) giving the same classification as :func:`classify_thread`.
THREAD_KIND_SQL = (
    "CASE "
    + " ".join(f"WHEN t.name LIKE '{prefix}%' THEN '{kind}'" for kind, (prefix, _) in THREAD_KINDS.items() if prefix)
    + " ELSE 'chat' END"
)


def classify_thread(name: str | None) -> str:
    """Return the kind of a conversation (``quiz``, ``exercise``, ``sheet``, ``code`` or ``chat``) from its name."""
    if not name:
        return "chat"
    for kind, (prefix, _) in THREAD_KINDS.items():
        if prefix and name.startswith(prefix):
            return kind
    return "chat"


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
