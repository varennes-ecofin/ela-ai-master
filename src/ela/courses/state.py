"""What the knowledge base holds, compared with the course files on disk.

Read-only queries on the vector store (``stats()``), aggregated per level and
course, plus a diagnostic against :func:`ela.courses.catalog.scan`:
a course on disk without chunks is *to ingest*, chunks without a folder
are *orphans* (deleted or moved course, pending an incremental update).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ela.courses.catalog import LEVELS, Tree

logger = logging.getLogger(__name__)

TO_INGEST = "to_ingest"
ORPHAN = "orphan"
EMPTY = "empty"  # folder without any .tex
OK = "ok"


@dataclass(frozen=True)
class CourseStats:
    """Chunks of one course in the store."""

    level: str
    course: str
    chunks: int
    by_type: dict[str, int] = field(default_factory=dict)
    by_kind: dict[str, int] = field(default_factory=dict)
    updated_at: datetime | None = None

    @property
    def key(self) -> str:
        """``level/course``."""
        return f"{self.level}/{self.course}"


@dataclass(frozen=True)
class KnowledgeBase:
    """Aggregated content of the store."""

    backend: str
    total: int = 0
    fingerprint: str = ""
    courses: list[CourseStats] = field(default_factory=list)
    by_level: dict[str, int] = field(default_factory=dict)
    by_type: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def course(self, level: str, name: str) -> CourseStats | None:
        """Stats of one course, or ``None``."""
        for c in self.courses:
            if c.level == level and c.course == name:
                return c
        return None


@dataclass(frozen=True)
class Diagnostic:
    """Disk vs store for one course key."""

    level: str
    course: str
    status: str  # OK, TO_INGEST, ORPHAN, EMPTY
    tex_files: int = 0
    chunks: int = 0
    updated_at: datetime | None = None

    @property
    def key(self) -> str:
        """``level/course``."""
        return f"{self.level}/{self.course}"


def knowledge_base(store: Any | None = None, settings: Any | None = None) -> KnowledgeBase:
    """Aggregate the store content; never raises (``error`` is set instead)."""
    try:
        if store is None:
            from ela.config import get_settings
            from ela.retrieval.store import build_store

            s = settings or get_settings()
            store = build_store(s)
        rows = store.stats()
        fingerprint = str(store.fingerprint())
        backend = getattr(store, "backend", "?")
    except Exception as exc:
        logger.warning("Knowledge base unavailable", exc_info=True)
        return KnowledgeBase(backend="?", error=f"{exc.__class__.__name__}: {exc}")

    per_course: dict[tuple[str, str], dict[str, Any]] = {}
    by_level: Counter = Counter()
    by_type: Counter = Counter()
    total = 0
    for r in rows:
        key = (r["level"], r["course"])
        entry = per_course.setdefault(key, {"n": 0, "types": Counter(), "kinds": Counter(), "updated": None})
        entry["n"] += r["n"]
        entry["types"][r["chunk_type"]] += r["n"]
        entry["kinds"][r["doc_kind"]] += r["n"]
        if r.get("updated_at") and (entry["updated"] is None or r["updated_at"] > entry["updated"]):
            entry["updated"] = r["updated_at"]
        by_level[r["level"]] += r["n"]
        by_type[r["chunk_type"]] += r["n"]
        total += r["n"]
    order = {lvl: i for i, lvl in enumerate(LEVELS)}
    courses = [
        CourseStats(
            level=k[0],
            course=k[1],
            chunks=v["n"],
            by_type=dict(v["types"]),
            by_kind=dict(v["kinds"]),
            updated_at=v["updated"],
        )
        for k, v in sorted(per_course.items(), key=lambda kv: (order.get(kv[0][0], 9), kv[0][1]))
    ]
    return KnowledgeBase(
        backend=backend,
        total=total,
        fingerprint=fingerprint,
        courses=courses,
        by_level=dict(by_level),
        by_type=dict(by_type),
    )


def diagnose(tree: Tree, kb: KnowledgeBase) -> list[Diagnostic]:
    """Compare the folders of ``tree`` with the courses of ``kb``.

    Returns:
        One :class:`Diagnostic` per course key, levels in order; ``OK`` rows first within a level.
    """
    disk = {(c.level, c.name): c for c in tree.all_courses}
    store = {(c.level, c.course): c for c in kb.courses}
    result: list[Diagnostic] = []
    for key in sorted(
        set(disk) | set(store), key=lambda k: ({lvl: i for i, lvl in enumerate(LEVELS)}.get(k[0], 9), k[1])
    ):
        on_disk, in_store = disk.get(key), store.get(key)
        if on_disk and in_store:
            status = OK
        elif on_disk:
            status = TO_INGEST if on_disk.tex_count else EMPTY
        else:
            status = ORPHAN
        result.append(
            Diagnostic(
                level=key[0],
                course=key[1],
                status=status,
                tex_files=on_disk.tex_count if on_disk else 0,
                chunks=in_store.chunks if in_store else 0,
                updated_at=in_store.updated_at if in_store else None,
            )
        )
    return result
