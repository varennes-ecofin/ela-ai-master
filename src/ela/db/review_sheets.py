"""Review sheets cache (table ``review_sheets``): generated once per chapter, shared.

Reads and writes never raise: without a database the sheet is simply
regenerated.
"""

from __future__ import annotations

import json
import logging

from ela.db.engine import execute, fetch_one
from ela.pedagogy.schemas import ReviewSheet

logger = logging.getLogger(__name__)


async def get_cached_sheet(
    course: str, chapter: str, level: str, *, prompt_version: str, store_fingerprint: str
) -> ReviewSheet | None:
    """Return the cached sheet of a chapter for this prompt version and corpus, or ``None``."""
    try:
        row = await fetch_one(
            """
            SELECT payload FROM review_sheets
            WHERE course = :course AND chapter = :chapter AND level = :level
              AND prompt_version = :pv AND store_fingerprint = :fp
            """,
            {"course": course, "chapter": chapter, "level": level, "pv": prompt_version, "fp": store_fingerprint},
        )
    except Exception:
        logger.warning("Review sheet cache unavailable", exc_info=True)
        return None
    if not row:
        return None
    payload = row["payload"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    try:
        return ReviewSheet.model_validate(payload)
    except Exception:
        logger.warning("Cached review sheet for %s › %s is not readable; regenerating", course, chapter)
        return None


async def store_sheet(sheet: ReviewSheet, *, prompt_version: str, store_fingerprint: str) -> bool:
    """Store (or replace) the sheet of a chapter (returns ``False`` instead of raising)."""
    try:
        await execute(
            """
            INSERT INTO review_sheets
                (course, chapter, level, prompt_version, store_fingerprint, title, payload, markdown)
            VALUES (:course, :chapter, :level, :pv, :fp, :title, CAST(:payload AS jsonb), :markdown)
            ON CONFLICT (course, chapter, level, prompt_version, store_fingerprint)
            DO UPDATE SET title = EXCLUDED.title, payload = EXCLUDED.payload, markdown = EXCLUDED.markdown,
                          created_at = now()
            """,
            {
                "course": sheet.course,
                "chapter": sheet.chapter,
                "level": sheet.level,
                "pv": prompt_version,
                "fp": store_fingerprint,
                "title": sheet.title,
                "payload": sheet.model_dump_json(),
                "markdown": sheet.to_markdown(),
            },
        )
        return True
    except Exception:
        logger.warning("Could not store the review sheet of %s › %s", sheet.course, sheet.chapter, exc_info=True)
        return False
