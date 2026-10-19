"""Review sheets cache (Phase 4d): one generated sheet per chapter, shared between students.

A sheet is keyed by course, chapter, level, prompt version and corpus
fingerprint, so it is regenerated when the slides or the prompt change.
No student data is stored here.

Revision ID: 0007_review_sheets
Revises: 0006_student_activity
Create Date: 2026-10-18
"""

from __future__ import annotations

from alembic import op

revision = "0007_review_sheets"
down_revision = "0006_student_activity"
branch_labels = None
depends_on = None

_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS review_sheets (
        id                BIGSERIAL PRIMARY KEY,
        course            TEXT NOT NULL,
        chapter           TEXT NOT NULL,
        level             TEXT NOT NULL DEFAULT '',
        prompt_version    TEXT NOT NULL,
        store_fingerprint TEXT NOT NULL,
        title             TEXT NOT NULL,
        payload           JSONB NOT NULL,
        markdown          TEXT NOT NULL,
        created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (course, chapter, level, prompt_version, store_fingerprint)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_review_sheets_course ON review_sheets (course, chapter)",
]


def upgrade() -> None:
    for statement in _STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS review_sheets")
