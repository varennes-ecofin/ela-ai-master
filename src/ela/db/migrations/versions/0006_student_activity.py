"""Student activity (Phase 4c): one row per question answered by the agent.

Feeds the student report (``/bilan``) and the backoffice « Pédagogie » page:
chapters worked, questions per chapter, subjects outside the corpus.

Revision ID: 0006_student_activity
Revises: 0005_exercise_attempts
Create Date: 2026-10-18
"""

from __future__ import annotations

from alembic import op

revision = "0006_student_activity"
down_revision = "0005_exercise_attempts"
branch_labels = None
depends_on = None

_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS student_activity (
        id                BIGSERIAL PRIMARY KEY,
        user_identifier   TEXT NOT NULL,
        thread_id         TEXT,
        level             TEXT,
        question          TEXT NOT NULL,
        course            TEXT,
        chapter           TEXT,
        n_searches        INTEGER NOT NULL DEFAULT 0,
        n_sources         INTEGER NOT NULL DEFAULT 0,
        cited             JSONB,
        general_knowledge BOOLEAN NOT NULL DEFAULT FALSE,
        refused           BOOLEAN NOT NULL DEFAULT FALSE,
        created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_student_activity_user ON student_activity (user_identifier, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_student_activity_course ON student_activity (course, chapter)",
]


def upgrade() -> None:
    for statement in _STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS student_activity")
