"""Exercise attempts (Phase 4b): one row per step checked in Socratic mode.

Revision ID: 0005_exercise_attempts
Revises: 0004_quiz_results
Create Date: 2026-10-18
"""

from __future__ import annotations

from alembic import op

revision = "0005_exercise_attempts"
down_revision = "0004_quiz_results"
branch_labels = None
depends_on = None

_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS exercise_attempts (
        id              BIGSERIAL PRIMARY KEY,
        user_identifier TEXT NOT NULL,
        thread_id       TEXT,
        exercise_id     TEXT NOT NULL,
        topic           TEXT NOT NULL,
        title           TEXT,
        level           TEXT,
        attempt_index   INTEGER NOT NULL,
        attempt         TEXT NOT NULL,
        verdict         TEXT NOT NULL,
        completed       BOOLEAN NOT NULL DEFAULT FALSE,
        hints_revealed  INTEGER NOT NULL DEFAULT 0,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_exercise_attempts_user ON exercise_attempts (user_identifier, created_at DESC)",
]


def upgrade() -> None:
    for statement in _STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS exercise_attempts")
