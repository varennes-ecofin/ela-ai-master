"""Code runs (Phase 4e): one row per script executed in the sandbox.

Enforces the daily quota per student and feeds the backoffice.

Revision ID: 0008_code_runs
Revises: 0007_review_sheets
Create Date: 2026-10-18
"""

from __future__ import annotations

from alembic import op

revision = "0008_code_runs"
down_revision = "0007_review_sheets"
branch_labels = None
depends_on = None

_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS code_runs (
        id              BIGSERIAL PRIMARY KEY,
        user_identifier TEXT NOT NULL,
        thread_id       TEXT,
        origin          TEXT NOT NULL DEFAULT 'workshop',
        topic           TEXT,
        language        TEXT NOT NULL DEFAULT 'python',
        status          TEXT NOT NULL,
        elapsed_ms      INTEGER NOT NULL DEFAULT 0,
        n_images        INTEGER NOT NULL DEFAULT 0,
        error           TEXT,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_code_runs_user_day ON code_runs (user_identifier, created_at DESC)",
]


def upgrade() -> None:
    for statement in _STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS code_runs")
