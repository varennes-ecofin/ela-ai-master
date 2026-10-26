"""Background jobs of the backoffice (Phase 5, lot 5.3): ingestion runs with their live log.

Revision ID: 0010_jobs
Revises: 0009_admin_audit
Create Date: 2026-10-26
"""

from __future__ import annotations

from alembic import op

revision = "0010_jobs"
down_revision = "0009_admin_audit"
branch_labels = None
depends_on = None

_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id           BIGSERIAL PRIMARY KEY,
        kind         TEXT NOT NULL,
        mode         TEXT,
        target       TEXT,
        status       TEXT NOT NULL DEFAULT 'queued',
        started_by   TEXT NOT NULL,
        created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
        started_at   TIMESTAMPTZ,
        finished_at  TIMESTAMPTZ,
        progress     INTEGER NOT NULL DEFAULT 0,
        step         TEXT,
        log          TEXT NOT NULL DEFAULT '',
        result       JSONB NOT NULL DEFAULT '{}'::jsonb,
        error        TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs (created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status)",
]


def upgrade() -> None:
    for statement in _STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS jobs")
