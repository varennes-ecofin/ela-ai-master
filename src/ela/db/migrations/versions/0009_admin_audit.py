"""Backoffice audit log (Phase 5, lot 5.1): one row per administrative action.

Logins, logouts and, from lot 5.2 on, every change made through the
backoffice (accounts, quotas, ingestion, evaluation runs) land here so
that several teachers can share the administration with a usable trail.

Revision ID: 0009_admin_audit
Revises: 0008_code_runs
Create Date: 2026-10-21
"""

from __future__ import annotations

from alembic import op

revision = "0009_admin_audit"
down_revision = "0008_code_runs"
branch_labels = None
depends_on = None

_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS admin_audit (
        id          BIGSERIAL PRIMARY KEY,
        at          TIMESTAMPTZ NOT NULL DEFAULT now(),
        actor       TEXT NOT NULL,
        action      TEXT NOT NULL,
        target      TEXT,
        details     JSONB NOT NULL DEFAULT '{}'::jsonb,
        ip          TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_admin_audit_actor_at ON admin_audit (actor, at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_admin_audit_at ON admin_audit (at DESC)",
]


def upgrade() -> None:
    for statement in _STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS admin_audit")
