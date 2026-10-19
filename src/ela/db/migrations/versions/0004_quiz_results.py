"""Quiz results (Phase 4a): one row per answered question.

Feeds the targeted follow-up (« refaire un quiz sur mes erreurs »), the
adaptive difficulty and, later, the student dashboard and the backoffice
« Pédagogie » page.

Revision ID: 0004_quiz_results
Revises: 0003_chainlit_212_steps
Create Date: 2026-10-18
"""

from __future__ import annotations

from alembic import op

revision = "0004_quiz_results"
down_revision = "0003_chainlit_212_steps"
branch_labels = None
depends_on = None

_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS quiz_results (
        id              BIGSERIAL PRIMARY KEY,
        user_identifier TEXT NOT NULL,
        thread_id       TEXT,
        quiz_id         TEXT NOT NULL,
        topic           TEXT NOT NULL,
        difficulty      TEXT,
        level           TEXT,
        course          TEXT,
        chapter         TEXT,
        question_index  INTEGER NOT NULL,
        question        TEXT NOT NULL,
        options         JSONB NOT NULL,
        correct_index   INTEGER NOT NULL,
        answer_index    INTEGER NOT NULL,
        correct         BOOLEAN NOT NULL,
        sources         JSONB,
        answered_at     TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_quiz_results_user ON quiz_results (user_identifier, answered_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_quiz_results_quiz ON quiz_results (quiz_id)",
]


def upgrade() -> None:
    for statement in _STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS quiz_results")
