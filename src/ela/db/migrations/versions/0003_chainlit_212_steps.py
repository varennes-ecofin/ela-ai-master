"""Columns written by Chainlit 2.12 steps (``autoCollapse``, ``modes``).

``cl.Step.to_dict()`` always includes ``autoCollapse`` and ``cl.Message``
may include ``modes``; without these columns the ``INSERT`` of every tool
step fails silently in the data layer, so the agent steps (and anything
nested under them) disappear when a thread is resumed.

Revision ID: 0003_chainlit_212_steps
Revises: 0002_rag_chunks
Create Date: 2026-10-18
"""

from __future__ import annotations

from alembic import op

revision = "0003_chainlit_212_steps"
down_revision = "0002_rag_chunks"
branch_labels = None
depends_on = None

_COLUMNS = [
    'ALTER TABLE steps ADD COLUMN IF NOT EXISTS "autoCollapse" BOOLEAN',
    "ALTER TABLE steps ADD COLUMN IF NOT EXISTS modes JSONB",
]


def upgrade() -> None:
    for statement in _COLUMNS:
        op.execute(statement)


def downgrade() -> None:
    op.execute('ALTER TABLE steps DROP COLUMN IF EXISTS "autoCollapse"')
    op.execute("ALTER TABLE steps DROP COLUMN IF EXISTS modes")
