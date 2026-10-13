"""Baseline: Chainlit 2.12 tables + ELA account columns.

Idempotent on purpose: every statement uses ``IF NOT EXISTS`` so that the
migration can be applied to a brand-new database as well as to a database
created by hand before Alembic was introduced (VPS, local dev).

Revision ID: 0001_baseline
Revises: None
Create Date: 2026-10-13
"""

from __future__ import annotations

from alembic import op

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


_CHAINLIT_TABLES = [
    """
    CREATE TABLE IF NOT EXISTS users (
        id          UUID PRIMARY KEY,
        identifier  TEXT NOT NULL UNIQUE,
        metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
        "createdAt" TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS threads (
        id               UUID PRIMARY KEY,
        "createdAt"      TEXT,
        name             TEXT,
        "userId"         UUID REFERENCES users(id) ON DELETE CASCADE,
        "userIdentifier" TEXT,
        tags             TEXT[],
        metadata         JSONB
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS steps (
        id              UUID PRIMARY KEY,
        name            TEXT NOT NULL,
        type            TEXT NOT NULL,
        "threadId"      UUID NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
        "parentId"      UUID,
        streaming       BOOLEAN NOT NULL DEFAULT FALSE,
        "waitForAnswer" BOOLEAN,
        "isError"       BOOLEAN,
        metadata        JSONB,
        tags            TEXT[],
        input           TEXT,
        output          TEXT,
        "createdAt"     TEXT,
        command         TEXT,
        start           TEXT,
        "end"           TEXT,
        generation      JSONB,
        "showInput"     TEXT,
        language        TEXT,
        indent          INT,
        "defaultOpen"   BOOLEAN
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS elements (
        id             UUID PRIMARY KEY,
        "threadId"     UUID REFERENCES threads(id) ON DELETE CASCADE,
        type           TEXT,
        url            TEXT,
        "chainlitKey"  TEXT,
        name           TEXT NOT NULL,
        display        TEXT,
        "objectKey"    TEXT,
        size           TEXT,
        page           INT,
        language       TEXT,
        "forId"        UUID,
        mime           TEXT,
        props          JSONB,
        "autoPlay"     BOOLEAN,
        "playerConfig" JSONB
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feedbacks (
        id         UUID PRIMARY KEY,
        "forId"    UUID NOT NULL,
        "threadId" UUID,
        value      INT NOT NULL,
        comment    TEXT
    )
    """,
]

# Columns added by newer Chainlit versions, for databases created earlier.
_CHAINLIT_COLUMNS = [
    "ALTER TABLE steps ADD COLUMN IF NOT EXISTS command TEXT",
    'ALTER TABLE steps ADD COLUMN IF NOT EXISTS "defaultOpen" BOOLEAN',
    "ALTER TABLE steps ADD COLUMN IF NOT EXISTS indent INT",
    'ALTER TABLE elements ADD COLUMN IF NOT EXISTS "autoPlay" BOOLEAN',
    'ALTER TABLE elements ADD COLUMN IF NOT EXISTS "playerConfig" JSONB',
    'ALTER TABLE feedbacks ADD COLUMN IF NOT EXISTS "threadId" UUID',
]

# ELA account management (see ela.db.users)
_ELA_USER_COLUMNS = [
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash TEXT",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'student'",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS level TEXT NOT NULL DEFAULT 'M1'",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS daily_quota INTEGER DEFAULT 50",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login TIMESTAMP",
]

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_users_identifier ON users(identifier)",
    "CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active)",
    'CREATE INDEX IF NOT EXISTS idx_threads_user ON threads("userId")',
    'CREATE INDEX IF NOT EXISTS idx_steps_thread_type ON steps("threadId", type)',
    'CREATE INDEX IF NOT EXISTS idx_steps_created ON steps("createdAt")',
    'CREATE INDEX IF NOT EXISTS idx_elements_thread ON elements("threadId")',
    'CREATE INDEX IF NOT EXISTS idx_feedbacks_for ON feedbacks("forId")',
]

_VIEW = """
CREATE OR REPLACE VIEW v_daily_usage AS
SELECT
    u.id AS user_id,
    u.identifier,
    u.daily_quota,
    COUNT(s.id) AS messages_today,
    CASE WHEN u.daily_quota IS NULL THEN NULL
         ELSE u.daily_quota - COUNT(s.id)::INTEGER END AS remaining
FROM users u
LEFT JOIN threads t ON t."userId" = u.id
LEFT JOIN steps s ON s."threadId" = t.id
    AND s.type = 'user_message'
    AND s."createdAt"::DATE = CURRENT_DATE
GROUP BY u.id, u.identifier, u.daily_quota
"""


def upgrade() -> None:
    for stmt in [*_CHAINLIT_TABLES, *_CHAINLIT_COLUMNS, *_ELA_USER_COLUMNS, *_INDEXES, _VIEW]:
        op.execute(stmt)


def downgrade() -> None:
    # The baseline never drops user data: only the helper view is removed.
    op.execute("DROP VIEW IF EXISTS v_daily_usage")
