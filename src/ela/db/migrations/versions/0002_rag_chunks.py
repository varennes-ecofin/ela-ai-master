"""Knowledge base in PostgreSQL: ``rag_chunks`` (pgvector + French full text).

The table is only created when the ``vector`` extension can be enabled
(pgvector installed on the server). When it cannot, the migration logs a
warning and succeeds anyway: ``RETRIEVAL_BACKEND=chroma`` keeps working
without this table, and ``ela-ingest`` creates it later on a server where
pgvector is available (``PgVectorStore.ensure_schema``).

Revision ID: 0002_rag_chunks
Revises: 0001_baseline
Create Date: 2026-10-14
"""

from __future__ import annotations

import logging

from alembic import op
from sqlalchemy import text

revision = "0002_rag_chunks"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")


def upgrade() -> None:
    from ela.config import get_settings
    from ela.retrieval.store import _CREATE_TABLE_SQL, RAG_TABLE

    settings = get_settings()
    conn = op.get_bind()
    try:
        with conn.begin_nested():
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception as exc:  # not installed, or role not allowed to create it
        logger.warning(
            "pgvector indisponible (%s) : table %s non créée. Installez pgvector et exécutez "
            "`CREATE EXTENSION vector;` en superuser, ou utilisez RETRIEVAL_BACKEND=chroma.",
            exc.__class__.__name__,
            RAG_TABLE,
        )
        return
    for statement in _CREATE_TABLE_SQL.format(
        table=RAG_TABLE, dim=settings.embedding_dim, fts=settings.retrieval_fts_config
    ).split(";"):
        if statement.strip():
            conn.execute(text(statement))
    conn.execute(
        text(
            f"CREATE INDEX IF NOT EXISTS {RAG_TABLE}_embedding_idx ON {RAG_TABLE} "
            "USING hnsw (embedding vector_cosine_ops)"
        )
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS rag_chunks")
