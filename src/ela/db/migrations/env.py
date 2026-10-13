"""Alembic environment for ELA AI.

The database URL comes from the application settings (``DATABASE_URL``),
never from ``alembic.ini``. Migrations are written in plain SQL (no ORM
metadata, no autogenerate) because the Chainlit tables are owned by
Chainlit's data layer.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine, pool

from ela.config import get_settings


def _url() -> str:
    return get_settings().sync_database_url


def run_migrations_offline() -> None:
    """Emit the SQL without connecting (``alembic upgrade head --sql``)."""
    context.configure(url=_url(), literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply the migrations against the configured database."""
    engine = create_engine(_url(), poolclass=pool.NullPool, connect_args={"client_encoding": "utf8"})
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
