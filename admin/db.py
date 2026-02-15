# admin/db.py
"""
Database connection module for the ELA backoffice.

Provides a shared PostgreSQL connection pool using psycopg2.
All queries go through get_connection() which returns a context-managed
connection from the pool.
"""

import os
import streamlit as st
# import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager


def get_database_url() -> str:
    """Retrieve database URL from environment or Streamlit secrets.

    Returns:
        PostgreSQL connection string (psycopg2 format).
    """
    # Priority: env var > streamlit secrets
    url = os.getenv("ADMIN_DATABASE_URL")
    if url:
        return url

    try:
        return st.secrets["ADMIN_DATABASE_URL"]
    except (KeyError, FileNotFoundError):
        pass

    raise RuntimeError(
        "ADMIN_DATABASE_URL not found. "
        "Set it as an environment variable or in .streamlit/secrets.toml"
    )


@st.cache_resource
def _get_pool():
    """Create a cached connection pool (one per Streamlit app lifecycle).

    Returns:
        psycopg2 connection pool.
    """
    from psycopg2 import pool

    db_url = get_database_url()
    return pool.ThreadedConnectionPool(minconn=1, maxconn=5, dsn=db_url)


@contextmanager
def get_connection():
    """Yield a database connection from the pool.

    Usage:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM users")
                rows = cur.fetchall()

    Yields:
        psycopg2 connection object.
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def execute_query(query: str, params: dict = None, fetch: bool = True):
    """Execute a SQL query and optionally return results.

    Args:
        query: SQL query string with %(name)s placeholders.
        params: Dict of parameter values.
        fetch: If True, return rows as list of dicts. If False, return rowcount.

    Returns:
        List of dicts (if fetch=True) or affected row count (if fetch=False).
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params or {})
            if fetch:
                return [dict(row) for row in cur.fetchall()]
            return cur.rowcount
