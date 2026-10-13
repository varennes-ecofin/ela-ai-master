"""Database access for ELA AI (PostgreSQL, SQLAlchemy).

* ``ela.db.engine`` — one async engine (connection pool) per process, plus
  a sync engine for command-line tools.
* ``ela.db.users`` — user accounts: authentication, profiles, quotas.
* ``ela.db.threads`` — small helpers on Chainlit's ``threads`` table.
"""
