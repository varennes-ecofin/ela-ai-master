"""ELA AI — backoffice (FastAPI + Jinja2 + HTMX).

Run from the project root::

    python apps/backoffice/main.py --reload      # http://localhost:8502/backoffice/

or, in Docker, ``uvicorn backoffice.main:create_app --factory --app-dir apps``.
The application only contains views: accounts, audit, figures and jobs live
in the ``ela`` package.
"""
