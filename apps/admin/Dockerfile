# ELA AI — backoffice (Streamlit). Build context = project root.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# The `ela` package (with its ingestion dependencies) + admin extras
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir ".[admin]"

COPY apps/admin/ apps/admin/

WORKDIR /app/apps/admin
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
