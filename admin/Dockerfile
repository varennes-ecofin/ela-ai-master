# admin/Dockerfile
# Build context = project root (to include ingest.py)

FROM python:3.11-slim

WORKDIR /app

# Dépendances système pour psycopg2, sentence-transformers, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 1. Install admin dependencies
COPY admin/requirements.txt /app/admin_requirements.txt
RUN pip install --no-cache-dir -r admin_requirements.txt

# 2. Install ingestion dependencies
COPY requirements.txt /app/main_requirements.txt
RUN pip install --no-cache-dir -r main_requirements.txt

# 3. Copy admin application
COPY admin/ /app/

# 4. Copy ingestion script (needed for rebuild from backoffice)
COPY ingest.py /app/ingest.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]