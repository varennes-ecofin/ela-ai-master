# ============================================================
# ELA AI — backoffice (Streamlit) — multi-stage build
# Build context = project root. Includes the `ela` package so that the
# ingestion can be launched from the UI (python -m ela.ingestion.pipeline).
# ============================================================

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu ".[admin]"

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/app/data/hf-cache \
    RERANK_CACHE_DIR=/app/data/flashrank

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 ela

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=ela:ela pyproject.toml ./
COPY --chown=ela:ela apps/admin/ apps/admin/
RUN mkdir -p data && chown -R ela:ela /app

USER ela
WORKDIR /app/apps/admin
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8501/admin/_stcore/health || curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
