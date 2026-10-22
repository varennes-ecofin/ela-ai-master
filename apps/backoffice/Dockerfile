# ============================================================
# ELA AI — backoffice (FastAPI + Jinja2 + HTMX) — multi-stage build
# Build context = project root. The image carries the whole `ela` package
# (torch CPU included) so that ingestion jobs can run in-process (lot 5.3).
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
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu ".[backoffice]"

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
COPY --chown=ela:ela apps/backoffice/ apps/backoffice/
COPY --chown=ela:ela public/ public/
RUN mkdir -p data && chown -R ela:ela /app

USER ela
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8080${BACKOFFICE_ROOT_PATH:-/backoffice}/healthz || exit 1

CMD ["uvicorn", "backoffice.main:create_app", "--factory", "--app-dir", "apps", \
     "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--forwarded-allow-ips", "*"]
