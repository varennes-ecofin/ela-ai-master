# ============================================================
# ELA AI — chat application (Chainlit) — multi-stage build
# ============================================================

# ---- builder: compile wheels / install into an isolated venv ----
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ src/
# CPU-only torch keeps the image ~2 GB smaller than the default CUDA build
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu .

# ---- runtime: slim image, no compiler, non-root user ----
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
COPY --chown=ela:ela apps/chat/ apps/chat/
COPY --chown=ela:ela .chainlit/ .chainlit/
COPY --chown=ela:ela public/ public/
COPY --chown=ela:ela chainlit.md ./
RUN mkdir -p data .files_ela && chown -R ela:ela /app

USER ela
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8000/ || exit 1

CMD ["chainlit", "run", "apps/chat/app.py", "--host", "0.0.0.0", "--port", "8000"]
