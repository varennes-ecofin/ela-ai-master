# ELA AI — chat application (Chainlit)
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first (layer cache), then the package
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir .

# Application files (config and assets are read from the working directory)
COPY apps/chat/ apps/chat/
COPY .chainlit/ .chainlit/
COPY public/ public/
COPY chainlit.md ./

EXPOSE 8000

CMD ["chainlit", "run", "apps/chat/app.py", "--host", "0.0.0.0", "--port", "8000"]
