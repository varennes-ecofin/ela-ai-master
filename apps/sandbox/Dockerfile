# ============================================================
# ELA AI — sandbox d'exécution des scripts de l'Atelier Code
# Python scientifique, sans réseau (réseau interne seulement), non-root.
# ============================================================
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg

RUN pip install --no-cache-dir \
      numpy==2.2.6 pandas==2.3.3 scipy==1.15.3 statsmodels==0.14.5 matplotlib==3.10.6 arch==7.2.0 \
    && useradd --create-home --uid 1000 runner

WORKDIR /sandbox
COPY runner.py server.py ./
USER runner

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health')"
CMD ["python", "server.py"]
