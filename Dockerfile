FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app \
    MODELS_DIR=/app/models

WORKDIR /app

# Só dependências de runtime (camada separada para aproveitar o cache)
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY api/ api/

# Treina na build com seed fixa: a imagem é reproduzível e sobe sem volume.
# Os dados são sintéticos (src/anomalias/data.py).
RUN python -m anomalias.train --n-samples 5000 --contamination 0.05 --seed 42 \
        --models-dir "$MODELS_DIR"

RUN useradd --system --no-create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

# 1 worker: as janelas de drift e de anomaly_ratio vivem na memória do processo
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
