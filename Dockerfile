FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/home/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    gnupg \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/app

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.4.1 \
    torchaudio==2.4.1
    
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY migrations ./migrations
COPY .env .
COPY alembic.ini .

RUN useradd -m app \
    && mkdir -p /home/app/ml_models /home/app/temp \
    && chown -R app:app /home/app

USER app

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]