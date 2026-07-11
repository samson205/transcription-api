FROM ubuntu:24.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/home/app

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    ffmpeg \
    git \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /home/app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

RUN pip install \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.4.1 \
    torchaudio==2.4.1

RUN pip install -r requirements.txt

COPY api ./api
COPY migrations migrations
COPY .env .
COPY alembic.ini .

RUN useradd -m app

RUN mkdir -p /home/app/ml_models /home/app/temp \
    && chown -R app:app /home/app

USER app

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]