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
    curl \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    -o /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm /tmp/cuda-keyring.deb

RUN apt-get update && apt-get install -y \
    libcublas12

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
COPY .env .

RUN useradd -m app

RUN mkdir -p /home/app/models /home/app/temp \
    && chown -R app:app /home/app

USER app

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]