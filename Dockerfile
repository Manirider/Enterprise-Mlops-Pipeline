FROM python:3.11-slim AS base

LABEL org.opencontainers.image.title="Enterprise MLOps Pipeline"
LABEL org.opencontainers.image.description="DVC-orchestrated modular ML pipeline"
LABEL org.opencontainers.image.version="1.0.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/root/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

FROM base AS deps

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

FROM deps AS app

COPY . .

RUN mkdir -p data models metrics logs reports

RUN git config --global user.email "mlops@pipeline.local" && \
    git config --global user.name "MLOps Pipeline" && \
    git init && \
    dvc init --no-scm 2>/dev/null || true

CMD ["dvc", "repro"]
