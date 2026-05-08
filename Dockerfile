# ============================================================
# Enterprise MLOps Pipeline — Dockerfile
# ============================================================
# Multi-stage optimized image based on python:3.11-slim.
# Produces a reproducible containerized DVC + sklearn env.
#
# Build:  docker build -t mlops-pipeline .
# Run:    docker run --rm -it mlops-pipeline bash
# ============================================================

# ── Base Stage ─────────────────────────────────────────────
FROM python:3.11-slim AS base

# Metadata labels
LABEL org.opencontainers.image.title="Enterprise MLOps Pipeline"
LABEL org.opencontainers.image.description="DVC-orchestrated modular ML pipeline"
LABEL org.opencontainers.image.version="1.0.0"

# Avoid interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive
# Ensure Python output is not buffered (important for logging)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Place pipenv/pip installed binaries on PATH
ENV PATH="/root/.local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# ── Dependency Stage ───────────────────────────────────────
FROM base AS deps

# Copy only requirements first to leverage Docker layer cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application Stage ──────────────────────────────────────
FROM deps AS app

# Copy project source code
COPY . .

# Create necessary output directories
RUN mkdir -p data models metrics logs reports

# Initialize DVC (no remote configured by default)
RUN git config --global user.email "mlops@pipeline.local" && \
    git config --global user.name "MLOps Pipeline" && \
    git init && \
    dvc init --no-scm 2>/dev/null || true

# Expose no ports (batch ML workload, not a web service)
# Default command: run the full DVC pipeline
CMD ["dvc", "repro"]
