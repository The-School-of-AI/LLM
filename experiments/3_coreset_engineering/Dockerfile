# Multi-stage build for coreset-engineering
# Ensures reproducible, deterministic builds with locked dependencies

ARG PYTHON_VERSION=3.12

# Builder stage
FROM python:${PYTHON_VERSION}-slim as builder

WORKDIR /app

# Environment: Ensure deterministic builds
ENV PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install uv and build package with locked dependencies
# --frozen ensures uv.lock file is respected (reproducible)
# --no-dev excludes development dependencies
RUN pip install uv && \
    cd experiments/3_coreset_engineering && \
    uv sync --frozen --no-dev && \
    uv pip install -e .

# Runtime stage
FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/experiments/3_coreset_engineering/.venv /app/.venv

# Copy application code
COPY --from=builder /app/experiments/3_coreset_engineering /app/coreset_engineering

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app/coreset_engineering

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "coreset_engine.api.server"]
