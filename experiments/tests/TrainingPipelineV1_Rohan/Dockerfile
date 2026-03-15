FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# System deps
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
    python3.12-venv python3-pip python3-dev \
    git curl awscli && \
    rm -rf /var/lib/apt/lists/*

# uv for fast Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Working directory
WORKDIR /workspace

# Copy code + lockfiles
COPY code/ code/
COPY configs/ configs/
COPY deepspeed/ deepspeed/
COPY scripts/ scripts/
COPY run.sh .
COPY requirements-pinned.txt .
COPY pyproject.toml .
COPY uv.lock .

# Install Python env from lockfile
ENV CUDA_HOME=/usr/local/cuda
RUN uv sync

# Environment
ENV PYTHON_BIN=/workspace/.venv/bin/python3
ENV DEEPSPEED_BIN=/workspace/.venv/bin/deepspeed
ENV PATH="/workspace/.venv/bin:$PATH"
ENV TORCHDYNAMO_DISABLE=1

# Verify at build time
RUN . .venv/bin/activate && python3 -c "\
import torch; print(f'torch={torch.__version__}'); \
import triton; print(f'triton={triton.__version__}'); \
import deepspeed; print(f'deepspeed={deepspeed.__version__}'); \
import fla; print(f'fla={fla.__version__}')"

# Data and results mount points
VOLUME ["/data", "/results"]

# Default: run training
CMD ["bash", "run.sh"]
