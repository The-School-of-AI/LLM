# =============================================================================
# Coreset Selection Engine - Docker Image
# =============================================================================
# Packages the coreset engine for AWS Batch Array Job deployment.
# Each container in the array runs one shard (--shard-id = AWS_BATCH_JOB_ARRAY_INDEX).
# =============================================================================

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached separately from source code)
COPY requirements.txt .

# torch is heavy; install CPU-only variant to keep image lean for CPU Batch jobs.
# If GPU instances are used, remove the --index-url override and use the default torch.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and configs
COPY src/ src/
COPY config/ config/
COPY coreset_builder.py .
COPY tools/ tools/

# Copy the Batch entrypoint script and make it executable
COPY batch_entrypoint.sh .
RUN chmod +x batch_entrypoint.sh

# Default entrypoint for AWS Batch
ENTRYPOINT ["/bin/bash", "batch_entrypoint.sh"]
