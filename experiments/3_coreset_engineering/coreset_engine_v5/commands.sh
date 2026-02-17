#!/bin/bash

# ==============================================================================
# Coreset Engine Deployment Script
# ==============================================================================
# This script automates the setup and execution of the coreset pipeline.
# It uses 'uv' for fast, reliable dependency management.
#
# Usage:
#   Manual EC2:  ./commands.sh            (runs full setup + pipeline)
#   GitHub CI:   Automatically detected via $CI env var (skips EC2-only steps)
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status

# --- Configuration (UPDATE THESE) ---------------------------------------------
# These can be overridden by environment variables (e.g. for CI/CD)
BRANCH_NAME="${BRANCH_NAME:-p3/feat/stage-wise-coreset-selection_v2}"
S3_BUCKET="${S3_BUCKET:-<container-name>}" 
S3_INPUT_PATH="${S3_INPUT_PATH:-s3://${S3_BUCKET}/processed_dataset/curriculum_pyspark_output/}"
NUM_SHARDS="${NUM_SHARDS:-8}"
STAGES="${STAGES:-1B}"
TOTAL_TOKENS="${TOTAL_TOKENS:-4523096944}"
RESUME="${RESUME:-true}" 
# ------------------------------------------------------------------------------

# ==============================================================================
# 1. System Setup & Prerequisites (EC2 only)
# ==============================================================================
if [ "${CI}" != "true" ]; then
    echo "### [1/5] System Setup & Prerequisites ###"
    sudo apt update
    sudo apt install -y python3.12 python3.12-venv git python3-pip unzip

    # Install uv if not present
    if ! command -v uv &> /dev/null; then
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.cargo/env
    fi
else
    echo "### [1/5] System Setup (SKIPPED - CI environment detected) ###"
fi

# ==============================================================================
# 2. AWS Authentication Check
# ==============================================================================
echo "### [2/5] AWS Authentication Check ###"
if ! aws sts get-caller-identity &> /dev/null; then
    echo "WARNING: AWS credentials not found."
    echo "Please run 'aws configure' or attach an IAM Role to this instance."
    echo "Note: The script will proceed but S3-dependent tasks will fail later."
fi

# ==============================================================================
# 3. Repository Setup (EC2 only)
# ==============================================================================
if [ "${CI}" != "true" ]; then
    echo "### [3/5] Repository Setup ###"
    # Detect if we are already inside the LLM repository
    if [ -d ".git" ] && [[ $(git remote get-url origin 2>/dev/null) == *"LLM.git"* ]]; then
        echo "Already inside LLM repository."
        REPO_ROOT=$(pwd)
    else
        if [ ! -d "LLM" ]; then
            echo "Cloning repository..."
            git clone https://github.com/The-School-of-AI/LLM.git
        fi
        cd LLM
        REPO_ROOT=$(pwd)
    fi

    # Fetch and checkout branch
    git fetch origin
    if git show-ref --verify --quiet "refs/heads/${BRANCH_NAME}"; then
        git checkout "${BRANCH_NAME}"
    else
        git checkout -b "${BRANCH_NAME}" "origin/${BRANCH_NAME}"
    fi
    git pull origin "${BRANCH_NAME}"
else
    echo "### [3/5] Repository Setup (SKIPPED - CI uses actions/checkout) ###"
    REPO_ROOT=$(pwd)
fi

# ==============================================================================
# 4. Dependency Sync (via UV)
# ==============================================================================
echo "### [4/5] Dependency Sync (via UV) ###"
cd "${REPO_ROOT}/experiments/3_coreset_engineering/"
uv venv .venv
export UV_PROJECT_ENVIRONMENT=$(pwd)/.venv
uv sync

# ==============================================================================
# 5. Launch Pipeline
# ==============================================================================
echo "### [5/5] Launching Pipeline ###"
cd "${REPO_ROOT}"

RESUME_FLAG=""
if [ "${RESUME}" = "true" ]; then
    RESUME_FLAG="--resume"
fi

if [ "${CI}" != "true" ]; then
    # EC2: Run in background with nohup for persistence
    echo "Starting shard.sh in background via nohup..."
    nohup bash experiments/3_coreset_engineering/coreset_engine_v5/shard.sh \
      --num-shards ${NUM_SHARDS} \
      --stages "${STAGES}" \
      --input-path "${S3_INPUT_PATH}" \
      --input-format jsonl \
      --total-tokens ${TOTAL_TOKENS} \
      ${RESUME_FLAG} \
      > shard_run.log 2>&1 &

    echo "-----------------------------------------------------------------------"
    echo "DEPLOYMENT COMPLETE"
    echo "Monitor logs via: tail -f ${REPO_ROOT}/shard_run.log"
    echo "Check process via: ps aux | grep shard.sh"
    echo "-----------------------------------------------------------------------"
else
    # CI: Run in foreground so GitHub Actions can track exit code
    echo "Running shard.sh in foreground (CI mode)..."
    bash experiments/3_coreset_engineering/coreset_engine_v5/shard.sh \
      --num-shards ${NUM_SHARDS} \
      --stages "${STAGES}" \
      --input-path "${S3_INPUT_PATH}" \
      --input-format jsonl \
      --total-tokens ${TOTAL_TOKENS} \
      ${RESUME_FLAG}
fi
