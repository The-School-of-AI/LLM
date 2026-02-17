#!/bin/bash

# ==============================================================================
# Coreset Engine EC2 Deployment Script
# ==============================================================================
# This script automates the setup and execution of the coreset pipeline.
# It uses 'uv' for fast, reliable dependency management.
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status

# --- Configuration (UPDATE THESE) ---------------------------------------------
BRANCH_NAME="p3/feat/stage-wise-coreset-selection_v2"
S3_BUCKET="<container-name>" # Replace with your bucket name
S3_INPUT_PATH="s3://${S3_BUCKET}/processed_dataset/curriculum_pyspark_output/"
NUM_SHARDS=8
STAGES="1B"
TOTAL_TOKENS=4523096944
# ------------------------------------------------------------------------------

echo "### [1/5] System Setup & Prerequisites ###"
sudo apt update
sudo apt install -y python3.12 python3.12-venv git python3-pip unzip

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

echo "### [2/5] AWS Authentication Check ###"
if ! aws sts get-caller-identity &> /dev/null; then
    echo "WARNING: AWS credentials not found."
    echo "Please run 'aws configure' or attach an IAM Role to this instance."
    echo "Note: The script will proceed but S3-dependent tasks will fail later."
fi

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

echo "### [4/5] Dependency Sync (via UV) ###"
cd "${REPO_ROOT}/experiments/3_coreset_engineering/"
uv venv .venv
export UV_PROJECT_ENVIRONMENT=$(pwd)/.venv
uv sync

echo "### [5/5] Launching Pipeline ###"
# Ensure we are at REPO_ROOT for shard.sh consistency
cd "${REPO_ROOT}"

# Optional: Run inside tmux for better persistence management
# tmux new-session -d -s coreset "bash experiments/3_coreset_engineering/coreset_engine_v5/shard.sh ..."

echo "Starting shard.sh in background via nohup..."
nohup bash experiments/3_coreset_engineering/coreset_engine_v5/shard.sh \
  --num-shards ${NUM_SHARDS} \
  --stages "${STAGES}" \
  --input-path "${S3_INPUT_PATH}" \
  --input-format jsonl \
  --total-tokens ${TOTAL_TOKENS} \
  --resume \
  > shard_run.log 2>&1 &

echo "-----------------------------------------------------------------------"
echo "DEPLOYMENT COMPLETE"
echo "Monitor logs via: tail -f ${REPO_ROOT}/shard_run.log"
echo "Check process via: ps aux | grep shard.sh"
echo "-----------------------------------------------------------------------"
