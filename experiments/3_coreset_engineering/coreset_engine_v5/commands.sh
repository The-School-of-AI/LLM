#!/bin/bash

# ==============================================================================
# Coreset Engine Deployment Script
# ==============================================================================
# This script automates the setup and execution of the coreset pipeline.
# It uses 'uv' for fast, reliable dependency management.
#
# Usage:
#   Manual EC2:       ./commands.sh                                  (full setup + pipeline in background)
#   Dry Run:          ./commands.sh --dry-run                        (validates setup, no pipeline)
#   CI (self-hosted): ./commands.sh --foreground --skip-repo-setup   (checkout already done)
#   CI (SSH):         ./commands.sh --foreground                     (clones repo on EC2)
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status

# --- Parse flags --------------------------------------------------------------
DRY_RUN=false
FOREGROUND=false
SKIP_REPO_SETUP=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --foreground) FOREGROUND=true ;;
        --skip-repo-setup) SKIP_REPO_SETUP=true ;;
    esac
done

if [ "${DRY_RUN}" = "true" ]; then
    echo "============================================"
    echo "  DRY RUN MODE — No pipeline will be launched"
    echo "============================================"
fi

# --- Configuration (UPDATE THESE) ---------------------------------------------
# These can be overridden by environment variables (e.g. for CI/CD)
BRANCH_NAME="${BRANCH_NAME:-p3/feat/stage-wise-coreset-selection_v2}"
S3_BUCKET="${S3_BUCKET:?ERROR: S3_BUCKET is not set. Export it before running: export S3_BUCKET=your-bucket-name}"
S3_INPUT_PATH="${S3_INPUT_PATH:-s3://${S3_BUCKET}/processed_dataset/curriculum_pyspark_output/}"
NUM_SHARDS="${NUM_SHARDS:-8}"
STAGES="${STAGES:-1B}"
TOTAL_TOKENS="${TOTAL_TOKENS:-4523096944}"
BATCH_SIZE="${BATCH_SIZE:-80000}"
CHECKPOINT_EVERY_N_BATCHES="${CHECKPOINT_EVERY_N_BATCHES:-3}"
USED_CACHE_MAX_ENTRIES="${USED_CACHE_MAX_ENTRIES:-0}"
USED_CACHE_STATS_EVERY="${USED_CACHE_STATS_EVERY:-0}"
BATCH_PREFETCH_MODE="${BATCH_PREFETCH_MODE:-off}"
BATCH_PREFETCH_QUEUE_SIZE="${BATCH_PREFETCH_QUEUE_SIZE:-1}"
BATCH_PREFETCH_AUTO_MIN_BATCH_SIZE="${BATCH_PREFETCH_AUTO_MIN_BATCH_SIZE:-50000}"
BATCH_PREFETCH_AUTO_MAX_SHARD_CPU_RATIO="${BATCH_PREFETCH_AUTO_MAX_SHARD_CPU_RATIO:-1.0}"
BATCH_PREFETCH_AUTO_MIN_WAIT_MS="${BATCH_PREFETCH_AUTO_MIN_WAIT_MS:-2.0}"
BATCH_PREFETCH_AUTO_WARMUP_BATCHES="${BATCH_PREFETCH_AUTO_WARMUP_BATCHES:-5}"
RESUME="${RESUME:-false}" 
# ------------------------------------------------------------------------------

# ==============================================================================
# 1. System Setup & Prerequisites
# ==============================================================================
echo "### [1/5] System Setup & Prerequisites ###"
OS_TYPE=$(uname -s)

if [ "${DRY_RUN}" = "true" ]; then
    echo "[DRY RUN] OS detected: ${OS_TYPE}"
    if [ "${OS_TYPE}" = "Linux" ]; then
        echo "[DRY RUN] Would run: sudo apt update && install python3.12, git, etc."
    else
        echo "[DRY RUN] Non-Linux OS — skipping apt packages."
    fi
    if command -v uv &> /dev/null; then
        echo "[OK] uv is installed: $(uv --version)"
    else
        echo "[WARN] uv is NOT installed. Would install via: curl -LsSf https://astral.sh/uv/install.sh | sh"
    fi
else
    if [ "${OS_TYPE}" = "Linux" ]; then
        sudo apt update
        sudo apt install -y python3.12 python3.12-venv git python3-pip unzip
    else
        echo "[SKIP] Non-Linux OS (${OS_TYPE}) — skipping apt packages."
    fi

    if ! command -v uv &> /dev/null; then
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.cargo/env
    fi
fi

# ==============================================================================
# 2. AWS Authentication Check
# ==============================================================================
echo "### [2/5] AWS Authentication Check ###"
if aws sts get-caller-identity &> /dev/null; then
    echo "[OK] AWS credentials found."
else
    echo "[WARN] AWS credentials not found."
    echo "Please run 'aws configure' or attach an IAM Role to this instance."
    if [ "${DRY_RUN}" != "true" ]; then
        echo "Note: The script will proceed but S3-dependent tasks will fail later."
    fi
fi

# ==============================================================================
# 3. Repository Setup
# ==============================================================================
echo "### [3/5] Repository Setup ###"
if [ "${SKIP_REPO_SETUP}" = "true" ]; then
    echo "[SKIP] Git clone/checkout skipped (--skip-repo-setup). Validating working directory..."
    REPO_ROOT=$(pwd)

    # Validate: must be inside a git repo
    if [ ! -d ".git" ]; then
        echo "[ERROR] --skip-repo-setup requires running from inside a git repository."
        echo "        Current directory: $(pwd)"
        exit 1
    fi

    # Validate: remote must point to the expected repo
    REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "")
    if [[ "${REMOTE_URL}" != *"LLM.git"* ]]; then
        echo "[ERROR] Git remote 'origin' does not point to the expected LLM repository."
        echo "        Got: ${REMOTE_URL}"
        exit 1
    fi

    # Validate: critical pipeline files exist
    SHARD_SCRIPT="experiments/3_coreset_engineering/coreset_engine_v5/shard.sh"
    PYPROJECT="experiments/3_coreset_engineering/pyproject.toml"
    if [ ! -f "${SHARD_SCRIPT}" ] || [ ! -f "${PYPROJECT}" ]; then
        echo "[ERROR] Critical pipeline files missing from working directory:"
        [ ! -f "${SHARD_SCRIPT}" ] && echo "        Missing: ${SHARD_SCRIPT}"
        [ ! -f "${PYPROJECT}" ] && echo "        Missing: ${PYPROJECT}"
        exit 1
    fi

    echo "[OK] Working directory validated: ${REPO_ROOT}"
elif [ "${DRY_RUN}" = "true" ]; then
    if [ -d ".git" ] && [[ $(git remote get-url origin 2>/dev/null) == *"LLM.git"* ]]; then
        echo "[OK] Already inside LLM repository at $(pwd)"
        REPO_ROOT=$(pwd)
    else
        echo "[INFO] Would clone repo and checkout branch: ${BRANCH_NAME}"
        echo "[DRY RUN] Skipping clone. Using current directory."
        REPO_ROOT=$(pwd)
    fi
else
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

    git fetch origin
    if git show-ref --verify --quiet "refs/heads/${BRANCH_NAME}"; then
        git checkout "${BRANCH_NAME}"
    else
        git checkout -b "${BRANCH_NAME}" "origin/${BRANCH_NAME}"
    fi
    git pull origin "${BRANCH_NAME}"
fi

# ==============================================================================
# 4. Dependency Sync (via UV)
# ==============================================================================
echo "### [4/5] Dependency Sync (via UV) ###"
EXPERIMENT_DIR="${REPO_ROOT}/experiments/3_coreset_engineering"

if [ -d "${EXPERIMENT_DIR}" ]; then
    cd "${EXPERIMENT_DIR}"
    if [ "${DRY_RUN}" = "true" ]; then
        echo "[DRY RUN] Would create .venv and run: uv sync"
        if [ -f "pyproject.toml" ]; then
            echo "[OK] pyproject.toml found at ${EXPERIMENT_DIR}/pyproject.toml"
        else
            echo "[ERROR] pyproject.toml NOT found at ${EXPERIMENT_DIR}/"
        fi
    else
        if [ ! -d ".venv" ]; then
            uv venv .venv
        fi
        export UV_PROJECT_ENVIRONMENT=$(pwd)/.venv
        uv sync
    fi
else
    echo "[ERROR] Experiment directory not found: ${EXPERIMENT_DIR}"
    exit 1
fi

# ==============================================================================
# 5. Launch Pipeline
# ==============================================================================
echo "### [5/5] Launching Pipeline ###"
cd "${REPO_ROOT}"

RESUME_FLAG=""
if [ "${RESUME}" = "true" ]; then
    RESUME_FLAG="--resume"
fi

if [ "${DRY_RUN}" = "true" ]; then
        echo ""
        echo "=========================================="
        echo "  DRY RUN SUMMARY — Validation Complete"
        echo "=========================================="
        echo "  Branch:       ${BRANCH_NAME}"
        echo "  S3 Input:     ${S3_INPUT_PATH}"
        echo "  Num Shards:   ${NUM_SHARDS}"
        echo "  Stages:       ${STAGES}"
        echo "  Total Tokens: ${TOTAL_TOKENS}"
        echo "  Batch Size:   ${BATCH_SIZE}"
        echo "  Ckpt Every N: ${CHECKPOINT_EVERY_N_BATCHES}"
        echo "  Used Cache:   max=${USED_CACHE_MAX_ENTRIES} stats_every=${USED_CACHE_STATS_EVERY}"
        echo "  Prefetch:     mode=${BATCH_PREFETCH_MODE} queue=${BATCH_PREFETCH_QUEUE_SIZE} auto_min_batch=${BATCH_PREFETCH_AUTO_MIN_BATCH_SIZE} auto_max_ratio=${BATCH_PREFETCH_AUTO_MAX_SHARD_CPU_RATIO} auto_min_wait_ms=${BATCH_PREFETCH_AUTO_MIN_WAIT_MS} auto_warmup=${BATCH_PREFETCH_AUTO_WARMUP_BATCHES}"
        echo "  Resume:       ${RESUME}"
        echo "  Foreground:   ${FOREGROUND}"
        echo ""
        echo "  Would execute:"
        echo "    bash experiments/3_coreset_engineering/coreset_engine_v5/shard.sh"
        echo "      --num-shards ${NUM_SHARDS} --stages \"${STAGES}\""
        echo "      --input-path \"${S3_INPUT_PATH}\" --total-tokens ${TOTAL_TOKENS} --batch-size ${BATCH_SIZE}"
        echo "      --checkpoint-every-n-batches ${CHECKPOINT_EVERY_N_BATCHES} ${RESUME_FLAG}"
        echo "      --used-cache-max-entries ${USED_CACHE_MAX_ENTRIES} --used-cache-stats-every ${USED_CACHE_STATS_EVERY}"
        echo "      --batch-prefetch-mode ${BATCH_PREFETCH_MODE}"
        echo "      --batch-prefetch-queue-size ${BATCH_PREFETCH_QUEUE_SIZE}"
        echo "      --batch-prefetch-auto-min-batch-size ${BATCH_PREFETCH_AUTO_MIN_BATCH_SIZE}"
        echo "      --batch-prefetch-auto-max-shard-cpu-ratio ${BATCH_PREFETCH_AUTO_MAX_SHARD_CPU_RATIO}"
        echo "      --batch-prefetch-auto-min-wait-ms ${BATCH_PREFETCH_AUTO_MIN_WAIT_MS}"
        echo "      --batch-prefetch-auto-warmup-batches ${BATCH_PREFETCH_AUTO_WARMUP_BATCHES}"
        echo "=========================================="
        exit 0
fi

if [ "${FOREGROUND}" = "true" ]; then
        # Foreground: Used by CI/SSH so exit code is tracked
        echo "Running shard.sh in foreground..."
        bash experiments/3_coreset_engineering/coreset_engine_v5/shard.sh \
            --num-shards ${NUM_SHARDS} \
            --stages "${STAGES}" \
            --input-path "${S3_INPUT_PATH}" \
            --input-format jsonl \
            --total-tokens ${TOTAL_TOKENS} \
            --batch-size ${BATCH_SIZE} \
            --checkpoint-every-n-batches ${CHECKPOINT_EVERY_N_BATCHES} \
            --used-cache-max-entries ${USED_CACHE_MAX_ENTRIES} \
            --used-cache-stats-every ${USED_CACHE_STATS_EVERY} \
            --batch-prefetch-mode ${BATCH_PREFETCH_MODE} \
            --batch-prefetch-queue-size ${BATCH_PREFETCH_QUEUE_SIZE} \
            --batch-prefetch-auto-min-batch-size ${BATCH_PREFETCH_AUTO_MIN_BATCH_SIZE} \
            --batch-prefetch-auto-max-shard-cpu-ratio ${BATCH_PREFETCH_AUTO_MAX_SHARD_CPU_RATIO} \
            --batch-prefetch-auto-min-wait-ms ${BATCH_PREFETCH_AUTO_MIN_WAIT_MS} \
            --batch-prefetch-auto-warmup-batches ${BATCH_PREFETCH_AUTO_WARMUP_BATCHES} \
            ${RESUME_FLAG}
else
        # Background: Used for manual EC2 runs with nohup for SSH disconnect safety
        echo "Starting shard.sh in background via nohup..."
        nohup bash experiments/3_coreset_engineering/coreset_engine_v5/shard.sh \
            --num-shards ${NUM_SHARDS} \
            --stages "${STAGES}" \
            --input-path "${S3_INPUT_PATH}" \
            --input-format jsonl \
            --total-tokens ${TOTAL_TOKENS} \
            --batch-size ${BATCH_SIZE} \
            --checkpoint-every-n-batches ${CHECKPOINT_EVERY_N_BATCHES} \
            --used-cache-max-entries ${USED_CACHE_MAX_ENTRIES} \
            --used-cache-stats-every ${USED_CACHE_STATS_EVERY} \
            --batch-prefetch-mode ${BATCH_PREFETCH_MODE} \
            --batch-prefetch-queue-size ${BATCH_PREFETCH_QUEUE_SIZE} \
            --batch-prefetch-auto-min-batch-size ${BATCH_PREFETCH_AUTO_MIN_BATCH_SIZE} \
            --batch-prefetch-auto-max-shard-cpu-ratio ${BATCH_PREFETCH_AUTO_MAX_SHARD_CPU_RATIO} \
            --batch-prefetch-auto-min-wait-ms ${BATCH_PREFETCH_AUTO_MIN_WAIT_MS} \
            --batch-prefetch-auto-warmup-batches ${BATCH_PREFETCH_AUTO_WARMUP_BATCHES} \
            ${RESUME_FLAG} \
            > shard_run.log 2>&1 &

        echo "-----------------------------------------------------------------------"
        echo "DEPLOYMENT COMPLETE"
        echo "Monitor logs via: tail -f ${REPO_ROOT}/shard_run.log"
        echo "Check process via: ps aux | grep shard.sh"
        echo "-----------------------------------------------------------------------"
fi
