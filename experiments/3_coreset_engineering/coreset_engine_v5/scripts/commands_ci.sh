#!/bin/bash

# ==============================================================================
# Coreset Engine CI/Foreground Playbook (commands_ci.sh)
# ==============================================================================
# Specialized for foreground execution (CI/CD or interactive troubleshooting).
# Automatically handles setup, validation, monitoring, pipeline run, 
# and post-run cleanup/reports.
#
# Steps:
#   1. System Setup & Prerequisites
#   2. AWS Authentication Check
#   3. Repository Setup
#   4. Dependency Sync (via UV)
#   5. Infrastructure Validation (validate_infra.sh)
#   6. Start Monitoring (monitor.sh)
#   7. Launch Pipeline (shard.sh)      - ALWAYS FOREGROUND
#   8. Post-Run Validation & Reports  - AUTOMATIC
#
# Usage:
#   Interactive / SSH: ./commands_ci.sh                               (tracks exit code)
#   Skip Repo Setup:   ./commands_ci.sh --skip-repo-setup             (cloning already done)
#
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status

# --- Parse flags --------------------------------------------------------------
SKIP_REPO_SETUP=false
SKIP_EBS_VALIDATION="${SKIP_EBS_VALIDATION:-true}"
SKIP_VALIDATION="${SKIP_VALIDATION:-false}"

for arg in "$@"; do
    case "$arg" in
        --skip-repo-setup) SKIP_REPO_SETUP=true ;;
        --skip-ebs) SKIP_EBS_VALIDATION=true ;;
        --skip-validation) SKIP_VALIDATION=true ;;
    esac
done

# Sanitize skip flags
SKIP_EBS_VALIDATION=$(echo "${SKIP_EBS_VALIDATION}" | sed "s/[”\"'“]//g")
SKIP_VALIDATION=$(echo "${SKIP_VALIDATION}" | sed "s/[”\"'“]//g")

# --- Configuration ------------------------------------------------------------
BRANCH_NAME="${BRANCH_NAME:-p3/feat/stage-wise-coreset-selection_v2}"
S3_BUCKET="${S3_BUCKET:-t2-datacurriculum-353}"
S3_INPUT_PATH="${S3_INPUT_PATH:-s3://${S3_BUCKET}/processed_dataset/curriculum_pyspark_output/source=C4/}"
NUM_SHARDS="${NUM_SHARDS:-8}"
STAGES="${STAGES:-1B}"
TOTAL_TOKENS="${TOTAL_TOKENS:-400000000000}"
BATCH_SIZE="${BATCH_SIZE:-80000}"
CHECKPOINT_EVERY_N_BATCHES="${CHECKPOINT_EVERY_N_BATCHES:-20}"
USED_CACHE_MAX_ENTRIES="${USED_CACHE_MAX_ENTRIES:-0}"
USED_CACHE_STATS_EVERY="${USED_CACHE_STATS_EVERY:-0}"
BATCH_PREFETCH_MODE="${BATCH_PREFETCH_MODE:-off}"
BATCH_PREFETCH_QUEUE_SIZE="${BATCH_PREFETCH_QUEUE_SIZE:-1}"
BATCH_PREFETCH_AUTO_MIN_BATCH_SIZE="${BATCH_PREFETCH_AUTO_MIN_BATCH_SIZE:-50000}"
BATCH_PREFETCH_AUTO_MAX_SHARD_CPU_RATIO="${BATCH_PREFETCH_AUTO_MAX_SHARD_CPU_RATIO:-1.0}"
BATCH_PREFETCH_AUTO_MIN_WAIT_MS="${BATCH_PREFETCH_AUTO_MIN_WAIT_MS:-2.0}"
BATCH_PREFETCH_AUTO_WARMUP_BATCHES="${BATCH_PREFETCH_AUTO_WARMUP_BATCHES:-5}"
RESUME="${RESUME:-false}" 

# Storage
NVME_MOUNT="${NVME_MOUNT:-/mnt/nvme}"

# 1. System Setup
echo "### [1/8] System Setup ###"
OS_TYPE=$(uname -s)
if [ "${OS_TYPE}" = "Linux" ]; then
    sudo apt update
    sudo apt install -y python3.12 python3.12-venv git python3-pip unzip dstat bc sysstat
    sudo sysctl -w vm.swappiness=0
    if ! aws --version &> /dev/null; then
        echo "Installing/Updating AWS CLI v2..."
        ARCH=$(uname -m)
        if [ "$ARCH" = "x86_64" ]; then
            AWS_ZIP_URL="https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip"
        elif [ "$ARCH" = "aarch64" ]; then
            AWS_ZIP_URL="https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip"
        else
            echo "[ERROR] Unsupported architecture for AWS CLI: $ARCH"
            exit 1
        fi
        curl "$AWS_ZIP_URL" -o "awscliv2.zip" && unzip -o awscliv2.zip
        if [ -d "/usr/local/aws-cli" ]; then
            sudo ./aws/install --update
        else
            sudo ./aws/install
        fi
        rm -rf aws awscliv2.zip
    fi
fi
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env" || source "$HOME/.cargo/env"
fi

# 2. AWS Auth
echo "### [2/8] AWS Authentication Check ###"
aws sts get-caller-identity > /dev/null

# 3. Repository Setup
echo "### [3/8] Repository Setup ###"
if [ "${SKIP_REPO_SETUP}" = "true" ]; then
    REPO_ROOT=$(pwd)
else
    [ ! -d "LLM" ] && git clone https://github.com/The-School-of-AI/LLM.git
    cd LLM
    REPO_ROOT=$(pwd)
    git fetch origin
    git checkout "${BRANCH_NAME}" || git checkout -b "${BRANCH_NAME}" "origin/${BRANCH_NAME}"
    git pull origin "${BRANCH_NAME}"
fi

# 4. Dependencies
EXPERIMENT_DIR="${REPO_ROOT}/experiments/3_coreset_engineering"
ENGINE_DIR="${EXPERIMENT_DIR}/coreset_engine_v5"
cd "${EXPERIMENT_DIR}"
[ ! -d ".venv" ] && uv venv .venv
export UV_PROJECT_ENVIRONMENT=$(pwd)/.venv
uv sync
source .venv/bin/activate

# 5. Infra Validation
echo "### [5/8] Infrastructure Validation ###"
VALIDATE_INFRA="${ENGINE_DIR}/scripts/validate_infra.sh"
# Export infra thresholds so sudo -E passes them to validate_infra.sh
export S3_BUCKET
[ -n "${ENABLE_NVME}" ] && export ENABLE_NVME
sudo -E bash "${VALIDATE_INFRA}"

# 6. Monitoring
echo "### [6/8] Start Monitoring ###"
MONITOR_SCRIPT="${ENGINE_DIR}/scripts/monitor.sh"
nohup bash "${MONITOR_SCRIPT}" > /dev/null 2>&1 &
MONITOR_PID=$!

# 7. Launch Pipeline
echo "### [7/8] Launching Pipeline (Foreground) ###"
cd "${REPO_ROOT}"
RESUME_FLAG=$([ "${RESUME}" = "true" ] && echo "--resume" || echo "")
PIPELINE_EXIT=0
bash experiments/3_coreset_engineering/coreset_engine_v5/shard.sh \
    --num-shards ${NUM_SHARDS} --stages "${STAGES}" \
    --input-path "${S3_INPUT_PATH}" --input-format jsonl \
    --total-tokens ${TOTAL_TOKENS} --batch-size ${BATCH_SIZE} \
    --checkpoint-every-n-batches ${CHECKPOINT_EVERY_N_BATCHES} \
    --used-cache-max-entries ${USED_CACHE_MAX_ENTRIES} --used-cache-stats-every ${USED_CACHE_STATS_EVERY} \
    --batch-prefetch-mode ${BATCH_PREFETCH_MODE} --batch-prefetch-queue-size ${BATCH_PREFETCH_QUEUE_SIZE} \
    --batch-prefetch-auto-min-batch-size ${BATCH_PREFETCH_AUTO_MIN_BATCH_SIZE} \
    --batch-prefetch-auto-max-shard-cpu-ratio ${BATCH_PREFETCH_AUTO_MAX_SHARD_CPU_RATIO} \
    --batch-prefetch-auto-min-wait-ms ${BATCH_PREFETCH_AUTO_MIN_WAIT_MS} \
    --batch-prefetch-auto-warmup-batches ${BATCH_PREFETCH_AUTO_WARMUP_BATCHES} \
    ${RESUME_FLAG} || PIPELINE_EXIT=$?

# 8. Post-Run
echo "### [8/8] Post-Run Validation & Reports ###"
[ -n "${MONITOR_PID}" ] && kill "${MONITOR_PID}" 2>/dev/null && sleep 2
bash "${ENGINE_DIR}/scripts/monitor_report.sh" "${LOG_DIR:-/mnt/nvme/logs}" || true
python "${ENGINE_DIR}/tools/validate_coreset_outputs.py" \
    --curriculum "${ENGINE_DIR}/config/curriculum.yaml" --stages ${STAGES} --format both || true

# 9. S3 Flush
if [ "${ENABLE_NVME}" = "true" ] && [ -d "${NVME_MOUNT}/coresets" ]; then
    echo "### [9/9] S3 Flush ###"
    aws s3 sync "${NVME_MOUNT}/coresets" "s3://${S3_BUCKET}/final_outputs/coresets/" --quiet
fi

echo "CI RUN COMPLETE (Exit: ${PIPELINE_EXIT})"
exit ${PIPELINE_EXIT}
