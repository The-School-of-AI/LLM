#!/bin/bash

# ==============================================================================
# Coreset Engine Production Run Playbook
# ==============================================================================
# Full 8-step production playbook: setup, validation, monitoring, pipeline,
# and post-run verification.
#
# Prerequisites (Step 0):
#   An AWS Admin must first run the EMR Serverless job: emr/T3_final_emr_serverless_stats.py
#   Once the EMR job completes, it generates chunked data files and source-wise stats in CSV format.
#   These stats must be aggregated to get TOTAL_TOKENS and passed to shard.sh as a parameter.
#   - To aggregate TOTAL_TOKENS: run tools/estimate_total_tokens.py
#   - For distribution analysis on bands/domains data: use notebooks/distribution_plots_notebook_extended.ipynb
#     (This notebook also creates an aggregate CSV `combined_source_distribution.csv` that provides TOTAL_TOKENS)
#
# Steps:
#   1. System Setup & Prerequisites
#   2. AWS Authentication Check
#   3. Repository Setup
#   4. Dependency Sync (via UV)
#   5. Infrastructure Validation (validate_infra.sh)
#   6. Start Monitoring (monitor.sh)
#   7. Launch Pipeline (shard.sh)
#   8. Post-Run Validation & Reports
#
# Usage:
#   Manual EC2:       ./commands.sh                                  (full setup + pipeline in background)
#   Dry Run:          ./commands.sh --dry-run                        (validates setup, no pipeline)
#   CI / Foreground:  ./commands_ci.sh                               (standard for CI/SSH)
#
# Examples:
#
#   # Production run on c7gd.16xlarge (defaults):
#   export S3_BUCKET="my-bucket"
#   sudo bash experiments/3_coreset_engineering/coreset_engine_v5/scripts/setup_nvme.sh # If using NVMe
#   ./commands.sh
#
#   # Dry run — preview all config without executing:
#   export S3_BUCKET="my-bucket"
#   ./commands.sh --dry-run
#
#   # Full pipeline run on a smaller instance:
#   export S3_BUCKET="my-bucket"
#   export ENABLE_NVME=false
#   ./commands.sh
#
#   # For foreground/CI execution, use: ./commands_ci.sh
#
#   # Override pipeline parameters:
#   export S3_BUCKET="my-bucket"
#   export NUM_SHARDS=4
#   export STAGES="1B 3B"
#   export BATCH_SIZE=50000
#   export RESUME=true
#   ./commands.sh
#   # Estimate TOTAL_TOKENS from post-dedup stats/ CSVs on EC2:
#   #   Option 1: Python tool (sums total_tokens from all source CSVs)
#   python3 experiments/3_coreset_engineering/coreset_engine_v5/tools/estimate_total_tokens.py \
#       --input-path "path/to/stats" --input-format csv --quiet
#   #   Option 2: Quick awk one-liner across all source CSVs
#   awk -F',' 'NR>1{s+=$COL}END{print s}' /mnt/nvme/stats/*.csv
#   # Then export before running:
#   export TOTAL_TOKENS=4523096944
#
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status

# --- Parse flags --------------------------------------------------------------
DRY_RUN=false
SKIP_REPO_SETUP=false
SKIP_EBS_VALIDATION="${SKIP_EBS_VALIDATION:-true}"
SKIP_VALIDATION="${SKIP_VALIDATION:-false}"

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --skip-repo-setup) SKIP_REPO_SETUP=true ;;
        --skip-ebs) SKIP_EBS_VALIDATION=true ;;
        --skip-validation) SKIP_VALIDATION=true ;;
    esac
done

# Sanitize skip flags (remove potential smart/standard quotes from environment exports)
SKIP_EBS_VALIDATION=$(echo "${SKIP_EBS_VALIDATION}" | sed "s/[”\"'“]//g")
SKIP_VALIDATION=$(echo "${SKIP_VALIDATION}" | sed "s/[”\"'“]//g")

if [ "${DRY_RUN}" = "true" ]; then
    echo "============================================"
    echo "  DRY RUN MODE — No pipeline will be launched"
    echo "============================================"
fi

# --- Configuration (UPDATE THESE) ---------------------------------------------
# These can be overridden by environment variables (e.g. for CI/CD)
BRANCH_NAME="${BRANCH_NAME:-p3/feat/stage-wise-coreset-selection_v2}"
S3_BUCKET="${S3_BUCKET:-t2-datacurriculum-353}"
S3_INPUT_PATH="${S3_INPUT_PATH:-s3://${S3_BUCKET}/processed_dataset/curriculum_pyspark_output/}"
S3_PREFIX="${S3_PREFIX:-processed_dataset/curriculum_pyspark_output/source=C4/}"
NUM_SHARDS="${NUM_SHARDS:-8}"
STAGES="${STAGES:-1B}"
TOTAL_TOKENS="${TOTAL_TOKENS:-400000000000}"
# ^ Get TOTAL_TOKENS from post-dedup stats/ CSVs (one CSV per source):
#   python3 ${ENGINE_DIR}/tools/estimate_total_tokens.py \
#       --input-path "/mnt/nvme/stats/" --input-format csv --quiet
#   Then: export TOTAL_TOKENS=<output>
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
S3_SYNC_INTERVAL="${S3_SYNC_INTERVAL:-600}"  # seconds (default 10 min); used when NVMe + S3 sync enabled

# NVMe Storage Redirection (Speed-with-Safety)
# If ENABLE_NVME is true, we generate a runtime pipeline config that redirects
# all high-volume outputs (indices, manifests, checkpoints, used_chunks DB) to NVMe.
# A background S3 sync runs periodically to persist outputs to S3.
NVME_MOUNT="${NVME_MOUNT:-/mnt/nvme}"
PIPELINE_CONFIG="${ENGINE_DIR}/config/pipeline.yaml"
CORESET_OUTPUT_DIR="${ENGINE_DIR}/output/coresets"
MANIFEST_OUTPUT_DIR="${ENGINE_DIR}/output/manifests"
CHECKPOINT_OUTPUT_DIR=""  # Set when NVMe enabled below, or after ENGINE_DIR is set
S3_SYNC_DEST=""  # Set later if NVMe is enabled

if [ "${ENABLE_NVME}" = "true" ] && [ -d "${NVME_MOUNT}" ]; then
    echo "[INFO] NVMe Storage Enabled: Generating runtime config with NVMe paths"
    PIPELINE_CONFIG="${ENGINE_DIR}/config/pipeline_runtime.yaml"
    CORESET_OUTPUT_DIR="${NVME_MOUNT}/output/coresets"
    MANIFEST_OUTPUT_DIR="${NVME_MOUNT}/output/manifests"
    CHECKPOINT_OUTPUT_DIR="${NVME_MOUNT}/output/checkpoints"
    mkdir -p "${CORESET_OUTPUT_DIR}" "${MANIFEST_OUTPUT_DIR}" "${CHECKPOINT_OUTPUT_DIR}"

    # Generate runtime pipeline.yaml with NVMe-redirected output paths
    python3 -c "
import yaml
with open('${ENGINE_DIR}/config/pipeline.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['io']['output_coreset_path'] = '${NVME_MOUNT}/output/coresets'
cfg['io']['output_manifest_path'] = '${NVME_MOUNT}/output/manifests'
with open('${PIPELINE_CONFIG}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('[OK] Runtime config written: ${PIPELINE_CONFIG}')
print('     output_coreset_path  = ${NVME_MOUNT}/output/coresets')
print('     output_manifest_path = ${NVME_MOUNT}/output/manifests')
"

    # S3 destination for periodic sync: override with S3_SYNC_DEST or use S3_BUCKET.
    # Normalize to exactly one trailing slash to avoid double-slash paths (e.g. .../t3-coreset_outputs//checkpoints/).
    S3_SYNC_DEST="${S3_SYNC_DEST:-s3://${S3_BUCKET}/t3-coreset_outputs}"
    S3_SYNC_DEST="${S3_SYNC_DEST%/}/"
    echo "[INFO] S3 sync destination: ${S3_SYNC_DEST}"
fi
# ------------------------------------------------------------------------------

# --- Infrastructure Validation Overrides (for validate_infra.sh) ---------------
# These flow through to validate_infra.sh via sudo -E.
ENABLE_NVME="${ENABLE_NVME:-}"                   # auto-detected if empty; set true/false to force
# ------------------------------------------------------------------------------

# ==============================================================================
# 1. System Setup & Prerequisites
# ==============================================================================
echo "### [1/8] System Setup & Prerequisites ###"
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
        sudo apt install -y python3.12 python3.12-venv git python3-pip unzip dstat bc sysstat
        sudo sysctl -w vm.swappiness=0

        # Install AWS CLI v2 if not present or broken
        if ! aws --version &> /dev/null; then
            echo "Installing/Updating AWS CLI v2..."
            # Detect architecture
            ARCH=$(uname -m)
            if [ "$ARCH" = "x86_64" ]; then
                AWS_ZIP_URL="https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip"
            elif [ "$ARCH" = "aarch64" ]; then
                AWS_ZIP_URL="https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip"
            else
                echo "[ERROR] Unsupported architecture for AWS CLI: $ARCH"
                exit 1
            fi

            curl "$AWS_ZIP_URL" -o "awscliv2.zip"
            unzip -o awscliv2.zip
            if [ -d "/usr/local/aws-cli" ]; then
                sudo ./aws/install --update
            else
                sudo ./aws/install
            fi
            rm -rf aws awscliv2.zip
        fi
    else
        echo "[SKIP] Non-Linux OS (${OS_TYPE}) — skipping apt packages."
    fi

    if ! command -v uv &> /dev/null; then
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        if [ -f "$HOME/.local/bin/env" ]; then
            source "$HOME/.local/bin/env"
        elif [ -f "$HOME/.cargo/env" ]; then
            source "$HOME/.cargo/env"
        fi
    fi
fi

# ==============================================================================
# 2. AWS Authentication Check
# ==============================================================================
echo "### [2/8] AWS Authentication Check ###"
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
echo "### [3/8] Repository Setup ###"
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
# echo "### [4/8] Dependency Sync (via UV) ###"
EXPERIMENT_DIR="${REPO_ROOT}/experiments/3_coreset_engineering"
ENGINE_DIR="${EXPERIMENT_DIR}/coreset_engine_v5"
# Checkpoint dir: use NVMe path if already set, else EBS/engine path
CHECKPOINT_OUTPUT_DIR="${CHECKPOINT_OUTPUT_DIR:-${ENGINE_DIR}/output/checkpoints}"

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
        source .venv/bin/activate
    fi
else
    echo "[ERROR] Experiment directory not found: ${EXPERIMENT_DIR}"
    exit 1
fi

# ==============================================================================
# 5. Infrastructure Validation
# ==============================================================================
echo "### [5/8] Infrastructure Validation ###"
VALIDATE_INFRA="${ENGINE_DIR}/scripts/validate_infra.sh"

# Export infra thresholds so sudo -E passes them to validate_infra.sh
export S3_BUCKET S3_PREFIX
[ -n "${ENABLE_NVME}" ] && export ENABLE_NVME

if [ "${SKIP_VALIDATION}" = "true" ]; then
    echo "[SKIP] Infrastructure validation skipped (--skip-validation)."
elif [ "${DRY_RUN}" = "true" ]; then
    echo "[DRY RUN] Would run: sudo -E bash ${VALIDATE_INFRA}"
    echo "          Thresholds: nvme=${ENABLE_NVME:-auto} skip_ebs=${SKIP_EBS_VALIDATION} skip_valid=${SKIP_VALIDATION}"
else
    if [ -f "${VALIDATE_INFRA}" ]; then
        echo "Running infrastructure validation..."
        echo "  Thresholds: nvme=${ENABLE_NVME:-auto} skip_ebs=${SKIP_EBS_VALIDATION} skip_valid=${SKIP_VALIDATION}"
        if sudo -E bash "${VALIDATE_INFRA}"; then
            echo "[OK] Infrastructure validation passed."
        else
            echo "[ERROR] Infrastructure validation failed. Fix issues before continuing."
            exit 1
        fi
    else
        echo "[WARN] validate_infra.sh not found at ${VALIDATE_INFRA}. Skipping."
    fi
fi

# ==============================================================================
# 6. Start Monitoring
# ==============================================================================
echo "### [6/8] Start Monitoring ###"
MONITOR_SCRIPT="${ENGINE_DIR}/scripts/monitor.sh"
MONITOR_PID=""
LOG_DIR="${LOG_DIR:-/mnt/nvme/logs}"

if [ "${DRY_RUN}" = "true" ]; then
    echo "[DRY RUN] Would run: nohup bash ${MONITOR_SCRIPT} &"
elif [ -f "${MONITOR_SCRIPT}" ]; then
    echo "Starting background monitoring..."
    nohup bash "${MONITOR_SCRIPT}" > /dev/null 2>&1 &
    MONITOR_PID=$!
    echo "[OK] Monitoring started (PID: ${MONITOR_PID})"
else
    echo "[WARN] monitor.sh not found at ${MONITOR_SCRIPT}. Skipping."
fi

# ==============================================================================
# 7. Launch Pipeline
# ==============================================================================
echo "### [7/8] Launching Pipeline ###"
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
        echo ""
        echo "  Infra Thresholds:"
        echo "    NVMe:           ${ENABLE_NVME:-auto}"
        echo "    Pipeline Config: ${PIPELINE_CONFIG}"
        echo ""
        echo "  Storage Layout:"
        echo "    Coreset Output: ${CORESET_OUTPUT_DIR}"
        echo "    Manifest Output: ${MANIFEST_OUTPUT_DIR}"
        echo "    Checkpoints:    ${CHECKPOINT_OUTPUT_DIR}"
        if [ -n "${S3_SYNC_DEST}" ]; then
        echo "    S3 Sync Dest:   ${S3_SYNC_DEST}"
        echo "    Sync Interval:  Every $((S3_SYNC_INTERVAL / 60)) minutes"
        fi
        echo ""
        echo "  Would execute:"
        echo "    bash experiments/3_coreset_engineering/coreset_engine_v5/shard.sh"
        echo "      --config \"${PIPELINE_CONFIG}\""
        echo "      --num-shards ${NUM_SHARDS} --stages \"${STAGES}\""
        echo "      --input-path \"${S3_INPUT_PATH}\" --total-tokens ${TOTAL_TOKENS} --batch-size ${BATCH_SIZE}"
        echo "      --checkpoint-base \"${CHECKPOINT_OUTPUT_DIR}\" --checkpoint-every-n-batches ${CHECKPOINT_EVERY_N_BATCHES} ${RESUME_FLAG}"
        echo "      --used-cache-max-entries ${USED_CACHE_MAX_ENTRIES} --used-cache-stats-every ${USED_CACHE_STATS_EVERY}"
        echo "      --batch-prefetch-mode ${BATCH_PREFETCH_MODE}"
        echo "      --batch-prefetch-queue-size ${BATCH_PREFETCH_QUEUE_SIZE}"
        echo "      --batch-prefetch-auto-min-batch-size ${BATCH_PREFETCH_AUTO_MIN_BATCH_SIZE}"
        echo "      --batch-prefetch-auto-max-shard-cpu-ratio ${BATCH_PREFETCH_AUTO_MAX_SHARD_CPU_RATIO}"
        echo "      --batch-prefetch-auto-min-wait-ms ${BATCH_PREFETCH_AUTO_MIN_WAIT_MS}"
        echo "      --batch-prefetch-auto-warmup-batches ${BATCH_PREFETCH_AUTO_WARMUP_BATCHES}"
        echo ""
        echo "  After-run instructions:"
        echo "    python3 ${ENGINE_DIR}/scripts/monitor_report.py"
        echo "    python3 ${ENGINE_DIR}/tools/validate_coreset_outputs.py --stages ${STAGES}"
        echo "=========================================="
        exit 0
fi

# Background: Used for manual EC2 runs with nohup for SSH disconnect safety
echo "Starting shard.sh in background via nohup..."
nohup bash experiments/3_coreset_engineering/coreset_engine_v5/shard.sh \
    --config "${PIPELINE_CONFIG}" \
    --num-shards ${NUM_SHARDS} \
    --stages "${STAGES}" \
    --input-path "${S3_INPUT_PATH}" \
    --input-format parquet \
    --total-tokens ${TOTAL_TOKENS} \
    --batch-size ${BATCH_SIZE} \
    --checkpoint-base "${CHECKPOINT_OUTPUT_DIR}" \
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

PIPELINE_PID=$!
echo "[OK] Pipeline launched (PID: ${PIPELINE_PID})"

# ==============================================================================
# 7b. Background S3 Sync (every 10 minutes)
# ==============================================================================
SYNC_PID=""

_s3_sync_once() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[S3 SYNC] ${ts} Syncing outputs to ${S3_SYNC_DEST}..."
    # S3_SYNC_DEST is normalized to one trailing slash; append subpath with no extra slash
    if ! aws s3 sync "${CORESET_OUTPUT_DIR}" "${S3_SYNC_DEST}coresets/" --only-show-errors --no-progress 2>&1; then
        echo "[S3 SYNC] ${ts} ERROR: coresets sync failed"
    fi
    if ! aws s3 sync "${MANIFEST_OUTPUT_DIR}" "${S3_SYNC_DEST}manifests/" --only-show-errors --no-progress 2>&1; then
        echo "[S3 SYNC] ${ts} ERROR: manifests sync failed"
    fi
    # Sync checkpoints
    if ! aws s3 sync "${CHECKPOINT_OUTPUT_DIR}/" "${S3_SYNC_DEST}checkpoints/" --only-show-errors --no-progress 2>&1; then
        echo "[S3 SYNC] ${ts} ERROR: checkpoints sync failed"
    fi
    echo "[S3 SYNC] $(date '+%Y-%m-%d %H:%M:%S') Sync complete."
}

if [ -n "${S3_SYNC_DEST}" ]; then
    echo "### [7b] Starting Background S3 Sync (every $((S3_SYNC_INTERVAL / 60)) min) ###"
    echo "  Sync destination: ${S3_SYNC_DEST}"
    (
        while kill -0 ${PIPELINE_PID} 2>/dev/null; do
            sleep ${S3_SYNC_INTERVAL}
            _s3_sync_once
        done
        # Final sync after pipeline exits
        echo "[S3 SYNC] Pipeline finished. Running final sync..."
        _s3_sync_once
    ) >> s3_sync.log 2>&1 &
    SYNC_PID=$!
    echo "[OK] Background S3 sync started (PID: ${SYNC_PID})"
fi

# ==============================================================================
# 8. Post-Run: Final Persistence & Validation
# ==============================================================================
echo "-----------------------------------------------------------------------"
echo "DEPLOYMENT COMPLETE — Pipeline running in background."
echo "  Monitor logs:    tail -f ${REPO_ROOT}/shard_run.log"
echo "  S3 sync logs:    tail -f ${REPO_ROOT}/s3_sync.log"
echo "  Check process:   ps aux | grep shard.sh"
echo "  Stop pipeline:   pkill -f shard.sh"
echo ""
echo "  Storage Layout:"
echo "    Coreset Output:  ${CORESET_OUTPUT_DIR}"
echo "    Manifest Output: ${MANIFEST_OUTPUT_DIR}"
echo "    Checkpoints:     ${CHECKPOINT_OUTPUT_DIR}"
if [ -n "${S3_SYNC_DEST}" ]; then
    echo "    S3 Destination:  ${S3_SYNC_DEST}"
    echo "    S3 Paths:       coresets/  manifests/  checkpoints/"
    echo "    Sync Interval:   Every $((S3_SYNC_INTERVAL / 60)) minutes (+ auto final sync)"
fi
echo ""
echo "  After pipeline finishes, run post-run validation:"
echo "    # Stop monitoring"
echo "    kill \$(cat ${LOG_DIR}/monitor.pid)"
echo "    # Generate monitoring report"
echo "    bash ${ENGINE_DIR}/scripts/monitor_report.sh"
echo "    # Validate coreset outputs"
echo "    python3 ${ENGINE_DIR}/tools/validate_coreset_outputs.py \\"
echo "        --curriculum ${ENGINE_DIR}/config/curriculum.yaml \\"
echo "        --output-dir ${CORESET_OUTPUT_DIR} \\"
echo "        --stages ${STAGES} --format both"
echo "-----------------------------------------------------------------------"
