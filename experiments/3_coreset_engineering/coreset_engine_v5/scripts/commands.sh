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
#   CI (self-hosted): ./commands.sh --foreground --skip-repo-setup   (checkout already done)
#   CI (SSH):         ./commands.sh --foreground                     (clones repo on EC2)
#
# Examples:
#
#   # Production run on c7gd.16xlarge (defaults):
#   export S3_BUCKET="my-bucket"
#   ./commands.sh
#
#   # Dry run — preview all config without executing:
#   export S3_BUCKET="my-bucket"
#   ./commands.sh --dry-run
#
#   # Full pipeline run on a smaller instance (c6i.8xlarge, no NVMe):
#   export S3_BUCKET="my-bucket"
#   export EXPECTED_INSTANCE_TYPE="c6i.8xlarge"
#   export MIN_VCPU=32
#   export MIN_RAM_GB=60
#   export ENABLE_NVME=false
#   export MIN_EBS_FREE_GB=200
#   export MIN_EBS_ROOT_GB=500
#   ./commands.sh --foreground
#
#   # Override pipeline parameters:
#   export S3_BUCKET="my-bucket"
#   export NUM_SHARDS=4
#   export STAGES="1B 3B"
#   export BATCH_SIZE=50000
#   export RESUME=true
#   ./commands.sh --foreground
#   # Estimate TOTAL_TOKENS from post-dedup stats/ CSVs on EC2:
#   #   Option 1: Python tool (sums total_tokens from all source CSVs)
#   python3 experiments/3_coreset_engineering/coreset_engine_v5/tools/estimate_total_tokens.py \
#       --input-path "/mnt/nvme/stats/" --input-format csv --quiet
#   #   Option 2: Quick awk one-liner across all source CSVs
#   awk -F',' 'NR>1{s+=$COL}END{print s}' /mnt/nvme/stats/*.csv
#   # Then export before running:
#   export TOTAL_TOKENS=4523096944
#
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status

# --- Parse flags --------------------------------------------------------------
DRY_RUN=false
FOREGROUND=false
SKIP_REPO_SETUP=false
SKIP_EBS_VALIDATION="${SKIP_EBS_VALIDATION:-false}"
SKIP_VALIDATION="${SKIP_VALIDATION:-false}"

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --foreground) FOREGROUND=true ;;
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
S3_BUCKET="${S3_BUCKET:?ERROR: S3_BUCKET is not set. Export it before running: export S3_BUCKET=your-bucket-name}"
S3_INPUT_PATH="${S3_INPUT_PATH:-s3://${S3_BUCKET}/processed_dataset/curriculum_pyspark_output/}"
S3_PREFIX="${S3_PREFIX:-processed_dataset/curriculum_pyspark_output/source=C4/}"
NUM_SHARDS="${NUM_SHARDS:-8}"
STAGES="${STAGES:-1B}"
TOTAL_TOKENS="${TOTAL_TOKENS:-4523096944}"
# ^ Get TOTAL_TOKENS from post-dedup stats/ CSVs (one CSV per source):
#   python3 ${ENGINE_DIR}/tools/estimate_total_tokens.py \
#       --input-path "/mnt/nvme/stats/" --input-format csv --quiet
#   Then: export TOTAL_TOKENS=<output>
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

# --- Infrastructure Validation Overrides (for validate_infra.sh) ---------------
# These flow through to validate_infra.sh via sudo -E.
# Less common thresholds (MAX_CPU_STEAL_PCT, MAX_NVME_LATENCY_US, MIN_NVME_IOPS,
# MAX_EBS_AWAIT_MS, MIN_EBS_IOPS, MAX_SWAPPINESS, MIN_OPEN_FILES, MIN_S3_SPEED_MBS)
# can be overridden directly via env vars without adding them here.
EXPECTED_INSTANCE_TYPE="${EXPECTED_INSTANCE_TYPE:-c7gd.16xlarge}"
MIN_VCPU="${MIN_VCPU:-64}"
MIN_RAM_GB="${MIN_RAM_GB:-120}"
ENABLE_NVME="${ENABLE_NVME:-}"                   # auto-detected if empty; set true/false to force
MIN_NVME_FREE_GB="${MIN_NVME_FREE_GB:-400}"
MIN_EBS_ROOT_GB="${MIN_EBS_ROOT_GB:-1000}"
MIN_EBS_FREE_GB="${MIN_EBS_FREE_GB:-800}"
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
        sudo apt install -y python3.12 python3.12-venv git python3-pip unzip
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

# if [ -d "${EXPERIMENT_DIR}" ]; then
#     cd "${EXPERIMENT_DIR}"
#     if [ "${DRY_RUN}" = "true" ]; then
#         echo "[DRY RUN] Would create .venv and run: uv sync"
#         if [ -f "pyproject.toml" ]; then
#             echo "[OK] pyproject.toml found at ${EXPERIMENT_DIR}/pyproject.toml"
#         else
#             echo "[ERROR] pyproject.toml NOT found at ${EXPERIMENT_DIR}/"
#         fi
#     else
#         if [ ! -d ".venv" ]; then
#             uv venv .venv
#         fi
#         export UV_PROJECT_ENVIRONMENT=$(pwd)/.venv
#         uv sync
#     fi
# else
#     echo "[ERROR] Experiment directory not found: ${EXPERIMENT_DIR}"
#     exit 1
# fi

# ==============================================================================
# 5. Infrastructure Validation
# ==============================================================================
echo "### [5/8] Infrastructure Validation ###"
VALIDATE_INFRA="${ENGINE_DIR}/scripts/validate_infra.sh"

# Export infra thresholds so sudo -E passes them to validate_infra.sh
export EXPECTED_INSTANCE_TYPE MIN_VCPU MIN_RAM_GB MIN_EBS_ROOT_GB MIN_EBS_FREE_GB
export MIN_NVME_FREE_GB SKIP_EBS_VALIDATION S3_BUCKET S3_PREFIX
[ -n "${ENABLE_NVME}" ] && export ENABLE_NVME

if [ "${SKIP_VALIDATION}" = "true" ]; then
    echo "[SKIP] Infrastructure validation skipped (--skip-validation)."
elif [ "${DRY_RUN}" = "true" ]; then
    echo "[DRY RUN] Would run: sudo -E bash ${VALIDATE_INFRA}"
    echo "          Thresholds: instance=${EXPECTED_INSTANCE_TYPE} vcpu>=${MIN_VCPU} ram>=${MIN_RAM_GB}GB nvme=${ENABLE_NVME:-auto} ebs>=${MIN_EBS_FREE_GB}GB skip_ebs=${SKIP_EBS_VALIDATION} skip_valid=${SKIP_VALIDATION}"
else
    if [ -f "${VALIDATE_INFRA}" ]; then
        echo "Running infrastructure validation..."
        echo "  Thresholds: instance=${EXPECTED_INSTANCE_TYPE} vcpu>=${MIN_VCPU} ram>=${MIN_RAM_GB}GB nvme=${ENABLE_NVME:-auto} ebs>=${MIN_EBS_FREE_GB}GB skip_ebs=${SKIP_EBS_VALIDATION} skip_valid=${SKIP_VALIDATION}"
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
        echo "  Foreground:   ${FOREGROUND}"
        echo ""
        echo "  Infra Thresholds:"
        echo "    Instance:   ${EXPECTED_INSTANCE_TYPE}"
        echo "    vCPU:       >= ${MIN_VCPU}"
        echo "    RAM:        >= ${MIN_RAM_GB} GB"
        echo "    NVMe:       ${ENABLE_NVME:-auto} (free >= ${MIN_NVME_FREE_GB} GB)"
        echo "    EBS Root:   >= ${MIN_EBS_ROOT_GB} GB (free >= ${MIN_EBS_FREE_GB} GB)"
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
        echo ""
        echo "  Post-run (foreground mode only):"
        echo "    python3 ${ENGINE_DIR}/scripts/monitor_report.py"
        echo "    python3 ${ENGINE_DIR}/tools/validate_coreset_outputs.py --stages ${STAGES}"
        echo "=========================================="
        exit 0
fi

if [ "${FOREGROUND}" = "true" ]; then
        # Foreground: Used by CI/SSH so exit code is tracked
        echo "Running shard.sh in foreground..."
        PIPELINE_EXIT=0
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
            ${RESUME_FLAG} || PIPELINE_EXIT=$?

        # ==========================================================================
        # 8. Post-Run Validation & Reports
        # ==========================================================================
        echo "### [8/8] Post-Run Validation & Reports ###"

        # 8a. Stop monitoring and generate report
        if [ -n "${MONITOR_PID}" ] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
            echo "Stopping monitoring (PID: ${MONITOR_PID})..."
            kill "${MONITOR_PID}" 2>/dev/null || true
            sleep 2
        fi

        MONITOR_REPORT_PY="${ENGINE_DIR}/scripts/monitor_report.py"
        if [ -f "${MONITOR_REPORT_PY}" ]; then
            LOG_DIR="${LOG_DIR:-/mnt/nvme/logs}"
            if [ -d "${LOG_DIR}" ]; then
                echo "Generating monitoring HTML report..."
                python3 "${MONITOR_REPORT_PY}" "${LOG_DIR}" || echo "[WARN] Monitoring report generation failed."
            else
                echo "[WARN] Log directory ${LOG_DIR} not found. Skipping monitoring report."
            fi
        fi

        # 8b. Validate coreset outputs against curriculum
        VALIDATE_OUTPUTS="${ENGINE_DIR}/tools/validate_coreset_outputs.py"
        if [ -f "${VALIDATE_OUTPUTS}" ]; then
            echo "Validating coreset outputs against curriculum..."
            python3 "${VALIDATE_OUTPUTS}" \
                --curriculum "${ENGINE_DIR}/config/curriculum.yaml" \
                --output-dir "${ENGINE_DIR}/output/coresets" \
                --stages ${STAGES} \
                --format both \
                --report-dir "${ENGINE_DIR}/output/validation_reports" \
                || echo "[WARN] Output validation reported issues (see above)."
        else
            echo "[WARN] validate_coreset_outputs.py not found. Skipping output validation."
        fi

        echo ""
        echo "=======================================================================" 
        echo "  PRODUCTION RUN COMPLETE"
        echo "  Pipeline exit code:  ${PIPELINE_EXIT}"
        echo "  Manifests:           output/coresets/*/manifest_shard*.json"
        echo "  Ablation Reports:    output/manifests/ablation_validation_report*.md"
        echo "  Validation Reports:  ${ENGINE_DIR}/output/validation_reports/"
        echo "  Monitoring Report:   ${LOG_DIR:-/mnt/nvme/logs}/report_*.html"
        echo "======================================================================="
        exit ${PIPELINE_EXIT}
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
        echo "DEPLOYMENT COMPLETE — Pipeline running in background."
        echo "  Monitor logs:    tail -f ${REPO_ROOT}/shard_run.log"
        echo "  Check process:   ps aux | grep shard.sh"
        echo ""
        echo "  After pipeline finishes, run post-run validation manually:"
        echo "    # Stop monitoring"
        echo "    kill \$(cat ${LOG_DIR}/monitor.pid)"
        echo "    # Generate monitoring report"
        echo "    python3 ${ENGINE_DIR}/scripts/monitor_report.py"
        echo "    # Validate coreset outputs"
        echo "    python3 ${ENGINE_DIR}/tools/validate_coreset_outputs.py \\"
        echo "        --curriculum ${ENGINE_DIR}/config/curriculum.yaml \\"
        echo "        --stages ${STAGES} --format both"
        echo "-----------------------------------------------------------------------"
fi
