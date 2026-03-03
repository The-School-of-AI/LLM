#!/usr/bin/env bash
# =============================================================================
# run_ruler.sh — Pre-configured RULER benchmark runner for NVIDIA L4
# Usage: bash run_ruler.sh [MODEL_NAME] [BENCHMARK]
#
# Defaults to: bash run_ruler.sh gemma-2b synthetic
# =============================================================================

set -euo pipefail

# ─── USER CONFIGURATION ──────────────────────────────────────────────────────

# Model key (must match a case in RULER/scripts/config_models.sh)
MODEL_NAME="${1:-gemma-2b}"

# Benchmark name (synthetic is the standard RULER suite of 13 tasks)
BENCHMARK="${2:-synthetic}"

# Path where models are stored. The model should be at:
#   ${MODEL_DIR}/google/gemma-2b   (for gemma-2b)
#   ${MODEL_DIR}/google/gemma-1b   (for gemma-1b)
MODEL_DIR="$(pwd)/models"

# Path where generated data and predictions will be written
ROOT_DIR="$(pwd)/results"

# Number of GPU(s) — L4 is single-GPU, keep at 1
GPUS=1

# Inference batch size — L4 can handle 32 for 1-2B models
BATCH_SIZE=32

# Sequence lengths to evaluate (space-separated)
# Remove 16384/32768 if you did NOT patch config.json for extended context
SEQ_LENGTHS=(4096 8192)
# SEQ_LENGTHS=(4096 8192 16384 32768)   # uncomment after extending context

# Samples per task.  Use 10 for a smoke test, 500 for the full benchmark.
NUM_SAMPLES=500
# NUM_SAMPLES=10   # smoke test

# ─── SANITY CHECKS ───────────────────────────────────────────────────────────

RULER_DIR="$(pwd)/RULER"
SCRIPTS_DIR="${RULER_DIR}/scripts"

if [ ! -d "${RULER_DIR}" ]; then
    echo "[ERROR] RULER directory not found at ${RULER_DIR}"
    echo "  Run: git clone https://github.com/NVIDIA/RULER.git RULER"
    exit 1
fi

if ! python -c "import vllm" 2>/dev/null; then
    echo "[ERROR] vllm not installed. Run: pip install -r requirements.txt"
    exit 1
fi

# ─── PATCH CONFIGS IF NEEDED ─────────────────────────────────────────────────

# Automatically apply patches if not yet applied
if ! grep -q "gemma-2b)" "${SCRIPTS_DIR}/config_models.sh" 2>/dev/null; then
    echo "[INFO] Applying config patches..."
    bash "$(pwd)/patch_configs.sh"
fi

# ─── INJECT RUNTIME VARIABLES ────────────────────────────────────────────────

# Temporarily override variables in run.sh via environment
export RULER_ROOT_DIR="${ROOT_DIR}"
export RULER_MODEL_DIR="${MODEL_DIR}"
export RULER_BATCH_SIZE="${BATCH_SIZE}"
export RULER_GPUS="${GPUS}"

# Patch run.sh values inline (sed into a temp copy)
TEMP_RUN="${SCRIPTS_DIR}/.run_patched.sh"
sed \
    -e "s|^ROOT_DIR=.*|ROOT_DIR=\"${ROOT_DIR}\"|" \
    -e "s|^MODEL_DIR=.*|MODEL_DIR=\"${MODEL_DIR}\"|" \
    -e "s|^BATCH_SIZE=.*|BATCH_SIZE=${BATCH_SIZE}|" \
    -e "s|^GPUS=.*|GPUS=\"${GPUS}\"|" \
    "${SCRIPTS_DIR}/run.sh" > "${TEMP_RUN}"

# Patch config_tasks.sh values
TEMP_TASKS="${SCRIPTS_DIR}/.config_tasks_patched.sh"
cp "${SCRIPTS_DIR}/config_tasks.sh" "${TEMP_TASKS}"

# Build SEQ_LENGTHS array string for bash
SEQ_ARR="($(printf '%s ' "${SEQ_LENGTHS[@]}"))"
python3 - <<PYEOF
import re, os

tasks_file = os.path.join("${SCRIPTS_DIR}", ".config_tasks_patched.sh")
with open(tasks_file) as f:
    content = f.read()

# Replace NUM_SAMPLES
content = re.sub(r'^NUM_SAMPLES=\d+', 'NUM_SAMPLES=${NUM_SAMPLES}', content, flags=re.MULTILINE)

# Replace SEQ_LENGTHS block
seq_str = " ".join([str(s) for s in ${SEQ_LENGTHS[@]/#/} if s])
PYEOF

# Use simpler sed approach for config_tasks.sh
sed -i.bak \
    -e "s|^NUM_SAMPLES=.*|NUM_SAMPLES=${NUM_SAMPLES}|" \
    "${SCRIPTS_DIR}/config_tasks.sh"

# Inject SEQ_LENGTHS into config_models.sh (it's defined there for the defaults)
# SEQ_LENGTHS in the actual run is controlled by config_tasks.sh via run.sh sourcing

# Patch SEQ_LENGTHS in config_models.sh
python3 - <<PYEOF
import re

path = "${SCRIPTS_DIR}/config_models.sh"
with open(path) as f:
    content = f.read()

seq_vals = "\n".join(["    " + str(s) for s in [${SEQ_LENGTHS[@]}]])
content = re.sub(
    r'SEQ_LENGTHS=\(\s*[^)]*\)',
    'SEQ_LENGTHS=(\n' + seq_vals + '\n)',
    content,
    flags=re.DOTALL
)

with open(path, "w") as f:
    f.write(content)

print("[INFO] SEQ_LENGTHS patched to: ${SEQ_LENGTHS[*]}")
PYEOF

# ─── RUN ─────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  RULER Benchmark"
echo "  Model    : ${MODEL_NAME}"
echo "  Benchmark: ${BENCHMARK}"
echo "  Seq Lens : ${SEQ_LENGTHS[*]}"
echo "  Samples  : ${NUM_SAMPLES}"
echo "  Model Dir: ${MODEL_DIR}"
echo "  Output   : ${ROOT_DIR}"
echo "============================================================"
echo ""

cd "${SCRIPTS_DIR}"
bash "${TEMP_RUN}" "${MODEL_NAME}" "${BENCHMARK}"

# Restore config_tasks.sh
cp "${SCRIPTS_DIR}/config_tasks.sh.bak" "${SCRIPTS_DIR}/config_tasks.sh" 2>/dev/null || true

echo ""
echo "[DONE] Results written to: ${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/"
