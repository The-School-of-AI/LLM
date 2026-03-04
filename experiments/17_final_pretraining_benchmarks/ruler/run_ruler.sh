#!/usr/bin/env bash
# =============================================================================
# run_ruler.sh — Pre-configured RULER benchmark runner for NVIDIA L4
# Usage: bash run_ruler.sh [MODEL_NAME] [BENCHMARK]
#
# Defaults to: bash run_ruler.sh gemma-3-1b-pt synthetic
# =============================================================================

set -euo pipefail

# ─── USER CONFIGURATION (edit these) ─────────────────────────────────────────

# Model key — must match a case in RULER/scripts/config_models.sh
MODEL_NAME="${1:-gemma-3-1b-pt}"

# Benchmark name (synthetic = standard 13-task RULER suite)
BENCHMARK="${2:-synthetic}"

# Directory where models live: model must be at ${MODEL_DIR}/<org>/<name>
MODEL_DIR="$(pwd)/models"

# Where generated data and predictions will be written
ROOT_DIR="$(pwd)/results"

# Number of GPUs — L4 is single-GPU
GPUS=1

# Inference batch size — L4 handles 32 comfortably for 1-3B models
BATCH_SIZE=32

# Sequence lengths to evaluate (space-separated inside the parens)
# Remove 16384/32768 if you did NOT extend max_position_embeddings in config.json
SEQ_LENGTHS=(4096 8192)
# SEQ_LENGTHS=(4096 8192 16384 32768)

# Samples per task. Use 10 for a quick smoke test, 500 for the full benchmark.
NUM_SAMPLES=500
# NUM_SAMPLES=10

# ─── PATHS ───────────────────────────────────────────────────────────────────

RULER_DIR="$(pwd)/RULER"
SCRIPTS_DIR="${RULER_DIR}/scripts"

# ─── CHECKS ──────────────────────────────────────────────────────────────────

if [ ! -d "${RULER_DIR}" ]; then
    echo "[ERROR] RULER not found at ${RULER_DIR}"
    echo "  Fix: git clone https://github.com/NVIDIA/RULER.git RULER"
    exit 1
fi

if ! python -c "import vllm" 2>/dev/null; then
    echo "[ERROR] vllm not found. Install with:"
    echo "    uv sync   (from the ruler directory)"
    exit 1
fi

# Apply model patches if gemma entries are not yet in config_models.sh
if ! grep -q "${MODEL_NAME})" "${SCRIPTS_DIR}/config_models.sh" 2>/dev/null; then
    echo "[INFO] Model '${MODEL_NAME}' not found in config_models.sh."
    echo "  Either run:  bash patch_configs.sh"
    echo "  Or add it manually to RULER/scripts/config_models.sh"
    echo "  See SETUP.md Step 7 for the snippet."
    exit 1
fi

# ─── BUILD SEQ_LENGTHS BASH ARRAY STRING ─────────────────────────────────────

# Convert bash array to a multi-line string for insertion into config_models.sh
SEQ_LENGTHS_STR=""
for s in "${SEQ_LENGTHS[@]}"; do
    SEQ_LENGTHS_STR="${SEQ_LENGTHS_STR}    ${s}"$'\n'
done

# ─── PATCH RULER/scripts/run.sh → temp copy ──────────────────────────────────

TEMP_RUN="${SCRIPTS_DIR}/.run_patched.sh"
sed \
    -e "s|^ROOT_DIR=.*|ROOT_DIR=\"${ROOT_DIR}\"|" \
    -e "s|^MODEL_DIR=.*|MODEL_DIR=\"${MODEL_DIR}\"|" \
    -e "s|^BATCH_SIZE=.*|BATCH_SIZE=${BATCH_SIZE}|" \
    -e "s|^GPUS=.*|GPUS=\"${GPUS}\"|" \
    "${SCRIPTS_DIR}/run.sh" > "${TEMP_RUN}"
chmod +x "${TEMP_RUN}"

# ─── PATCH config_tasks.sh NUM_SAMPLES (in place, restore after) ─────────────

cp "${SCRIPTS_DIR}/config_tasks.sh" "${SCRIPTS_DIR}/config_tasks.sh.bak"
sed -i "s|^NUM_SAMPLES=.*|NUM_SAMPLES=${NUM_SAMPLES}|" "${SCRIPTS_DIR}/config_tasks.sh"

# ─── PATCH SEQ_LENGTHS in config_models.sh (in place, restore after) ─────────

cp "${SCRIPTS_DIR}/config_models.sh" "${SCRIPTS_DIR}/config_models.sh.bak"

# Build the replacement block (pure bash, no Python needed)
LENGTHS_BLOCK="SEQ_LENGTHS=($'\n'${SEQ_LENGTHS_STR})"

# Use awk to replace the multi-line SEQ_LENGTHS block
awk -v lengths="${SEQ_LENGTHS_STR}" '
    /^SEQ_LENGTHS=\(/ {
        print "SEQ_LENGTHS=("
        n = split(lengths, arr, "\n")
        for (i = 1; i <= n; i++) {
            if (arr[i] != "") print arr[i]
        }
        # skip until closing paren
        while ($0 !~ /\)/) { getline }
        print ")"
        next
    }
    { print }
' "${SCRIPTS_DIR}/config_models.sh.bak" > "${SCRIPTS_DIR}/config_models.sh"

# ─── RUN ─────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  RULER Benchmark"
echo "  Model     : ${MODEL_NAME}"
echo "  Benchmark : ${BENCHMARK}"
echo "  Seq Lens  : ${SEQ_LENGTHS[*]}"
echo "  Samples   : ${NUM_SAMPLES}"
echo "  Model Dir : ${MODEL_DIR}"
echo "  Output    : ${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/"
echo "============================================================"
echo ""

cd "${SCRIPTS_DIR}"
bash "${TEMP_RUN}" "${MODEL_NAME}" "${BENCHMARK}"
EXIT_CODE=$?

# ─── RESTORE original configs ─────────────────────────────────────────────────

cp "${SCRIPTS_DIR}/config_tasks.sh.bak"  "${SCRIPTS_DIR}/config_tasks.sh"
cp "${SCRIPTS_DIR}/config_models.sh.bak" "${SCRIPTS_DIR}/config_models.sh"

echo ""
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[DONE] Results written to: ${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/"
else
    echo "[ERROR] Benchmark exited with code ${EXIT_CODE}"
fi
exit ${EXIT_CODE}
