#!/usr/bin/env bash
set -euo pipefail

TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$TEST_ROOT/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
CODE_DIR="$TEST_ROOT/code"
CFG="$TEST_ROOT/configs/test9_extend_from_test8_to_3000steps.yaml"
RESULTS_DIR="$TEST_ROOT/results"
INIT_CKPT="$RESULTS_DIR/init/model_init.pt"
INIT_META="$RESULTS_DIR/init/model_init_meta.json"

NUM_GPUS="${NUM_GPUS:-8}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE_REWRITE_INIT="${FORCE_REWRITE_INIT:-0}"

SOURCE_TEST8_CKPT_DIR="${SOURCE_TEST8_CKPT_DIR:-/Users/rohanshravan/Downloads/LLM-code-20260219-1351_rohan_patch_v3/experiments/tests/Test_8_additional_fused_kernels_1000steps/results/run/checkpoints}"
RESUME_TAG="${RESUME_TAG:-epoch0_step1000}"
DEST_CKPT_DIR="$RESULTS_DIR/run/checkpoints"

mkdir -p "$RESULTS_DIR/init" "$RESULTS_DIR/run" "$DEST_CKPT_DIR"

if [[ ! -d "$SOURCE_TEST8_CKPT_DIR/$RESUME_TAG" ]]; then
  echo "ERROR: Missing source checkpoint tag: $SOURCE_TEST8_CKPT_DIR/$RESUME_TAG" >&2
  echo "Run Test 8 first or override SOURCE_TEST8_CKPT_DIR/RESUME_TAG." >&2
  exit 1
fi

if [[ ! -d "$DEST_CKPT_DIR/$RESUME_TAG" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Staging Test 8 checkpoint tag $RESUME_TAG into Test 9 output dir..."
  cp -R "$SOURCE_TEST8_CKPT_DIR/$RESUME_TAG" "$DEST_CKPT_DIR/"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Resume checkpoint already staged: $DEST_CKPT_DIR/$RESUME_TAG"
fi

# Keep init artifact for traceability (resume path uses checkpoint state).
if [[ ! -f "$INIT_CKPT" || "$FORCE_REWRITE_INIT" == "1" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Saving deterministic init model artifact..."
  "$PYTHON_BIN" "$TEST_ROOT/scripts/save_init_model.py" \
    --config "$CFG" \
    --output "$INIT_CKPT" \
    --meta "$INIT_META"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Reusing existing init artifact: $INIT_CKPT"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Test 9 continuation from tag $RESUME_TAG..."
(
  cd "$CODE_DIR"
  "$DEEPSPEED_BIN" --num_gpus="$NUM_GPUS" main.py --config "$CFG"
) 2>&1 | tee "$RESULTS_DIR/run/train.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Test 9 completed"
echo "  Resume source: $SOURCE_TEST8_CKPT_DIR/$RESUME_TAG"
echo "  Train log:     $RESULTS_DIR/run/train.log"
echo "  Metrics:       $RESULTS_DIR/run/metrics.jsonl"
