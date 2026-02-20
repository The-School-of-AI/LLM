#!/usr/bin/env bash
set -euo pipefail

TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$TEST_ROOT/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
CODE_DIR="$TEST_ROOT/code"
CFG="$TEST_ROOT/configs/test8_reversible_kronecker_triton_gsa_deltanet_extra_fused_1000steps.yaml"
RESULTS_DIR="$TEST_ROOT/results"
INIT_CKPT="$RESULTS_DIR/init/model_init.pt"
INIT_META="$RESULTS_DIR/init/model_init_meta.json"

NUM_GPUS="${NUM_GPUS:-8}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE_REWRITE_INIT="${FORCE_REWRITE_INIT:-0}"

mkdir -p "$RESULTS_DIR/init" "$RESULTS_DIR/run"

if [[ ! -f "$INIT_CKPT" || "$FORCE_REWRITE_INIT" == "1" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Saving deterministic init model for Test 8..."
  "$PYTHON_BIN" "$TEST_ROOT/scripts/save_init_model.py" \
    --config "$CFG" \
    --output "$INIT_CKPT" \
    --meta "$INIT_META"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Reusing existing init model: $INIT_CKPT"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Test 8 (additional fused kernels ON: fused CE path, 1000 steps)..."
(
  cd "$CODE_DIR"
  "$DEEPSPEED_BIN" --num_gpus="$NUM_GPUS" main.py --config "$CFG"
) 2>&1 | tee "$RESULTS_DIR/run/train.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Test 8 completed"
echo "  Init model: $INIT_CKPT"
echo "  Train log:  $RESULTS_DIR/run/train.log"
echo "  Metrics:    $RESULTS_DIR/run/metrics.jsonl"
