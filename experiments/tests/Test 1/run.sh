#!/usr/bin/env bash
set -euo pipefail

TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$TEST_ROOT/code"
CONFIG_DIR="$TEST_ROOT/configs"
RESULTS_DIR="$TEST_ROOT/results"

NUM_GPUS="${NUM_GPUS:-8}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"

mkdir -p "$RESULTS_DIR/lead_wo_rev" "$RESULTS_DIR/diff_rec"

run_one() {
  local name="$1"
  local cfg="$2"
  local out_log="$RESULTS_DIR/$name/train.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running $name with config: $cfg"
  (
    cd "$CODE_DIR"
    "$DEEPSPEED_BIN" --num_gpus="$NUM_GPUS" main.py --config "$cfg"
  ) 2>&1 | tee "$out_log"
}

run_one "lead_wo_rev" "$CONFIG_DIR/test1_lead_wo_rev.yaml"
run_one "diff_rec" "$CONFIG_DIR/test1_diff_rec.yaml"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Test 1 comparative run completed."
echo "Results:"
echo "  - $RESULTS_DIR/lead_wo_rev/train.log"
echo "  - $RESULTS_DIR/diff_rec/train.log"
