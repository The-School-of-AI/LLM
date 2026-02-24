#!/usr/bin/env bash
set -euo pipefail

TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$TEST_ROOT/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
CODE_DIR="$TEST_ROOT/code"
CFG="$TEST_ROOT/configs/test14_gsa_only_liger_kernels_1000steps.yaml"
RESULTS_DIR="$TEST_ROOT/results"
INIT_CKPT="$RESULTS_DIR/init/model_init.pt"
INIT_META="$RESULTS_DIR/init/model_init_meta.json"

NUM_GPUS="${NUM_GPUS:-8}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE_REWRITE_INIT="${FORCE_REWRITE_INIT:-1}"

mkdir -p "$RESULTS_DIR/init" "$RESULTS_DIR/run"

# ---------------------------------------------------------------------------
# Parse loader_type, shard_dir, eval_shard_dir from YAML config
# ---------------------------------------------------------------------------
_yaml_val() {
  "$PYTHON_BIN" -c "
import sys, yaml
with open('$CFG') as f:
    cfg = yaml.safe_load(f)
key = '$1'
val = cfg.get('data', {}).get(key)
print('' if val is None else str(val))
"
}

LOADER_TYPE="$(_yaml_val loader_type)"
SHARD_DIR="$(_yaml_val shard_dir)"
EVAL_SHARD_DIR="$(_yaml_val eval_shard_dir)"

# ---------------------------------------------------------------------------
# Auto-create shards if loader_type=bin_idx and shard dirs are missing/empty
# ---------------------------------------------------------------------------
if [[ "$LOADER_TYPE" == "bin_idx" ]]; then
  if [[ -n "$SHARD_DIR" && ( ! -d "$SHARD_DIR" || -z "$(ls -A "$SHARD_DIR" 2>/dev/null)" ) ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Shard dir missing/empty: $SHARD_DIR — creating train shards..."
    mkdir -p "$SHARD_DIR"
    (
      cd "$CODE_DIR"
      "$PYTHON_BIN" "$TEST_ROOT/scripts/create_shards.py" \
        --dataset wikitext \
        --dataset-config wikitext-103-raw-v1 \
        --split train \
        --output-dir "$SHARD_DIR" \
        --tokenizer "$CODE_DIR/src/tokenizer" \
        --tokens-per-shard 4096000 \
        --band B1 \
        --domain general \
        --stage 1
    )
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Train shards written to: $SHARD_DIR"
  fi

  if [[ -n "$EVAL_SHARD_DIR" && ( ! -d "$EVAL_SHARD_DIR" || -z "$(ls -A "$EVAL_SHARD_DIR" 2>/dev/null)" ) ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Eval shard dir missing/empty: $EVAL_SHARD_DIR — creating eval shards..."
    mkdir -p "$EVAL_SHARD_DIR"
    (
      cd "$CODE_DIR"
      "$PYTHON_BIN" "$TEST_ROOT/scripts/create_shards.py" \
        --dataset wikitext \
        --dataset-config wikitext-103-raw-v1 \
        --split validation \
        --output-dir "$EVAL_SHARD_DIR" \
        --tokenizer "$CODE_DIR/src/tokenizer" \
        --tokens-per-shard 4096000 \
        --band B1 \
        --domain general \
        --stage 1
    )
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Eval shards written to: $EVAL_SHARD_DIR"
  fi
fi

if [[ ! -f "$INIT_CKPT" || "$FORCE_REWRITE_INIT" == "1" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Saving deterministic init model for Test 14..."
  "$PYTHON_BIN" "$TEST_ROOT/scripts/save_init_model.py" \
    --config "$CFG" \
    --output "$INIT_CKPT" \
    --meta "$INIT_META"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Reusing existing init model: $INIT_CKPT"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Test 14 (GSA-only, Liger RoPE/MLP/fused CE, no DeltaNet, 1000 steps)..."
(
  cd "$CODE_DIR"
  "$DEEPSPEED_BIN" --num_gpus="$NUM_GPUS" main.py --config "$CFG"
) 2>&1 | tee "$RESULTS_DIR/run/train.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Test 14 completed"
echo "  Init model:      $INIT_CKPT"
echo "  Train log:       $RESULTS_DIR/run/train.log"
echo "  Metrics:         $RESULTS_DIR/run/metrics.jsonl"
echo "  Profile report:  $RESULTS_DIR/run/profile_report.txt  (if profile_steps were set)"
echo "  Profile JSONL:   $RESULTS_DIR/run/profile.jsonl       (if profile_steps were set)"
