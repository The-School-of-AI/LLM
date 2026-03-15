#!/usr/bin/env bash
set -euo pipefail

TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$TEST_ROOT/code"
export PYTHONPATH="${TEST_ROOT}:${PYTHONPATH:-}"
CFG="${CFG:-$TEST_ROOT/configs/train_1b_nonrev_z1.yaml}"
RESULTS_DIR="$TEST_ROOT/results"
INIT_CKPT="$RESULTS_DIR/init/model_init.pt"
INIT_META="$RESULTS_DIR/init/model_init_meta.json"

NUM_GPUS="${NUM_GPUS:-8}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE_REWRITE_INIT="${FORCE_REWRITE_INIT:-0}"

# Best config from autoresearch (exp53: SC15+RoPE100K+beta2=0.99+max_live=5e8)
export EXP_SOFTCAP="${EXP_SOFTCAP:-15}"
export EXP_ROPE_BASE="${EXP_ROPE_BASE:-100000}"
export EXP_DN_ROPE_BASE="${EXP_DN_ROPE_BASE:-100000}"
export EXP_MAX_FUSED_SIZE="${EXP_MAX_FUSED_SIZE:-4096}"

# ZeRO-3 on 3B MoE model with 80GB GPUs: no per-step cleanup needed
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export T19_STEP_CUDA_SYNC="${T19_STEP_CUDA_SYNC:-0}"
export T19_STEP_GC_COLLECT="${T19_STEP_GC_COLLECT:-0}"
export T19_STEP_EMPTY_CACHE="${T19_STEP_EMPTY_CACHE:-0}"
export T19_STEP_IPC_COLLECT="${T19_STEP_IPC_COLLECT:-0}"
export T19_ZERO3_RELEASE_EVERY="${T19_ZERO3_RELEASE_EVERY:-0}"
export T19_ZERO3_FORCE_CLEAR_CONTAINERS="${T19_ZERO3_FORCE_CLEAR_CONTAINERS:-0}"
export T19_CLEAR_ROUTER_CACHE_EVERY="${T19_CLEAR_ROUTER_CACHE_EVERY:-0}"
export T19_TRACK_CUDA_MEMORY="${T19_TRACK_CUDA_MEMORY:-1}"
export T19_REV_CKPT_USE_REENTRANT="${T19_REV_CKPT_USE_REENTRANT:-0}"

mkdir -p "$RESULTS_DIR/init" "$RESULTS_DIR/run"

# ---------------------------------------------------------------------------
# Pre-flight version check
# ---------------------------------------------------------------------------
if [[ "${SKIP_VERSION_CHECK:-0}" != "1" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running pre-flight version check..."
  VERSION_OK=1
  _check_ver() {
    local name="$1" expected="$2" actual="$3"
    if [[ "$actual" != "$expected" ]]; then
      echo "  FAIL: $name expected=$expected got=$actual"
      VERSION_OK=0
    else
      echo "  OK:   $name=$actual"
    fi
  }

  TORCH_VER=$("$PYTHON_BIN" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "MISSING")
  TRITON_VER=$("$PYTHON_BIN" -c "import triton; print(triton.__version__)" 2>/dev/null || echo "MISSING")
  DS_VER=$("$PYTHON_BIN" -c "import deepspeed; print(deepspeed.__version__)" 2>/dev/null || echo "MISSING")
  FLA_VER=$("$PYTHON_BIN" -c "import fla; print(fla.__version__)" 2>/dev/null || echo "MISSING")

  _check_ver "torch"     "2.7.1+cu128" "$TORCH_VER"
  _check_ver "triton"    "3.3.1"       "$TRITON_VER"
  _check_ver "deepspeed" "0.18.6"      "$DS_VER"
  _check_ver "fla"       "0.4.2"       "$FLA_VER"

  if [[ "$VERSION_OK" == "0" ]]; then
    echo ""
    echo "  VERSION MISMATCH — aborting to prevent memory leak."
    echo "  Install: pip install -r requirements-pinned.txt --index-url https://download.pytorch.org/whl/cu128"
    echo "  Bypass:  SKIP_VERSION_CHECK=1 bash run.sh"
    exit 1
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Version check passed."
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Config: $CFG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] TORCHDYNAMO_DISABLE=$TORCHDYNAMO_DISABLE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] T19_STEP_CUDA_SYNC=$T19_STEP_CUDA_SYNC"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] T19_STEP_GC_COLLECT=$T19_STEP_GC_COLLECT"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] T19_STEP_EMPTY_CACHE=$T19_STEP_EMPTY_CACHE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] T19_TRACK_CUDA_MEMORY=$T19_TRACK_CUDA_MEMORY"

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
_raw_shard_dir="$(_yaml_val shard_dir)"
_raw_eval_shard_dir="$(_yaml_val eval_shard_dir)"

CFG_DIR="$(cd "$(dirname "$CFG")" && pwd)"
_abs_path() {
  local p="$1"
  [[ -z "$p" ]] && echo "" && return
  [[ "$p" = /* ]] && echo "$p" || echo "$CFG_DIR/$p"
}
SHARD_DIR="$(_abs_path "$_raw_shard_dir")"
EVAL_SHARD_DIR="$(_abs_path "$_raw_eval_shard_dir")"

# ---------------------------------------------------------------------------
# Auto-create shards if needed
# ---------------------------------------------------------------------------
if [[ "$LOADER_TYPE" == "curriculum_v2" ]]; then
  if [[ -n "$SHARD_DIR" && ( ! -d "$SHARD_DIR" || -z "$(ls -A "$SHARD_DIR" 2>/dev/null)" ) ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Downloading curriculum test shards from S3: $SHARD_DIR"
    bash "$TEST_ROOT/scripts/download_test_shards.sh" "$SHARD_DIR"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Curriculum test shards ready: $SHARD_DIR"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Curriculum shards already present: $SHARD_DIR"
  fi

elif [[ "$LOADER_TYPE" == "bin_idx" ]]; then
  if [[ -n "$SHARD_DIR" && ( ! -d "$SHARD_DIR" || -z "$(ls -A "$SHARD_DIR" 2>/dev/null)" ) ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating train shards: $SHARD_DIR"
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
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Train shards ready: $SHARD_DIR"
  fi

  if [[ -n "$EVAL_SHARD_DIR" && ( ! -d "$EVAL_SHARD_DIR" || -z "$(ls -A "$EVAL_SHARD_DIR" 2>/dev/null)" ) ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating eval shards: $EVAL_SHARD_DIR"
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
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Eval shards ready: $EVAL_SHARD_DIR"
  fi
fi

# ---------------------------------------------------------------------------
# Save deterministic init model
# ---------------------------------------------------------------------------
if [[ ! -f "$INIT_CKPT" || "$FORCE_REWRITE_INIT" == "1" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Saving deterministic init model..."
  "$PYTHON_BIN" "$TEST_ROOT/scripts/save_init_model.py" \
    --config "$CFG" \
    --output "$INIT_CKPT" \
    --meta "$INIT_META"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Reusing existing init model: $INIT_CKPT"
fi

# ---------------------------------------------------------------------------
# Launch training
# ---------------------------------------------------------------------------
# Kill stale processes from previous runs
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 29500/tcp 2>/dev/null || true
sleep 5  # Wait for port TIME_WAIT to clear

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Training..."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP_SOFTCAP=$EXP_SOFTCAP EXP_ROPE_BASE=$EXP_ROPE_BASE EXP_MAX_FUSED_SIZE=$EXP_MAX_FUSED_SIZE"
(
  cd "$CODE_DIR"
  "$DEEPSPEED_BIN" --num_gpus="$NUM_GPUS" main.py --config "$CFG"
)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed"
echo "  Init model:   $INIT_CKPT"
echo "  Train log:    $RESULTS_DIR/run/train.log"
echo "  Metrics:      $RESULTS_DIR/run/metrics.jsonl"
if [[ "$LOADER_TYPE" == "bin_idx" ]]; then
  echo "  Train shards: $SHARD_DIR"
  [[ -n "$EVAL_SHARD_DIR" ]] && echo "  Eval shards:  $EVAL_SHARD_DIR"
elif [[ "$LOADER_TYPE" == "curriculum_v2" ]]; then
  echo "  Shard root:   $SHARD_DIR"
  echo "  Loader:       curriculum_v2"
fi
