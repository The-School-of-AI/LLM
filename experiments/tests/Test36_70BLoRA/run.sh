#!/usr/bin/env bash
set -euo pipefail

TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$TEST_ROOT/code"
export PYTHONPATH="${TEST_ROOT}:${PYTHONPATH:-}"
CFG="${CFG:-$TEST_ROOT/configs/test_70b_moe_lora_4096_bs32_10steps.yaml}"
RESULTS_DIR="$TEST_ROOT/results"
INIT_CKPT="$RESULTS_DIR/init/model_init.pt"
INIT_META="$RESULTS_DIR/init/model_init_meta.json"

NUM_GPUS="${NUM_GPUS:-8}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE_REWRITE_INIT="${FORCE_REWRITE_INIT:-0}"

# LoRA Adaptive Head — freeze base embeddings, train head via LoRA
export EXP_LORA_HEAD="${EXP_LORA_HEAD:-1}"
export EXP_LORA_RANK="${EXP_LORA_RANK:-256}"
export EXP_LORA_HEAD_SIZE="${EXP_LORA_HEAD_SIZE:-8192}"

# ZeRO-3 on 70B MoE: aggressive per-step cleanup required (matches Test20)
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export T19_STEP_CUDA_SYNC="${T19_STEP_CUDA_SYNC:-1}"
export T19_STEP_GC_COLLECT="${T19_STEP_GC_COLLECT:-1}"
export T19_STEP_EMPTY_CACHE="${T19_STEP_EMPTY_CACHE:-1}"
export T19_STEP_IPC_COLLECT="${T19_STEP_IPC_COLLECT:-0}"
export T19_ZERO3_RELEASE_EVERY="${T19_ZERO3_RELEASE_EVERY:-1}"
export T19_ZERO3_FORCE_CLEAR_CONTAINERS="${T19_ZERO3_FORCE_CLEAR_CONTAINERS:-0}"
export T19_CLEAR_ROUTER_CACHE_EVERY="${T19_CLEAR_ROUTER_CACHE_EVERY:-1}"
export T19_TRACK_CUDA_MEMORY="${T19_TRACK_CUDA_MEMORY:-1}"
export T19_REV_CKPT_USE_REENTRANT="${T19_REV_CKPT_USE_REENTRANT:-0}"

# Intelligent expandable_segments: only on 40-48GB GPUs, never on 80GB (from Test20)
GPU_MEM_MB=$("$PYTHON_BIN" -c "
import torch
if torch.cuda.is_available():
    print(torch.cuda.get_device_properties(0).total_mem // (1024*1024))
else:
    print(0)
" 2>/dev/null || echo "0")
if [[ "$GPU_MEM_MB" -gt 0 && "$GPU_MEM_MB" -le 49152 ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Enabled expandable_segments (GPU=${GPU_MEM_MB}MB <= 48GB)"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skipping expandable_segments (GPU=${GPU_MEM_MB}MB > 48GB)"
fi

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
if [[ "$LOADER_TYPE" == "bin_idx" ]]; then
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
# Save deterministic init model (skipped for 70B — uses deepspeed.zero.Init)
# ---------------------------------------------------------------------------
MODEL_NAME=$("$PYTHON_BIN" -c "
import yaml
with open('$CFG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('model', {}).get('model_name', ''))
" 2>/dev/null || echo "")

if [[ "$MODEL_NAME" == "70bmoe" || "$MODEL_NAME" == "120bmoe" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skipping init model save (${MODEL_NAME} uses zero.Init)"
elif [[ ! -f "$INIT_CKPT" || "$FORCE_REWRITE_INIT" == "1" ]]; then
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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP_LORA_HEAD=$EXP_LORA_HEAD EXP_LORA_RANK=$EXP_LORA_RANK EXP_LORA_HEAD_SIZE=$EXP_LORA_HEAD_SIZE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting 70B LoRA Training (ZeRO-3)..."
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
fi
