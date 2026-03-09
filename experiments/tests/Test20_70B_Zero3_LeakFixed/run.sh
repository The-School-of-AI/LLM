#!/usr/bin/env bash
# Re-exec with bash if launched via sh (e.g. sh run.sh)
if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi
set -euo pipefail
# Prevent exit 141 (SIGPIPE) when a pipeline writer outlives the reader (e.g. cmd | head -1).
trap '' PIPE

TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$TEST_ROOT/code"
export PYTHONPATH="${TEST_ROOT}:${PYTHONPATH:-}"
CFG="${CFG:-$TEST_ROOT/configs/test17_3b_moe.yaml}"
RESULTS_DIR="$TEST_ROOT/results"
INIT_CKPT="$RESULTS_DIR/init/model_init.pt"
INIT_META="$RESULTS_DIR/init/model_init_meta.json"

NUM_GPUS="${NUM_GPUS:-8}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE_REWRITE_INIT="${FORCE_REWRITE_INIT:-1}"
# Auto-detect expandable_segments based on model size AND GPU memory.
# Benchmark on A100-40GB: expandable_segments HELPS 8B+ (>40% VRAM), HURTS 1B/3B (-16% to -22%).
# On A100-80GB (p4de): even 8B uses only ~25% VRAM, so expandable_segments hurts everywhere.
# Rule: only enable when model is large AND GPU memory is <=48GB (i.e. 40GB class).
if [[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
  _gpu_mem_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  _model_name=$("$PYTHON_BIN" -c "import yaml; print(yaml.safe_load(open('$CFG'))['model']['model_name'])" 2>/dev/null || echo "unknown")
  if [[ "${_gpu_mem_mib:-0}" -le 49152 ]]; then
    # 40GB GPU (p4d): only large models need expandable_segments
    case "$_model_name" in
      *8b*|*70b*|*120b*) export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" ;;
    esac
  fi
  # 80GB GPU (p4de): never set expandable_segments — all models fit comfortably
  unset _model_name _gpu_mem_mib
fi
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

mkdir -p "$RESULTS_DIR/init" "$RESULTS_DIR/run"

# ---------------------------------------------------------------------------
# Pre-flight version check — refuse to run if critical libs don't match
# Pinned versions verified leak-free (d_alloc=+0.00G) on 2026-03-03.
# See BUGFIX_REPORT_FLA_AUTOCAST_DTYPE.md and requirements-pinned.txt.
# Set SKIP_VERSION_CHECK=1 to bypass (NOT recommended).
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
  _check_ver "triton"    "3.6.0"       "$TRITON_VER"
  _check_ver "deepspeed" "0.18.6"      "$DS_VER"
  _check_ver "fla"       "0.4.2"       "$FLA_VER"

  if [[ "$VERSION_OK" == "0" ]]; then
    echo ""
    echo "  VERSION MISMATCH DETECTED — aborting to prevent memory leak."
    echo "  Install pinned versions:  pip install -r requirements-pinned.txt --index-url https://download.pytorch.org/whl/cu128"
    echo "  Or bypass (risky):        SKIP_VERSION_CHECK=1 bash run.sh"
    exit 1
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Version check passed."
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using config: $CFG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-<unset>}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] TORCHDYNAMO_DISABLE=$TORCHDYNAMO_DISABLE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] T19_STEP_CUDA_SYNC=$T19_STEP_CUDA_SYNC"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] T19_STEP_GC_COLLECT=$T19_STEP_GC_COLLECT"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] T19_STEP_EMPTY_CACHE=$T19_STEP_EMPTY_CACHE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] T19_ZERO3_RELEASE_EVERY=$T19_ZERO3_RELEASE_EVERY"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] T19_ZERO3_FORCE_CLEAR_CONTAINERS=$T19_ZERO3_FORCE_CLEAR_CONTAINERS"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] T19_CLEAR_ROUTER_CACHE_EVERY=$T19_CLEAR_ROUTER_CACHE_EVERY"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] T19_REV_CKPT_USE_REENTRANT=$T19_REV_CKPT_USE_REENTRANT"

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

# Resolve relative paths against config file directory (matches main.py _resolve_path)
CFG_DIR="$(cd "$(dirname "$CFG")" && pwd)"
_abs_path() {
  local p="$1"
  [[ -z "$p" ]] && echo "" && return
  [[ "$p" = /* ]] && echo "$p" || echo "$CFG_DIR/$p"
}
SHARD_DIR="$(_abs_path "$_raw_shard_dir")"
EVAL_SHARD_DIR="$(_abs_path "$_raw_eval_shard_dir")"

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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Test 14 (GSA-only, Liger RoPE/MLP/fused CE, bin_idx loader support)..."
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
if [[ "$LOADER_TYPE" == "bin_idx" ]]; then
  echo "  Train shards:    $SHARD_DIR"
  [[ -n "$EVAL_SHARD_DIR" ]] && echo "  Eval shards:     $EVAL_SHARD_DIR"
fi
