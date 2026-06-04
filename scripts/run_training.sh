#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CODE_DIR="$TEST_ROOT"
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
        --tokenizer "$TEST_ROOT/tokenizer" \
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
        --tokenizer "$TEST_ROOT/tokenizer" \
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

# DeepSpeed's launcher has its own SIGINT handler that immediately kills all
# worker processes, preventing our Python checkpoint handler from running.
# Workaround: catch SIGINT in bash, forward it as SIGUSR1 to the rank-0
# worker (which triggers a graceful checkpoint-then-exit), then wait for
# DeepSpeed to finish on its own.
DS_PID=""
SIGINT_COUNT=0
SIGINT_LAST=0
_forward_sigint() {
  if [[ -n "$DS_PID" ]]; then
    NOW=$(date +%s)
    SIGINT_COUNT=$((SIGINT_COUNT + 1))

    # Second Ctrl+C within 15s → hard kill the process group
    if [[ $SIGINT_COUNT -ge 2 ]] && [[ $((NOW - SIGINT_LAST)) -lt 15 ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Second Ctrl+C — sending SIGTERM to process group"
      kill -TERM -- -"$DS_PID" 2>/dev/null || true
      return
    fi

    # Reset counter if more than 15s since last press
    if [[ $((NOW - SIGINT_LAST)) -ge 15 ]]; then
      SIGINT_COUNT=1
    fi
    SIGINT_LAST=$NOW

    # First Ctrl+C → send SIGUSR1 to workers (checkpoint + stop)
    # DeepSpeed runs in its own session (setsid), so SIGINT does NOT reach it.
    # Find the main.py worker processes (not upload child processes) and send SIGUSR1.
    # Use LOCAL_RANK env var as a discriminator — only training workers have it set.
    WORKER_PIDS=""
    for pid in $(pgrep -f "main.py.*--config" 2>/dev/null); do
      # Skip processes that are upload child processes (they have S3-Uploader in /proc/PID/comm or environ)
      # Quick check: training workers have LOCAL_RANK in their environ
      if [[ -r "/proc/$pid/environ" ]] && grep -qz "LOCAL_RANK" "/proc/$pid/environ" 2>/dev/null; then
        WORKER_PIDS="$WORKER_PIDS $pid"
      elif [[ ! -r "/proc/$pid/environ" ]]; then
        # macOS or no procfs — fall back to parent check
        _p=$pid
        while [[ -n "$_p" && "$_p" != "1" && "$_p" != "0" ]]; do
          _p=$(ps -o ppid= -p "$_p" 2>/dev/null | tr -d ' ')
          if [[ "$_p" == "$DS_PID" ]]; then
            WORKER_PIDS="$WORKER_PIDS $pid"
            break
          fi
        done
      fi
    done
    WORKER_PIDS=$(echo "$WORKER_PIDS" | xargs)  # trim whitespace
    if [[ -z "$WORKER_PIDS" ]]; then
      # Fallback: find any main.py python processes
      WORKER_PIDS=$(pgrep -f "python.*main.py" 2>/dev/null || true)
    fi
    if [[ -n "$WORKER_PIDS" ]]; then
      for pid in $WORKER_PIDS; do
        kill -USR1 "$pid" 2>/dev/null || true
      done
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ctrl+C intercepted — sent SIGUSR1 to $( echo "$WORKER_PIDS" | wc -w | tr -d ' ') workers (checkpoint + stop)"
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Press Ctrl+C again within 15s to force-kill"
    else
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ctrl+C intercepted — no worker PIDs found, sending SIGTERM to process group"
      kill -TERM -- -"$DS_PID" 2>/dev/null || true
    fi
  fi
}
trap _forward_sigint INT

# Run DeepSpeed in its own process group via setsid.
# This prevents Ctrl+C (SIGINT) from reaching the launcher/workers directly.
# Our bash trap catches SIGINT and forwards SIGUSR1 to workers instead.
setsid bash -c "cd '$TEST_ROOT' && '$DEEPSPEED_BIN' --num_gpus='$NUM_GPUS' scripts/train_entrypoint.py --config '$CFG'" &
DS_PID=$!
# wait in a loop: the first wait may return early when the trap fires
while true; do
  wait $DS_PID 2>/dev/null
  TRAIN_EXIT=$?
  # If the process actually exited (not just interrupted by signal), break
  if ! kill -0 "$DS_PID" 2>/dev/null; then
    break
  fi
done
trap - INT  # Restore default SIGINT handling for the rest of the script
if [[ $TRAIN_EXIT -ne 0 ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training exited with code $TRAIN_EXIT"
fi

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
