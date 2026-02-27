#!/usr/bin/env bash
# =============================================================================
# Sharded Coreset Builder Runner
# Runs N parallel shards of the coreset selection pipeline.
#
# Usage:
#   bash shard.sh \
#     --input-path "data/books/bands/" \
#
#   bash shard.sh \
#     --num-shards 8 --stages "1B 3B 8B 70B" \
#     --input-path "data/books/bands/" --input-format parquet \
#     --config config/pipeline.yaml --curriculum config/curriculum_v7.yaml \
#     --checkpoint-base output/checkpoints --total-tokens 4523096944
# =============================================================================
set -euo pipefail

# --------------- SIGNAL HANDLING ---------------
PIDS=()
cleanup() {
  local sig=$1
  echo ""
  echo "[!] Interrupted by $sig. Shutting down all shards..."
  
  # Send TERM to the entire process group
  # This kills the script and all its children (python, sed, etc.)
  trap - SIGINT SIGTERM EXIT
  kill 0 2>/dev/null || true
  
  # Final fallback for stubborn processes
  sleep 1
  kill -9 0 2>/dev/null || true
  exit 1
}

trap 'cleanup SIGINT' SIGINT
trap 'cleanup SIGTERM' SIGTERM

# --------------- DEFAULTS ---------------
NUM_SHARDS=4
STAGES="1B 3B 8B 70B"
INPUT_PATH="data/datasets/large_sample_chunks.parquet"
INPUT_FORMAT="parquet"
CONFIG="config/pipeline.yaml"
CURRICULUM="config/curriculum.yaml"
CHECKPOINT_BASE="output/checkpoints"
BAND_INFERENCE="none"
BAND_SCORE_SOURCE="band_score"
BATCH_SIZE=80000
CHECKPOINT_EVERY_N_BATCHES=3
USED_CACHE_MAX_ENTRIES=0
USED_CACHE_STATS_EVERY=0
BATCH_PREFETCH_MODE="off"
BATCH_PREFETCH_QUEUE_SIZE=1
BATCH_PREFETCH_AUTO_MIN_BATCH_SIZE=50000
BATCH_PREFETCH_AUTO_MAX_SHARD_CPU_RATIO=1.0
BATCH_PREFETCH_AUTO_MIN_WAIT_MS=2.0
BATCH_PREFETCH_AUTO_WARMUP_BATCHES=5
TOTAL_TOKENS=""
RESUME=false

# --------------- PARSE ARGS ---------------
usage() {
  echo "Usage: $0 --input-path <path> --total-tokens <N> [options]"
  echo ""
  echo "Required:"
  echo "  --input-path        Path to input data directory or file"
  echo ""
  echo "Optional:"
  echo "  --num-shards        Number of parallel shards (default: 4)"
  echo "  --stages            Space-separated stage list (default: \"1B 3B 8B 70B\")"
  echo "  --input-format      Input format: parquet or jsonl (default: jsonl)"
  echo "  --config            Pipeline config path (default: config/pipeline.yaml)"
  echo "  --curriculum        Curriculum config path (default: config/curriculum.yaml)"
  echo "  --checkpoint-base   Base dir for checkpoints (default: output/checkpoints)"
  echo "  --band-inference    Band inference mode (default: none)"
  echo "                     Values: none | infer_if_missing | infer_if_ineligible | force"
  echo "  --band-score-source Band score source (default: auto)"
  echo "                     Values: auto | band_score | difficulty_score | band_p_max | band_p_argmax | band_p_B0..band_p_B5"
  echo "  --batch-size        Rows/chunks per batch in streaming mode (default: 80000)"
  echo "  --checkpoint-every-n-batches  Checkpoint cadence passed to coreset_builder (default: 3)"
  echo "  --used-cache-max-entries   Optional in-memory LRU size for used-chunk checks (default: 0=off)"
  echo "  --used-cache-stats-every   Log used-cache hit-rate every N batches (default: 0=off)"
  echo "  --batch-prefetch-mode      Batch prefetch mode (default: auto)"
  echo "                     Values: off | on | auto"
  echo "  --batch-prefetch-queue-size  Prefetch queue size (default: 1)"
  echo "  --batch-prefetch-auto-min-batch-size  Auto mode min batch size (default: 50000)"
  echo "  --batch-prefetch-auto-max-shard-cpu-ratio  Auto mode max shard/cpu ratio (default: 1.0)"
  echo "  --batch-prefetch-auto-min-wait-ms  Auto mode warmup wait threshold in ms (default: 2.0)"
  echo "  --batch-prefetch-auto-warmup-batches  Auto mode warmup batch count (default: 5)"
  echo "  --resume            Resume from last checkpoints (don't clean output dirs)"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-shards)       NUM_SHARDS="$2";       shift 2 ;;
    --stages)           STAGES="$2";           shift 2 ;;
    --input-path)       INPUT_PATH="$2";       shift 2 ;;
    --input-format)     INPUT_FORMAT="$2";     shift 2 ;;
    --config)           CONFIG="$2";           shift 2 ;;
    --curriculum)       CURRICULUM="$2";       shift 2 ;;
    --checkpoint-base)  CHECKPOINT_BASE="$2";  shift 2 ;;
    --band-inference)   BAND_INFERENCE="$2";   shift 2 ;;
    --band-score-source) BAND_SCORE_SOURCE="$2"; shift 2 ;;
    --batch-size)       BATCH_SIZE="$2";       shift 2 ;;
    --checkpoint-every-n-batches) CHECKPOINT_EVERY_N_BATCHES="$2"; shift 2 ;;
    --used-cache-max-entries) USED_CACHE_MAX_ENTRIES="$2"; shift 2 ;;
    --used-cache-stats-every) USED_CACHE_STATS_EVERY="$2"; shift 2 ;;
    --batch-prefetch-mode) BATCH_PREFETCH_MODE="$2"; shift 2 ;;
    --batch-prefetch-queue-size) BATCH_PREFETCH_QUEUE_SIZE="$2"; shift 2 ;;
    --batch-prefetch-auto-min-batch-size) BATCH_PREFETCH_AUTO_MIN_BATCH_SIZE="$2"; shift 2 ;;
    --batch-prefetch-auto-max-shard-cpu-ratio) BATCH_PREFETCH_AUTO_MAX_SHARD_CPU_RATIO="$2"; shift 2 ;;
    --batch-prefetch-auto-min-wait-ms) BATCH_PREFETCH_AUTO_MIN_WAIT_MS="$2"; shift 2 ;;
    --batch-prefetch-auto-warmup-batches) BATCH_PREFETCH_AUTO_WARMUP_BATCHES="$2"; shift 2 ;;
    --total-tokens)     TOTAL_TOKENS="$2";     shift 2 ;;
    --resume)           RESUME=true;           shift 1 ;;
    -h|--help)          usage ;;
    *)                  echo "Unknown option: $1"; usage ;;
  esac
done

if [[ -z "$INPUT_PATH" ]]; then echo "ERROR: --input-path is required"; usage; fi

# Change to project root (directory containing this script)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Coreset Sharded Run"
echo "  Shards       : $NUM_SHARDS"
echo "  Stages       : $STAGES"
echo "  Input        : $INPUT_PATH ($INPUT_FORMAT)"
echo "  Config       : $CONFIG"
echo "  Curriculum   : $CURRICULUM"
echo "  Checkpoints  : $CHECKPOINT_BASE"
echo "  Batch Size   : $BATCH_SIZE"
echo "  Band Infer   : $BAND_INFERENCE"
echo "  Band Score   : $BAND_SCORE_SOURCE"
echo "  Ckpt Every N : $CHECKPOINT_EVERY_N_BATCHES"
echo "  Used Cache   : max=$USED_CACHE_MAX_ENTRIES stats_every=$USED_CACHE_STATS_EVERY"
echo "  Prefetch     : mode=$BATCH_PREFETCH_MODE queue=$BATCH_PREFETCH_QUEUE_SIZE auto_min_batch=$BATCH_PREFETCH_AUTO_MIN_BATCH_SIZE auto_max_ratio=$BATCH_PREFETCH_AUTO_MAX_SHARD_CPU_RATIO auto_min_wait_ms=$BATCH_PREFETCH_AUTO_MIN_WAIT_MS auto_warmup=$BATCH_PREFETCH_AUTO_WARMUP_BATCHES"
echo "============================================================"

# --------------- PYTHON DETECTION (WINDOWS/GIT-BASH FRIENDLY) ---------------
# Key pitfall on Windows: `python` may resolve to the Microsoft Store alias stub.
# So we don't just check `command -v`; we also verify the interpreter can execute.

_python_cmd_works() {
  local -a _cmd=("$@")
  "${_cmd[@]}" -c "import sys; sys.exit(0)" >/dev/null 2>&1
}

_choose_python() {
  local spec
  local -a cmd

  # Allow override (supports values like: "py -3")
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    read -r -a cmd <<<"$PYTHON_BIN"
    if _python_cmd_works "${cmd[@]}"; then
      PYTHON_CMD=("${cmd[@]}")
      return 0
    fi
    echo "ERROR: PYTHON_BIN='$PYTHON_BIN' does not appear to work." >&2
    return 1
  fi

  # Prefer a local virtualenv interpreter if present (keeps deps consistent).
  # These paths work in Git Bash on Windows and also in typical Unix venv layouts.
  for spec in \
    "./.venv/Scripts/python.exe" \
    "./venv/Scripts/python.exe" \
    "./.venv/bin/python" \
    "./venv/bin/python"; do
    if [[ -f "$spec" ]]; then
      cmd=("$spec")
      if _python_cmd_works "${cmd[@]}"; then
        PYTHON_CMD=("${cmd[@]}")
        return 0
      fi
    fi
  done

  # Prefer py launcher (most reliable on Windows), then python3, then python.
  for spec in "py -3" "python3" "python"; do
    read -r -a cmd <<<"$spec"
    if ! command -v "${cmd[0]}" >/dev/null 2>&1; then
      continue
    fi
    if _python_cmd_works "${cmd[@]}"; then
      PYTHON_CMD=("${cmd[@]}")
      return 0
    fi
  done

  return 1
}

if ! _choose_python; then
  echo "ERROR: Could not find a working Python interpreter." >&2
  echo "Tried: 'py -3', 'python3', 'python' (and optional PYTHON_BIN override)." >&2
  echo "Hint: install Python 3.10+ and ensure it's on PATH, or set PYTHON_BIN='py -3'." >&2
  echo "Hint: disable the Microsoft Store python alias: Settings > Apps > Advanced app settings > App execution aliases." >&2
  exit 1
fi

echo "  Python       : ${PYTHON_CMD[*]}"

# Clean old outputs
if [[ "$RESUME" != "true" ]]; then
  echo "[*] Cleaning previous outputs..."
  rm -rf "$CHECKPOINT_BASE" output/coresets output/manifests 2>/dev/null || true
else
  echo "[*] Resuming: keeping previous outputs..."
fi

# Launch all shards in parallel using background processes
echo "[*] Launching $NUM_SHARDS shards..."
for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
  SHARD_DIR="${CHECKPOINT_BASE}/shard$(printf '%03d' "$SHARD_ID")"
  mkdir -p "$SHARD_DIR"

  (
    echo "[shard $SHARD_ID] Starting..."
    "${PYTHON_CMD[@]}" coreset_builder.py \
      --config "$CONFIG" \
      --curriculum "$CURRICULUM" \
      --input-path "$INPUT_PATH" \
      --input-format "$INPUT_FORMAT" \
      --batch-size "$BATCH_SIZE" \
      --stages $STAGES \
      --num-shards "$NUM_SHARDS" \
      --shard-id "$SHARD_ID" \
      --checkpoint-dir "$SHARD_DIR" \
      --checkpoint-every-n-batches "$CHECKPOINT_EVERY_N_BATCHES" \
      --used-cache-max-entries "$USED_CACHE_MAX_ENTRIES" \
      --used-cache-stats-every "$USED_CACHE_STATS_EVERY" \
      --batch-prefetch-mode "$BATCH_PREFETCH_MODE" \
      --batch-prefetch-queue-size "$BATCH_PREFETCH_QUEUE_SIZE" \
      --batch-prefetch-auto-min-batch-size "$BATCH_PREFETCH_AUTO_MIN_BATCH_SIZE" \
      --batch-prefetch-auto-max-shard-cpu-ratio "$BATCH_PREFETCH_AUTO_MAX_SHARD_CPU_RATIO" \
      --batch-prefetch-auto-min-wait-ms "$BATCH_PREFETCH_AUTO_MIN_WAIT_MS" \
      --batch-prefetch-auto-warmup-batches "$BATCH_PREFETCH_AUTO_WARMUP_BATCHES" \
      --band-inference "$BAND_INFERENCE" \
      --band-score-source "$BAND_SCORE_SOURCE" \
      ${TOTAL_TOKENS:+--total-input-tokens-estimate "$TOTAL_TOKENS"} \
      2>&1 | sed "s/^/[shard $SHARD_ID] /"
    echo "[shard $SHARD_ID] Done."
  ) &
  PIDS+=($!)
done

# Wait for all shards and track failures
FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "============================================================"
if [[ $FAILED -eq 0 ]]; then
  echo "  All $NUM_SHARDS shards completed successfully!"
else
  echo "  WARNING: $FAILED / $NUM_SHARDS shards failed!"
fi
echo "  Manifests: output/coresets/*/manifest_shard*.json"
echo "  Reports:   output/manifests/ablation_validation_report_shard*.md"
echo "============================================================"

exit $FAILED