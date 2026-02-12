#!/usr/bin/env bash
# =============================================================================
# Sharded Coreset Builder Runner
# Runs N parallel shards of the coreset selection pipeline.
#
# Usage:
#   bash shard.sh \
#     --input-path "data/books/bands/" \
#     --total-tokens 4523096944
#
#   bash shard.sh \
#     --num-shards 8 --stages "1B 3B 8B 70B" \
#     --input-path "data/books/bands/" --input-format parquet \
#     --config config/pipeline.yaml --curriculum config/curriculum_v7.yaml \
#     --checkpoint-base output/checkpoints --total-tokens 4523096944
# =============================================================================
set -euo pipefail

# --------------- DEFAULTS ---------------
NUM_SHARDS=4
STAGES="1B 3B 8B 70B"
INPUT_PATH="data/books/bands/"
INPUT_FORMAT="jsonl"
CONFIG="config/pipeline.yaml"
CURRICULUM="config/curriculum.yaml"
CHECKPOINT_BASE="output/checkpoints"
BAND_INFERENCE="none"
#TOTAL_TOKENS="4523096944"

# --------------- PARSE ARGS ---------------
usage() {
  echo "Usage: $0 --input-path <path> --total-tokens <N> [options]"
  echo ""
  echo "Required:"
  echo "  --input-path        Path to input data directory or file"
  #echo "  --total-tokens      Estimated total input tokens"
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
    #--total-tokens)     TOTAL_TOKENS="$2";     shift 2 ;;
    -h|--help)          usage ;;
    *)                  echo "Unknown option: $1"; usage ;;
  esac
done

if [[ -z "$INPUT_PATH" ]]; then echo "ERROR: --input-path is required"; usage; fi
#if [[ -z "$TOTAL_TOKENS" ]]; then echo "ERROR: --total-tokens is required"; usage; fi

# Change to project root (directory containing this script)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Coreset Sharded Run"
echo "  Shards       : $NUM_SHARDS"
echo "  Stages       : $STAGES"
echo "  Input        : $INPUT_PATH ($INPUT_FORMAT)"
#echo "  Total Tokens : $(printf "%'d" "$TOTAL_TOKENS")"
echo "  Config       : $CONFIG"
echo "  Curriculum   : $CURRICULUM"
echo "  Checkpoints  : $CHECKPOINT_BASE"
echo "  Band Infer   : $BAND_INFERENCE"
echo "============================================================"

# Clean old outputs
echo "[*] Cleaning previous outputs..."
rm -rf "$CHECKPOINT_BASE" output/coresets output/manifests 2>/dev/null || true

# Launch all shards in parallel using background processes
echo "[*] Launching $NUM_SHARDS shards..."
PIDS=()
for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
  SHARD_DIR="${CHECKPOINT_BASE}/shard$(printf '%03d' "$SHARD_ID")"
  mkdir -p "$SHARD_DIR"

  (
    echo "[shard $SHARD_ID] Starting..."
    python coreset_builder.py \
      --config "$CONFIG" \
      --curriculum "$CURRICULUM" \
      --input-path "$INPUT_PATH" \
      --input-format "$INPUT_FORMAT" \
      --stages $STAGES \
      --num-shards "$NUM_SHARDS" \
      --shard-id "$SHARD_ID" \
      --checkpoint-dir "$SHARD_DIR" \
      --band-inference "$BAND_INFERENCE" \
      #--total-input-tokens-estimate "$TOTAL_TOKENS" \
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