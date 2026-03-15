#!/usr/bin/env bash
set -euo pipefail
#
# Download a subset of real shards from S3 for curriculum checkpoint testing.
#
# Distribution: proportional to 1B stage weights (900 total shards)
#   D1 (band_B0):              378 shards  (weight 0.42)
#   D2 (band_B1):              270 shards  (weight 0.30)
#   D3 (band_B3/code_*):       117 shards  (weight 0.13)
#   D4 (band_B4/B5):            63 shards  (weight 0.07)
#   AON_bench (band_B6):         36 shards  (weight 0.04)
#   AON_indic (indic_numerals):  36 shards  (weight 0.04)
#   ─────────────────────────────────────────
#   Total:                      900 shards  (~14 MB)
#
# Each shard is a directory with a single tokens.bin file (~16KB).
#
# Usage:
#   bash scripts/download_test_shards.sh [OUTPUT_DIR]
#
# Default output: /mnt/local-nvme/data/curriculum_test_shards
# Override:       bash scripts/download_test_shards.sh /tmp/my_test_shards

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST_DIR="$PROJECT_ROOT/manifests"

OUTPUT_DIR="${1:-/mnt/local-nvme/data/curriculum_test_shards}"
S3_BASE="s3://t1-dataacquisition-datasets-2/shards_reordered"

# Pool → manifest file → shard count
declare -A POOL_MANIFEST
POOL_MANIFEST[D1]="D1_shards.txt"
POOL_MANIFEST[D2]="D2_shards.txt"
POOL_MANIFEST[D3]="D3_shards.txt"
POOL_MANIFEST[D4]="D4_shards.txt"
POOL_MANIFEST[AON_bench]="AON_bench_train_shards.txt"
POOL_MANIFEST[AON_indic]="AON_indic_shards.txt"

declare -A POOL_COUNT
POOL_COUNT[D1]=378
POOL_COUNT[D2]=270
POOL_COUNT[D3]=117
POOL_COUNT[D4]=63
POOL_COUNT[AON_bench]=36
POOL_COUNT[AON_indic]=36

TOTAL_EXPECTED=900
TEST_MANIFEST_DIR="$OUTPUT_DIR/_test_manifests"

echo "============================================================"
echo "Curriculum Test Shard Downloader"
echo "============================================================"
echo "S3 source:    $S3_BASE"
echo "Output dir:   $OUTPUT_DIR"
echo "Total shards: $TOTAL_EXPECTED"
echo "============================================================"

mkdir -p "$OUTPUT_DIR" "$TEST_MANIFEST_DIR"

TOTAL_DOWNLOADED=0
TOTAL_SKIPPED=0

for POOL in D1 D2 D3 D4 AON_bench AON_indic; do
  MANIFEST_FILE="${POOL_MANIFEST[$POOL]}"
  COUNT="${POOL_COUNT[$POOL]}"
  MANIFEST_PATH="$MANIFEST_DIR/$MANIFEST_FILE"

  if [[ ! -f "$MANIFEST_PATH" ]]; then
    echo "WARNING: $MANIFEST_PATH not found, skipping $POOL"
    continue
  fi

  # Take first N shards from manifest (deterministic — same order every time)
  SELECTED_SHARDS=$(head -n "$COUNT" "$MANIFEST_PATH")
  ACTUAL_COUNT=$(echo "$SELECTED_SHARDS" | wc -l | tr -d ' ')

  echo ""
  echo "── $POOL: downloading $ACTUAL_COUNT shards ──"

  # Write trimmed manifest for testing
  echo "$SELECTED_SHARDS" > "$TEST_MANIFEST_DIR/$MANIFEST_FILE"

  POOL_DOWNLOADED=0
  POOL_SKIPPED=0

  while IFS= read -r SHARD_REL; do
    [[ -z "$SHARD_REL" ]] && continue

    LOCAL_DIR="$OUTPUT_DIR/$SHARD_REL"
    LOCAL_BIN="$LOCAL_DIR/tokens.bin"

    # Skip if already downloaded
    if [[ -f "$LOCAL_BIN" ]]; then
      POOL_SKIPPED=$((POOL_SKIPPED + 1))
      continue
    fi

    mkdir -p "$LOCAL_DIR"
    S3_PATH="$S3_BASE/$SHARD_REL/tokens.bin"

    if aws s3 cp "$S3_PATH" "$LOCAL_BIN" --quiet 2>/dev/null; then
      POOL_DOWNLOADED=$((POOL_DOWNLOADED + 1))
    else
      echo "  WARN: failed to download $S3_PATH"
      rmdir "$LOCAL_DIR" 2>/dev/null || true
    fi
  done <<< "$SELECTED_SHARDS"

  echo "  $POOL: downloaded=$POOL_DOWNLOADED, skipped(existing)=$POOL_SKIPPED"
  TOTAL_DOWNLOADED=$((TOTAL_DOWNLOADED + POOL_DOWNLOADED))
  TOTAL_SKIPPED=$((TOTAL_SKIPPED + POOL_SKIPPED))
done

# Copy curriculum_v2_manifest.json (pool structure unchanged)
cp "$MANIFEST_DIR/curriculum_v2_manifest.json" "$TEST_MANIFEST_DIR/curriculum_v2_manifest.json"

echo ""
echo "============================================================"
echo "Download complete"
echo "  New downloads: $TOTAL_DOWNLOADED"
echo "  Already existed: $TOTAL_SKIPPED"
echo "  Total shards: $((TOTAL_DOWNLOADED + TOTAL_SKIPPED))"
echo "  Test manifests: $TEST_MANIFEST_DIR"
echo "  Shard root:     $OUTPUT_DIR"
echo ""
echo "Disk usage:"
du -sh "$OUTPUT_DIR"
echo ""
echo "To run training:"
echo "  CFG=configs/test_curriculum_checkpoint.yaml bash run.sh"
echo "============================================================"
