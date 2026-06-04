#!/usr/bin/env bash
set -euo pipefail
#
# Generate .idx index files for all tokens.bin shards in a directory tree.
# Works for both flat (d1_shards/) and nested (curriculum_test_shards/band_*/shard_*/) layouts.
#
# Usage:
#   bash scripts/generate_idx_files.sh [DATA_DIR]
#
# Default: data/curriculum_test_shards in this release tree

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${1:-$ROOT_DIR/data/curriculum_test_shards}"

echo "[$(date '+%H:%M:%S')] Generating .idx files for: $DATA_DIR"

EXISTING_IDX=$(find "$DATA_DIR" -name 'tokens.idx' 2>/dev/null | wc -l | tr -d ' ')
TOTAL_BIN=$(find "$DATA_DIR" -name 'tokens.bin' 2>/dev/null | wc -l | tr -d ' ')

echo "[$(date '+%H:%M:%S')] Found $TOTAL_BIN tokens.bin files, $EXISTING_IDX already have .idx"

if [[ "$EXISTING_IDX" -ge "$TOTAL_BIN" && "$TOTAL_BIN" -gt 0 ]]; then
  echo "[$(date '+%H:%M:%S')] All .idx files already generated. Nothing to do."
  exit 0
fi

python3 << PYEOF
import numpy as np
from pathlib import Path

data_dir = Path('$DATA_DIR')
BYTES_PER_BLOCK = 4096 * 4  # seq_len=4096, 4 bytes per token (int32)
IDX_HEADER = b'\x00' * 8

count = 0
skipped = 0

for bp in sorted(data_dir.rglob('tokens.bin')):
    ip = bp.parent / 'tokens.idx'
    if ip.exists():
        skipped += 1
        continue

    file_size = bp.stat().st_size
    if file_size == 0:
        print(f'  WARN: empty file, skipping: {bp}')
        continue

    n = file_size // BYTES_PER_BLOCK
    offsets = np.arange(n + 1, dtype=np.uint64) * BYTES_PER_BLOCK

    with open(ip, 'wb') as f:
        f.write(IDX_HEADER)
        f.write(offsets.tobytes())

    count += 1
    if count % 200 == 0:
        print(f'  Generated {count} .idx files...')

print(f'  Done: generated {count}, skipped {skipped} (already existed)')
PYEOF

FINAL_IDX=$(find "$DATA_DIR" -name 'tokens.idx' 2>/dev/null | wc -l | tr -d ' ')
echo "[$(date '+%H:%M:%S')] Total .idx files: $FINAL_IDX"
