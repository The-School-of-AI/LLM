#!/usr/bin/env bash
# Run OPUS benchmark: 1B non-rev model, 10 steps, 8x A100-40GB
# Usage: bash run_opus_1b.sh
set -euo pipefail

TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CFG="$TEST_ROOT/configs/test_1b_nonrev_opus_4096_10steps.yaml"

mkdir -p "$TEST_ROOT/results/run_opus"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] OPUS 1B benchmark: $CFG"
exec bash "$TEST_ROOT/run.sh"
