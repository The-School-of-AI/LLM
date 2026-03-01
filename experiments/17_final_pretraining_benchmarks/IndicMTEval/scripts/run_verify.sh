#!/usr/bin/env bash
# Quick sanity check (~1 min, CPU only)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
python -m benchmark_indic_mt_eval --config configs/verify.yaml "$@"
