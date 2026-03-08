#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_JSON="${1:-$ROOT/results/optimizer_ab_1b_500steps.json}"
NAMO_DIR="$ROOT/third_party/namo"

echo "[setup] Ensuring local NAMO checkout exists..."
mkdir -p "$ROOT/third_party"
if [[ ! -d "$NAMO_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/minxin-zhg/namo.git "$NAMO_DIR"
else
  git -C "$NAMO_DIR" fetch --depth 1 origin main
  git -C "$NAMO_DIR" reset --hard FETCH_HEAD
fi

echo "[run] Starting 1B optimizer A/B benchmark (AdamW -> NAMO-D), 500 steps each"
export PYTHONPATH="$NAMO_DIR/src:${PYTHONPATH:-}"
python3 "$ROOT/scripts/benchmark_optimizer_ab_1b.py" \
  --config "$ROOT/configs/optimizer_ab_1b_500steps.yaml" \
  --json-out "$OUT_JSON"

echo "[done] Results: $OUT_JSON"
