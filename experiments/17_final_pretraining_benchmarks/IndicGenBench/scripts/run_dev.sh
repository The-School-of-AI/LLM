#!/usr/bin/env bash
# Dev run: real model, small subset. Override with env vars.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

MODEL="${MODEL:-google/gemma-3-1b-it}"
LANG="${LANG:-hi}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
DEVICE="${DEVICE:-cpu}"
OUT="${OUT:-results_dev.json}"

python -m benchmark_indicgenbench \
  --config configs/dev.yaml \
  --model-name "$MODEL" \
  --lang "$LANG" \
  --max-samples "$MAX_SAMPLES" \
  --device "$DEVICE" \
  -o "$OUT"

echo "Dev results: $OUT"
