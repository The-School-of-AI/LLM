#!/usr/bin/env bash
# Full evaluation: all languages, test split. Requires GPU and model path.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

MODEL="${MODEL:?Set MODEL env var (e.g. MODEL=google/gemma-3-1b-it)}"
DEVICE="${DEVICE:-cuda}"
OUT="${OUT:-results_test.json}"

python -m benchmark_indicgenbench \
  --config configs/test.yaml \
  --model-name "$MODEL" \
  --device "$DEVICE" \
  -o "$OUT"

echo "Test results: $OUT"
