#!/usr/bin/env bash
# Dev run: small subset, configurable language(s). Good for debugging and quick metrics.

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

# Default: Indic-Rag-Suite, dev split, Hindi, small models. Override with env or args.
DATASET="${DATASET:-ai4bharat/Indic-Rag-Suite}"
LANG="${LANG:-hi}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
OUT="${OUT:-results_dev.json}"

python -m benchmark_indic_rag_suite \
  --dataset "$DATASET" \
  --split dev \
  --lang "$LANG" \
  --max-samples "$MAX_SAMPLES" \
  --retrieval-backend small \
  --generation-backend small \
  --tasks retrieval generation \
  -o "$OUT"

echo "Dev results: $OUT"
