#!/usr/bin/env bash
# Test run: full (or capped) evaluation. Use for paper-comparable numbers.
# For IndicMSMARCO, official metric is MRR@10 with monolingual pool per language.

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

# Default: Indic-Rag-Suite, test split, all languages. Override as needed.
DATASET="${DATASET:-ai4bharat/Indic-Rag-Suite}"
SPLIT="${SPLIT:-test}"
LANG="${LANG:-all}"
OUT="${OUT:-results_test.json}"
# Optional: cap samples for quicker test, e.g. MAX_SAMPLES=100
EXTRA="${EXTRA:-}"

python -m benchmark_indic_rag_suite \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --lang "$LANG" \
  $([ -n "$MAX_SAMPLES" ] && echo "--max-samples $MAX_SAMPLES") \
  --retrieval-backend small \
  --generation-backend small \
  --mrr-at-k 10 \
  --tasks retrieval generation \
  -o "$OUT" \
  $EXTRA

echo "Test results: $OUT"
