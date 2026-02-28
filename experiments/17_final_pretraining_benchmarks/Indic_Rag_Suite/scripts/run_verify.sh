#!/usr/bin/env bash
# Quick verification: small retrieval + small generation, one language, few samples.
# Use this to confirm the harness runs end-to-end before full dev/test.

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

python -m benchmark_indic_rag_suite \
  --dataset ai4bharat/Indic-Rag-Suite \
  --split dev \
  --lang hi \
  --max-samples 10 \
  --retrieval-backend small \
  --generation-backend small \
  --tasks retrieval generation \
  -o results_verify.json

echo "Verify results: results_verify.json"
