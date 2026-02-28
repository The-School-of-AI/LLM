#!/usr/bin/env bash
# Verify with Gemma-1B for generation (requires GPU and sufficient RAM).
# Retrieval stays small for speed; generation uses HuggingFace Gemma.

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

python -m benchmark_indic_rag_suite \
  --dataset ai4bharat/Indic-Rag-Suite \
  --split dev \
  --lang hi \
  --max-samples 10 \
  --retrieval-backend small \
  --generation-backend hf \
  --generation-model google/gemma-2-1b \
  --device cuda \
  --tasks retrieval generation \
  -o results_verify_gemma.json

echo "Verify (Gemma) results: results_verify_gemma.json"
