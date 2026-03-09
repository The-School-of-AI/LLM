#!/usr/bin/env bash
# Smoke test: dummy model, 5 samples, verifies pipeline runs end-to-end
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

python -m benchmark_indicgenbench \
  --config configs/verify.yaml \
  -o results_verify.json

echo "Verify results: results_verify.json"
