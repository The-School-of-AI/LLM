#!/bin/bash

# Quickstart Script for Tokenizer Frequency Analysis & Reindexing
# This script runs the complete workflow end-to-end

set -e  # Exit on error

echo "=================================================="
echo "Tokenizer Frequency Analysis & Reindexing"
echo "=================================================="
echo ""

# Configuration
TOKENIZER="ds_filtered"  # Change this to your selected tokenizer
CONFIG="../config.yaml"
RESULTS_DIR="../results"

echo "Step 1: Installing dependencies..."
pip install -r ../requirements.txt

echo ""
echo "Step 2: Evaluating tokenizers..."
cd src
python tokenizer_evaluator.py --config ${CONFIG}

echo ""
echo "Step 3: Analyzing frequency on Indic dataset..."
python frequency_analyzer.py \
  --tokenizer ${TOKENIZER} \
  --dataset indic \
  --config ${CONFIG} \
  --output ${RESULTS_DIR}/frequency_stats/${TOKENIZER}_indic_freq.json

echo ""
echo "Step 4: Analyzing frequency on Code dataset..."
python frequency_analyzer.py \
  --tokenizer ${TOKENIZER} \
  --dataset code \
  --config ${CONFIG} \
  --output ${RESULTS_DIR}/frequency_stats/${TOKENIZER}_code_freq.json

echo ""
echo "Step 5: Merging frequency statistics..."
# Note: You may need to implement merge logic or use Python script
# For now, we'll use just the indic stats as primary
FREQ_STATS="${RESULTS_DIR}/frequency_stats/${TOKENIZER}_indic_freq.json"

echo ""
echo "Step 6: Reindexing tokenizer..."
python id_reindexer.py \
  --tokenizer ${TOKENIZER} \
  --frequency-stats ${FREQ_STATS} \
  --config ${CONFIG} \
  --output ${RESULTS_DIR}/reindexed_tokenizers/${TOKENIZER}_reindexed/

echo ""
echo "Step 7: Validating reindexed tokenizer..."
python validation_suite.py \
  --original ${TOKENIZER} \
  --reindexed ${RESULTS_DIR}/reindexed_tokenizers/${TOKENIZER}_reindexed/ \
  --config ${CONFIG} \
  --output ${RESULTS_DIR}/validation_report_${TOKENIZER}.json

echo ""
echo "=================================================="
echo "Workflow Complete!"
echo "=================================================="
echo ""
echo "Results:"
echo "  - Evaluation: ${RESULTS_DIR}/evaluation_results.json"
echo "  - Frequency stats: ${RESULTS_DIR}/frequency_stats/"
echo "  - Reindexed tokenizer: ${RESULTS_DIR}/reindexed_tokenizers/${TOKENIZER}_reindexed/"
echo "  - Validation report: ${RESULTS_DIR}/validation_report_${TOKENIZER}.json"
echo ""
echo "Next steps:"
echo "  1. Review validation report to ensure all tests passed"
echo "  2. Check ID_SCHEME.md for understanding ID allocation"
echo "  3. Use reindexed tokenizer in your training pipeline"
echo ""
