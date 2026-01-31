#!/bin/bash
# Run HuggingFace dataset-based tokenizer evaluation

set -e

echo "=========================================="
echo "HuggingFace Dataset Tokenizer Evaluation"
echo "=========================================="
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if dependencies are installed
echo "Checking dependencies..."
python -c "import datasets, tqdm, loguru, yaml" 2>/dev/null || {
    echo "Missing dependencies. Installing..."
    pip install -r requirements.txt
}

echo ""
echo "Starting evaluation on HuggingFace datasets..."
echo "  - Indic: ai4bharat/IndicCorpV2 (10,000 samples)"
echo "  - Code: allenai/dolma and dolma3_dolmino_mix (5,000 each, 5% sampling)"
echo ""
echo "This may take 1-2 hours depending on your internet speed and CPU."
echo ""

# Run the evaluator
cd src
python tokenizer_evaluator_hf.py --config ../config.yaml

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: results/evaluation_results_hf.json"
echo "=========================================="
