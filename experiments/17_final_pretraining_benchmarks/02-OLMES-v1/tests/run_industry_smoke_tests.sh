#!/bin/bash
# run_industry_smoke_tests.sh
# Purpose: Execute a smoke test for the INDUSTRY pipeline with MPS and Batch Size 16.

CONFIG="02-OLMES-v1/configs/industry-benchmarks.yaml"
MODEL="HuggingFaceTB/SmolLM2-135M"
STAGES=("pretrain_small" "pretrain_8b" "pretrain_70b" "sft")
DEVICE="mps"
BATCH_SIZE="16"

echo "=========================================="
echo "🚀 Starting OLMES INDUSTRY Smoke Test Suite"
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Batch Size: $BATCH_SIZE"
echo "Limit: 2 samples per task"
echo "=========================================="

for STAGE in "${STAGES[@]}"; do
    echo ""
    echo "--- Scaling to Industry Stage: $STAGE ---"
    # Ensure we use the local .venv if it exists, fallback to python3
    PYTHON_EXEC="python3"
    if [ -f ".venv/bin/python3" ]; then
        PYTHON_EXEC=".venv/bin/python3"
    fi

    $PYTHON_EXEC 02-OLMES-v1/src/pipeline_runner.py \
        --config "$CONFIG" \
        --stage "$STAGE" \
        --model_args "pretrained=$MODEL" \
        --device "$DEVICE" \
        --batch_size "$BATCH_SIZE" \
        --sample
    
    if [ $? -ne 0 ]; then
        echo "❌ Stage $STAGE failed!"
    else
        echo "✅ Stage $STAGE complete."
    fi
done

echo ""
echo "=========================================="
echo "🎉 Industry Smoke Test Suite Finished"
echo "Check benchmark-results/ for results and reports."
echo "=========================================="
