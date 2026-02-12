#!/bin/bash
# run_smoke_tests.sh
# Purpose: Execute a full "smoke test" by running every training stage with a small sample limit.

CONFIG="02-OLMES-v1/configs/benchmark-config.yaml"
MODEL="HuggingFaceTB/SmolLM2-135M"
STAGES=("pretrain_1b" "pretrain_3b" "pretrain_8b" "pretrain_70b" "sft" "ci_breadth")

echo "=========================================="
echo "🚀 Starting OLMES Smoke Test Suite"
echo "Model: $MODEL"
echo "Limit: 2 samples per task"
echo "=========================================="

for STAGE in "${STAGES[@]}"; do
    echo ""
    echo "--- Scaling to Stage: $STAGE ---"
    .venv/bin/python3 02-OLMES-v1/src/pipeline_runner.py \
        --config "$CONFIG" \
        --stage "$STAGE" \
        --model_args "pretrained=$MODEL" \
        --device "cpu" \
        --sample
    
    if [ $? -ne 0 ]; then
        echo "❌ Stage $STAGE failed!"
    else
        echo "✅ Stage $STAGE complete."
    fi
done

echo ""
echo "=========================================="
echo "🎉 Smoke Test Suite Finished"
echo "Check benchmark-results/ for results and reports."
echo "=========================================="
