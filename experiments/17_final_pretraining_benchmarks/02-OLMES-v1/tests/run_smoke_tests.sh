#!/bin/bash
# run_smoke_tests.sh
# Purpose: Execute a full "smoke test" by running every training stage with a small sample limit.
#
# Usage:
#   tests/run_smoke_tests.sh [OPTIONS]
#
# Options:
#   -c, --config    Path to benchmark config YAML  (default: 02-OLMES-v1/configs/benchmark-config.yaml)
#   -m, --model     HuggingFace model name          (default: HuggingFaceTB/SmolLM2-135M)
#   -s, --stages    Comma-separated list of stages  (default: pretrain_1b,pretrain_3b,pretrain_8b,pretrain_70b,sft,ci_breadth)
#   -d, --device    Execution device                (default: cpu)
#   -b, --batch-size Batch size for benchmarks       (default: 2)
#   -t, --hf-token  HuggingFace API token            (for gated datasets; can also use env var HF_TOKEN)
#   -h, --help      Show this help message

# Defaults
CONFIG="configs/benchmark-config.yaml"
MODEL="HuggingFaceTB/SmolLM2-135M"
STAGES_STR="pretrain_1b,pretrain_3b,pretrain_8b,pretrain_70b,sft,ci_breadth"
DEVICE="cpu"
BATCH_SIZE="2"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -s|--stages)
            STAGES_STR="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -t|--hf-token)
            export HF_TOKEN="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --config    Path to benchmark config YAML  (default: configs/benchmark-config.yaml)"
            echo "  -m, --model     HuggingFace model name          (default: HuggingFaceTB/SmolLM2-135M)"
            echo "  -s, --stages    Comma-separated list of stages  (default: pretrain_1b,pretrain_3b,...,ci_breadth)"
            echo "  -d, --device    Execution device                (default: cpu)"
            echo "  -b, --batch-size Batch size                      (default: 2)"
            echo "  -t, --hf-token  HuggingFace API token            (for gated datasets)"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information."
            exit 1
            ;;
    esac
done

# Convert comma-separated stages to array
IFS=',' read -r -a STAGES <<< "$STAGES_STR"

echo "=========================================="
echo "🚀 Starting OLMES Smoke Test Suite"
echo "Config: $CONFIG"
echo "Model:  $MODEL"
echo "Device: $DEVICE"
echo "Batch Size: $BATCH_SIZE"
echo "Stages: ${STAGES[*]}"
echo "Limit:  2 samples per task"
if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN: ✅ Set"
else
    echo "HF_TOKEN: ❌ Not set (gated datasets will fail)"
fi
echo "=========================================="

for STAGE in "${STAGES[@]}"; do
    echo ""
    echo "--- Scaling to Stage: $STAGE ---"
    .venv/bin/python3 src/pipeline_runner.py \
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
echo "🎉 Smoke Test Suite Finished"
echo "Check benchmark-results/ for results and reports."
echo "=========================================="
