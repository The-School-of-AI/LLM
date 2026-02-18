#!/bin/bash
# Quick start script for training with reversible model

echo "=================================================="
echo "Reversible Model Training with DeepSpeed"
echo "=================================================="
echo ""
echo "This script trains a memory-efficient reversible LLM"
echo "based on arXiv:2512.02056v2 (Dec 2024)"
echo ""
echo "Key features:"
echo "- ~10x memory reduction for activations"
echo "- Larger batch sizes possible"
echo "- No activation checkpointing needed"
echo "- Comparable or better performance"
echo ""
echo "=================================================="
echo ""

# Check if DeepSpeed is installed
if ! command -v deepspeed &> /dev/null; then
    echo "ERROR: DeepSpeed is not installed"
    echo "Install with: pip install deepspeed"
    exit 1
fi

# Default values
NUM_GPUS=1
CONFIG="config_reversible.yaml"
EXTRA_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

echo "Configuration:"
echo "  Number of GPUs: $NUM_GPUS"
echo "  Config file: $CONFIG"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    echo "Please create it or use --config to specify a different file"
    exit 1
fi

echo "Starting training..."
echo ""

# Run DeepSpeed training
if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU
    deepspeed main.py --config "$CONFIG" $EXTRA_ARGS
else
    # Multi-GPU
    deepspeed --num_gpus=$NUM_GPUS main.py --config "$CONFIG" $EXTRA_ARGS
fi

echo ""
echo "=================================================="
echo "Training completed!"
echo "=================================================="
