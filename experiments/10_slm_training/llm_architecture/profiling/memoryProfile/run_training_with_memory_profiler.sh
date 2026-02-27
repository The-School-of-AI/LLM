#!/bin/bash

# Script to run training with memory profiling enabled
# This script runs the LLM training with PyTorch profiler enabled

#set -e  # Exit on error

# Default values
PRESET="${PRESET:-1b-gsa}"
MAX_STEPS="${MAX_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-4}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-1b-gsa-training}"
PROFILING_OUTPUT_DIR="${PROFILING_OUTPUT_DIR:-logs/}"
PROFILING_ACTIVE_STEPS="${PROFILING_ACTIVE_STEPS:-20}"
PROFILING_WAIT_STEPS="${PROFILING_WAIT_STEPS:-10}"
PROFILING_WARMUP_STEPS="${PROFILING_WARMUP_STEPS:-10}"
PROFILING_REPEAT="${PROFILING_REPEAT:-10}"

# Print configuration
echo "========================================="
echo "Running Training with Profiler"
echo "========================================="
echo "Preset: $PRESET"
echo "Max Steps: $MAX_STEPS"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation: $GRADIENT_ACCUMULATION"
echo "Learning Rate: $LEARNING_RATE"
echo "Experiment Name: $EXPERIMENT_NAME"
echo "Profiling Output Dir: $PROFILING_OUTPUT_DIR"
echo "Profiling Active Steps: $PROFILING_ACTIVE_STEPS"
echo "Profiling Wait Steps: $PROFILING_WAIT_STEPS"
echo "Profiling Warmup Steps: $PROFILING_WARMUP_STEPS"
echo "Profiling Repeat: $PROFILING_REPEAT"
echo "========================================="
echo ""

# Change to project root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Script directory: $SCRIPT_DIR"
echo "Changing to project root: $PROJECT_ROOT"
if ! cd "$PROJECT_ROOT"; then
    echo "❌ Error: Could not change to project root directory"
    exit 1
fi

echo ""
echo "Running profiling from: $(pwd)"
echo ""

# Run training with profiling
uv run python ${SCRIPT_DIR}/train_with_memory_profiler.py \
    --preset "$PRESET" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation "$GRADIENT_ACCUMULATION" \
    --learning-rate "$LEARNING_RATE" \
    --experiment-name "$EXPERIMENT_NAME" \
    --enable-profiling \
    --profiling-output-dir "$PROFILING_OUTPUT_DIR" \
    --profiling-active-steps "$PROFILING_ACTIVE_STEPS" \
    --profiling-wait-steps "$PROFILING_WAIT_STEPS" \
    --profiling-warmup-steps "$PROFILING_WARMUP_STEPS" \
    --profiling-repeat "$PROFILING_REPEAT"

echo ""
echo "========================================="
echo "Training with profiling completed!"
echo "Profiling results saved to: $PROFILING_OUTPUT_DIR"
echo "========================================="
