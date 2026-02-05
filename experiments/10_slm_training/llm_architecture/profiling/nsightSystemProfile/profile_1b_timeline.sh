#!/bin/bash

################################################################################
# Nsight Systems Profiling Script for 1B LLM Training Pipeline
################################################################################
# Purpose: Profile training steps 10-20 to identify bottlenecks, gaps, stalls,
#          and overlap issues in GPU utilization
#
# Output: seed_1b_timeline.nsys-rep (Nsight Systems report file)
#
# Usage:
#   chmod +x profile_1b_timeline.sh
#   ./profile_1b_timeline.sh
################################################################################

# Configuration
PROFILE_NAME="seed_1b_timeline"
PRESET="1b-deepseek-gsa"
# Profile 20 steps for statistical significance and pattern identification
PROFILE_STEPS="10-15"  # 20 steps to capture multiple iterations and average performance

#Training hyperparameters (optimized for Tesla T4 14GB)
MAX_STEPS=100
BATCH_SIZE=1
GRADIENT_ACCUMULATION=1
SEQ_LENGTH=512
LEARNING_RATE=3e-4
EXPERIMENT_NAME="profiling_${PROFILE_NAME}"

echo "============================================================"
echo "Starting Nsight Systems Profiling"
echo "============================================================"
echo "Profiling ${PROFILE_STEPS} (20 steps for statistical averaging)"
echo "Expected runtime: ~5-8 minutes with profiling overhead"
echo "============================================================"
echo "Profile Name: ${PROFILE_NAME}"
echo "Profile Steps: ${PROFILE_STEPS}"
echo "Training Configuration:"
echo "  - Preset: ${PRESET}"
echo "  - Max Steps: ${MAX_STEPS}"
echo "  - Batch Size: ${BATCH_SIZE}"
echo "  - Gradient Accumulation: ${GRADIENT_ACCUMULATION}"
echo "  - Sequence Length: ${SEQ_LENGTH}"
echo "  - Learning Rate: ${LEARNING_RATE}"
echo "============================================================"
echo

# Change to project root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Changing to project root: $PROJECT_ROOT"
if ! cd "$PROJECT_ROOT"; then
    echo "❌ Error: Could not change to project root directory"
    return 1 2>/dev/null || true
fi

echo ""
echo "Running profiling from: $(pwd)"
echo ""

# Run nsys profile with comprehensive tracing
# Note: NCCL tracing removed - not supported in Nsight Systems 2025.5.2
# For multi-GPU analysis, use MPI trace instead

# Output to the nsightSystemProfile directory
OUTPUT_DIR="$SCRIPT_DIR"
OUTPUT_PATH="${OUTPUT_DIR}/${PROFILE_NAME}"

nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas,mpi \
  --cuda-memory-usage=true \
  --gpu-metrics-devices=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --stats=true \
  -o ${OUTPUT_PATH} \
  uv run python ${SCRIPT_DIR}/train_nsys.py \
    --preset ${PRESET} \
    --max-steps ${MAX_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --gradient-accumulation ${GRADIENT_ACCUMULATION} \
    --seq-length ${SEQ_LENGTH} \
    --learning-rate ${LEARNING_RATE} \
    --experiment-name ${EXPERIMENT_NAME} \
    --profile-steps ${PROFILE_STEPS}

echo
echo "============================================================"
echo "Profiling Complete!"
echo "============================================================"
echo "Output files in: profiling/nsightSystemProfile/"
echo "  - ${PROFILE_NAME}.nsys-rep"
echo "  - ${PROFILE_NAME}.sqlite"
echo
echo "Next Steps:"
echo "1. Export summary statistics:"
echo "   cd profiling/nsightSystemProfile"
echo "   nsys stats ${PROFILE_NAME}.nsys-rep"
echo
echo "2. Transfer to local machine for GUI analysis:"
echo "   scp user@remote:$(pwd)/${PROFILE_NAME}.nsys-rep ."
echo
echo "3. Open in Nsight Systems GUI (on local machine):"
echo "   - Launch Nsight Systems"
echo "   - File > Open > ${PROFILE_NAME}.nsys-rep"
echo "============================================================"
