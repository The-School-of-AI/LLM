#!/bin/bash

################################################################################
# Iteration 2: Disable Gradient Clipping - Nsight Systems Profile
################################################################################
# Purpose: Test hypothesis that gradient clipping causes 60% sync overhead
#          by profiling training WITHOUT gradient clipping
#
# Expected Results:
#   - cudaStreamSynchronize: 60% → ~15-20% (drop of 40 percentage points)
#   - Sync time: 1973ms → ~400-600ms
#   - Throughput: +20-30% tokens/sec
#
# Output: iter2_no_grad_clip.nsys-rep (Nsight Systems report file)
#
# Usage:
#   chmod +x test_iter2_no_grad_clip.sh
#   ./test_iter2_no_grad_clip.sh
################################################################################

# Configuration
PROFILE_NAME="iter2_no_grad_clip"
PRESET="1b-deepseek-gsa"
# Profile 20 steps for statistical significance
PROFILE_STEPS="10-15"  # 20 steps to capture multiple iterations

# Training hyperparameters (optimized for profiling)
MAX_STEPS=100
BATCH_SIZE=1
GRADIENT_ACCUMULATION=4
GRADIENT_CLIP=0          # ⭐ DISABLED - Testing sync impact
SEQ_LENGTH=512
LEARNING_RATE=3e-4
EXPERIMENT_NAME="profiling_${PROFILE_NAME}"

echo "============================================================"
echo "Iteration 2: Testing Without Gradient Clipping"
echo "============================================================"
echo "Hypothesis: Gradient clipping causes 60% sync overhead"
echo "Expected: Sync drops from 60% to ~15-20%"
echo "============================================================"
echo "Profile Name: ${PROFILE_NAME}"
echo "Profile Steps: ${PROFILE_STEPS}"
echo "Training Configuration:"
echo "  - Preset: ${PRESET}"
echo "  - Max Steps: ${MAX_STEPS}"
echo "  - Batch Size: ${BATCH_SIZE}"
echo "  - Gradient Accumulation: ${GRADIENT_ACCUMULATION}"
echo "  - Gradient Clip: ${GRADIENT_CLIP} ⭐ DISABLED"
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
    --gradient-clip ${GRADIENT_CLIP} \
    --seq-length ${SEQ_LENGTH} \
    --learning-rate ${LEARNING_RATE} \
    --experiment-name ${EXPERIMENT_NAME} \
    --profile-steps ${PROFILE_STEPS}

echo
echo "============================================================"
echo "Iteration 2 Profiling Complete!"
echo "============================================================"
echo "Output files in: profiling/nsightSystemProfile/"
echo "  - ${PROFILE_NAME}.nsys-rep"
echo "  - ${PROFILE_NAME}.sqlite"
echo
echo "Compare Results:"
echo "  Baseline:    cudaStreamSynchronize 60.7% (1990ms)"
echo "  Iteration 1: cudaStreamSynchronize 60.0% (1973ms)"
echo "  Iteration 2: cudaStreamSynchronize ???% (???ms)"
echo
echo "Next Steps:"
echo "1. Analyze sync reduction:"
echo "   cd profiling/nsightSystemProfile"
echo "   nsys stats ${PROFILE_NAME}.nsys-rep | grep -A 10 'cuda_api_sum'"
echo
echo "2. If sync dropped to ~15-20%:"
echo "   ✅ Confirmed: Gradient clipping was the 60% sync bottleneck!"
echo "   → Proceed to Iteration 3 (torch.compile)"
echo
echo "3. If sync still high (~50%+):"
echo "   ⚠️  Need deeper investigation into other sync sources"
echo "============================================================"
