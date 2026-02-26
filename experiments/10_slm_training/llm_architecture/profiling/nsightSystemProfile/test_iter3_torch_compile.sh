#!/bin/bash

################################################################################
# Iteration 3: torch.compile + No Gradient Clipping - Nsight Systems Profile
################################################################################
# Purpose: Test combined optimizations for maximum performance:
#          - No gradient clipping (from Iteration 2)
#          - torch.compile graph optimization
#
# Expected Results:
#   - cudaStreamSynchronize: ~15-20% (from no gradient clipping)
#   - cudaLaunchKernel: Reduced kernel count (fusion)
#   - Total speedup: ~35-60% faster than baseline
#
# Output: iter3_torch_compile.nsys-rep (Nsight Systems report file)
#
# Note: First iteration will be slow (10-60s) due to model compilation
#
# Usage:
#   chmod +x test_iter3_torch_compile.sh
#   ./test_iter3_torch_compile.sh
################################################################################

# Configuration
PROFILE_NAME="iter3_torch_compile"
PRESET="1b-deepseek-gsa"
# Profile 20 steps for statistical significance
PROFILE_STEPS="10-15"  # 20 steps to capture multiple iterations

# Training hyperparameters (optimized for max performance)
MAX_STEPS=100
BATCH_SIZE=1
GRADIENT_ACCUMULATION=4
GRADIENT_CLIP=0          # ⭐ DISABLED - From Iteration 2
USE_TORCH_COMPILE=true   # ⭐ ENABLED - Graph optimization
TORCH_COMPILE_MODE="reduce-overhead"  # Optimized for throughput
SEQ_LENGTH=512
LEARNING_RATE=3e-4
EXPERIMENT_NAME="profiling_${PROFILE_NAME}"

echo "============================================================"
echo "Iteration 3: torch.compile + No Gradient Clipping"
echo "============================================================"
echo "Optimizations Applied:"
echo "  ✅ GPU loss accumulation (Iter 1)"
echo "  ✅ Non-blocking transfers (Iter 1)"
echo "  ✅ zero_grad(set_to_none) (Iter 1)"
echo "  ✅ No gradient clipping (Iter 2)"
echo "  ✅ torch.compile (Iter 3) ⭐ NEW"
echo "============================================================"
echo "Expected Combined Speedup: ~35-60% vs Baseline"
echo "============================================================"
echo "Profile Name: ${PROFILE_NAME}"
echo "Profile Steps: ${PROFILE_STEPS}"
echo "Training Configuration:"
echo "  - Preset: ${PRESET}"
echo "  - Max Steps: ${MAX_STEPS}"
echo "  - Batch Size: ${BATCH_SIZE}"
echo "  - Gradient Accumulation: ${GRADIENT_ACCUMULATION}"
echo "  - Gradient Clip: ${GRADIENT_CLIP} (disabled)"
echo "  - torch.compile: ENABLED (${TORCH_COMPILE_MODE})"
echo "  - Sequence Length: ${SEQ_LENGTH}"
echo "  - Learning Rate: ${LEARNING_RATE}"
echo "============================================================"
echo "⚠️  Note: First iteration will be slow (compilation time)"
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
    --use-torch-compile \
    --torch-compile-mode ${TORCH_COMPILE_MODE} \
    --seq-length ${SEQ_LENGTH} \
    --learning-rate ${LEARNING_RATE} \
    --experiment-name ${EXPERIMENT_NAME} \
    --profile-steps ${PROFILE_STEPS}

echo
echo "============================================================"
echo "Iteration 3 Profiling Complete!"
echo "============================================================"
echo "Output files in: profiling/nsightSystemProfile/"
echo "  - ${PROFILE_NAME}.nsys-rep"
echo "  - ${PROFILE_NAME}.sqlite"
echo
echo "Performance Progression:"
echo "  Baseline     (Iter 0): 100% tokens/sec, 60.7% sync"
echo "  Optimized    (Iter 1): ~105% tokens/sec, 60.0% sync"
echo "  No Grad Clip (Iter 2): ~125% tokens/sec, ~15% sync (expected)"
echo "  + Compiled   (Iter 3): ~150% tokens/sec, ~15% sync (expected)"
echo
echo "Next Steps:"
echo "1. Analyze overall improvements:"
echo "   cd profiling/nsightSystemProfile"
echo "   nsys stats ${PROFILE_NAME}.nsys-rep"
echo
echo "2. Compare kernel launches (should be reduced due to fusion):"
echo "   nsys stats ${PROFILE_NAME}.nsys-rep | grep -A 5 'cudaLaunchKernel'"
echo
echo "3. Verify sync time remains low:"
echo "   nsys stats ${PROFILE_NAME}.nsys-rep | grep 'cudaStreamSynchronize'"
echo
echo "4. Measure actual tokens/sec improvement in training logs"
echo "============================================================"
