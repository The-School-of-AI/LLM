#!/bin/bash

################################################################################
# Nsight Compute Full Kernel Profiling Script for 1B LLM Training Pipeline
################################################################################
# Purpose: Deep-dive into individual kernel performance to identify compute
#          vs memory bottlenecks, occupancy issues, and optimization opportunities
#
# WARNING: This profiles ALL kernels and is VERY SLOW. Use for detailed analysis only.
#
# Output: seed_1b_kernels.ncu-rep (Nsight Compute report file)
#
# Usage:
#   chmod +x profile_all_kernels.sh
#   ./profile_all_kernels.sh
################################################################################

# Configuration
PROFILE_NAME="seed_1b_kernels"
PRESET="1b-gsa"
# Note: ncu profiles ALL kernels during the entire run
# We keep training steps low (5) to make profiling faster
PROFILE_STEPS="10-12"  # Not used by ncu - just kept for documentation

# Training hyperparameters (optimized for Tesla T4 14GB)
MAX_STEPS=12
BATCH_SIZE=6
GRADIENT_ACCUMULATION=1
SEQ_LENGTH=2048
LEARNING_RATE=3e-4
EXPERIMENT_NAME="kernel_profiling_${PROFILE_NAME}"

echo "============================================================"
echo "Starting Nsight Compute Full Kernel Profiling"
echo "============================================================"
echo "⚠️  WARNING: Full kernel profiling is VERY SLOW!"
echo "⚠️  Profiling ${PROFILE_STEPS} (5 steps for statistical averaging)"
echo "⚠️  Expected runtime: 45-90 minutes"
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

# Check if ncu is installed
if ! command -v ncu &> /dev/null; then
    echo "============================================================"
    echo "❌ ERROR: Nsight Compute (ncu) not found"
    echo "============================================================"
    echo ""
    echo "Please install Nsight Compute first:"
    echo ""
    echo "  sudo apt-get install -y nsight-compute-2025.4.1"
    echo ""
    echo "Or run the installation script:"
    echo "  ./install_profiling_tools.sh"
    echo ""
    echo "After installation, you may need to add ncu to your PATH:"
    echo "  export PATH=\$PATH:/opt/nvidia/nsight-compute/2025.4.1"
    echo "============================================================"
    return 1 2>/dev/null || exit 1
fi

echo "✅ Found ncu: $(which ncu)"
echo ""


# Output to the kernelProfile directory
OUTPUT_DIR="$SCRIPT_DIR"
OUTPUT_PATH="${OUTPUT_DIR}/${PROFILE_NAME}"

# Activate the virtual environment
source "${PROJECT_ROOT}/.venv/bin/activate"

# Run ncu with full analysis sections
# NOTE: sudo is required because RmProfilingAdminOnly=1 restricts GPU performance counters
ncu \
  -f \
  --set full \
  --target-processes all \
  --import-source yes \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section ComputeWorkloadAnalysis \
  --section Occupancy \
  -o ${OUTPUT_PATH} \
  ${PROJECT_ROOT}/.venv/bin/python ${SCRIPT_DIR}/train_ncu.py \
    --preset ${PRESET} \
    --max-steps ${MAX_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --gradient-accumulation ${GRADIENT_ACCUMULATION} \
    --seq-length ${SEQ_LENGTH} \
    --learning-rate ${LEARNING_RATE} \
    --experiment-name ${EXPERIMENT_NAME}

echo
echo "============================================================"
echo "Profiling Complete!"
echo "============================================================"
echo "Output files in: profiling/kernelProfile/"
echo "  - ${PROFILE_NAME}.ncu-rep"
echo
echo "Next Steps:"
echo "1. Export kernel metrics to CSV:"
echo "   cd profiling/kernelProfile"
echo "   ncu --import ${PROFILE_NAME}.ncu-rep --csv > kernel_metrics.csv"
echo
echo "2. Transfer to local machine for GUI analysis:"
echo "   scp user@remote:$(pwd)/${PROFILE_NAME}.ncu-rep ."
echo
echo "3. Open in Nsight Compute GUI (on local machine):"
echo "   - Launch Nsight Compute"
echo "   - File > Open > ${PROFILE_NAME}.ncu-rep"
echo
echo "4. Analyze kernels:"
echo "   - Sort by duration to find top kernels"
echo "   - Check Speed of Light metrics"
echo "   - Identify compute vs memory bottlenecks"
echo "   - Review occupancy and warp stall reasons"
echo "============================================================"
