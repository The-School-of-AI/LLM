#!/bin/bash
# Test Iteration 2: Disable Gradient Clipping to Measure Sync Impact
# Expected: cudaStreamSynchronize should drop from 60% to ~15-20%

echo "🔬 Testing Iteration 2: No Gradient Clipping"
echo "=============================================="
echo ""
echo "Hypothesis: Gradient clipping causes 60% sync overhead"
echo "Expected: Sync time drops from 1973ms (60%) to ~400-600ms (15-20%)"
echo ""

cd "$(dirname "$0")"

# Profile with gradient clipping disabled
nsys profile \
    --capture-range=cudaProfilerApi \
    --output=iter2_no_grad_clip \
    --force-overwrite=true \
    python train_nsys.py \
        --profile-steps 10-20 \
        --max-steps 100 \
        --gradient-clip 0 \
        --batch-size 2 \
        --gradient-accumulation 4

echo ""
echo "✅ Profiling complete!"
echo ""
echo "📊 Analyze with:"
echo "   nsys stats iter2_no_grad_clip.nsys-rep"
echo ""
echo "🔍 Compare cudaStreamSynchronize:"
echo "   Baseline:    1990ms (60.7%)"
echo "   Iteration 1: 1973ms (60.0%)"  
echo "   Iteration 2: ???ms (???%)"
