#!/bin/bash
# Test Iteration 3: Enable torch.compile for Additional Speedup
# Expected: +15-30% throughput from kernel fusion and graph optimization
# Note: First iteration will be slow due to compilation

echo "🔬 Testing Iteration 3: torch.compile"
echo "======================================"
echo ""
echo "Goal: Graph optimization for additional 15-30% speedup"
echo "Note: First iteration ~10-60s slower (model compilation)"
echo ""

cd "$(dirname "$0")"

# Test with torch.compile (and no gradient clipping for max speedup)
nsys profile \
    --capture-range=cudaProfilerApi \
    --output=iter3_torch_compile \
    --force-overwrite=true \
    python train_nsys.py \
        --profile-steps 10-20 \
        --max-steps 100 \
        --gradient-clip 0 \
        --use-torch-compile \
        --torch-compile-mode reduce-overhead \
        --batch-size 2 \
        --gradient-accumulation 4

echo ""
echo "✅ Profiling complete!"
echo ""
echo "📊 Analyze with:"
echo "   nsys stats iter3_torch_compile.nsys-rep"
echo ""
echo "🎯 Expected Combined Improvements:"
echo "   Iter 1 (GPU accum, non-blocking, etc.):  ~3-7%"
echo "   Iter 2 (no gradient clipping):           +20-30%"
echo "   Iter 3 (torch.compile):                  +15-30%"
echo "   ================================================"
echo "   TOTAL:                                   ~35-60% faster!"
