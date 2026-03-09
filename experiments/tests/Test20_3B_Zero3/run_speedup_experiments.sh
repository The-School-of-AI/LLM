#!/bin/bash
# ============================================================================
# Run all 6 speedup experiments for 70B training on A100 at seq 4096
# ============================================================================
#
# Prerequisites:
#   - A100 GPU(s) available
#   - Python environment with triton, torch, deepspeed installed
#   - Data shards at /mnt/local-nvme/LLM/experiments/tests/Test17_3B_Zero3/data/wikitext_shards/
#
# Usage:
#   # Run all kernel-level benchmarks (Exp 1, 2, 3, 5, 6 - single GPU)
#   bash run_speedup_experiments.sh bench
#
#   # Run individual experiments (multi-GPU training tests)
#   bash run_speedup_experiments.sh exp2     # torch.compile
#   bash run_speedup_experiments.sh exp4a    # ZeRO-3 large buckets
#   bash run_speedup_experiments.sh exp4b    # ZeRO-3 large persistence
#   bash run_speedup_experiments.sh exp6     # cleanup frequency test
#   bash run_speedup_experiments.sh baseline # baseline (no changes)
# ============================================================================

set -e
cd "$(dirname "$0")/code"

case "${1:-bench}" in
  bench)
    echo "=== Running kernel-level benchmarks (Experiments 1, 2, 3, 5, 6) ==="
    python tests/bench_speedup_experiments.py
    ;;

  exp2)
    echo "=== Experiment 2: torch.compile (reduce-overhead mode) ==="
    deepspeed main.py --config ../configs/test_70b_exp2_compile_1024_bs8_10steps.yaml
    ;;

  exp4a)
    echo "=== Experiment 4a: ZeRO-3 large buckets (200M) ==="
    deepspeed main.py --config ../configs/test_70b_exp4_large_buckets_1024_bs8_10steps.yaml
    ;;

  exp4b)
    echo "=== Experiment 4b: ZeRO-3 large persistence threshold (10M) ==="
    deepspeed main.py --config ../configs/test_70b_exp4_large_persistence_1024_bs8_10steps.yaml
    ;;

  exp6)
    echo "=== Experiment 6: Reduced cleanup frequency (every 10 steps) ==="
    T19_CLEANUP_EVERY_N=10 \
    T19_STEP_CUDA_SYNC=0 \
      deepspeed main.py --config ../configs/test_70b_moe_lora_1024_bs8_10steps.yaml
    ;;

  exp6-none)
    echo "=== Experiment 6: No cleanup at all ==="
    T19_STEP_GC_COLLECT=0 \
    T19_STEP_EMPTY_CACHE=0 \
    T19_STEP_CUDA_SYNC=0 \
      deepspeed main.py --config ../configs/test_70b_moe_lora_1024_bs8_10steps.yaml
    ;;

  baseline)
    echo "=== Baseline: no changes ==="
    deepspeed main.py --config ../configs/test_70b_moe_lora_1024_bs8_10steps.yaml
    ;;

  all)
    echo "=== Running ALL experiments sequentially ==="
    echo ""
    echo "--- Kernel benchmarks ---"
    python tests/bench_speedup_experiments.py
    echo ""
    echo "--- Baseline ---"
    deepspeed main.py --config ../configs/test_70b_moe_lora_1024_bs8_10steps.yaml 2>&1 | tee ../results/baseline.log
    echo ""
    echo "--- Exp 2: torch.compile ---"
    deepspeed main.py --config ../configs/test_70b_exp2_compile_1024_bs8_10steps.yaml 2>&1 | tee ../results/exp2_compile.log
    echo ""
    echo "--- Exp 4a: large buckets ---"
    deepspeed main.py --config ../configs/test_70b_exp4_large_buckets_1024_bs8_10steps.yaml 2>&1 | tee ../results/exp4a_large_buckets.log
    echo ""
    echo "--- Exp 4b: large persistence ---"
    deepspeed main.py --config ../configs/test_70b_exp4_large_persistence_1024_bs8_10steps.yaml 2>&1 | tee ../results/exp4b_large_persist.log
    echo ""
    echo "--- Exp 6: reduced cleanup ---"
    T19_CLEANUP_EVERY_N=10 T19_STEP_CUDA_SYNC=0 \
      deepspeed main.py --config ../configs/test_70b_moe_lora_1024_bs8_10steps.yaml 2>&1 | tee ../results/exp6_reduced_cleanup.log
    echo ""
    echo "=== All experiments complete. Compare tok/s in the log files. ==="
    ;;

  *)
    echo "Usage: $0 {bench|exp2|exp4a|exp4b|exp6|exp6-none|baseline|all}"
    exit 1
    ;;
esac
