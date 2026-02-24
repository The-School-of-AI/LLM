# Optimization Benchmark

Profiles the 1B reversible model across 6 optimization axes and reports tokens/sec improvement.

## Files

| File | Description |
|------|-------------|
| `triton_optimizations.py` | Optimized Triton kernels (fused RoPE, SiLU*mul, RMSNorm+SwishGate, autotuned RMSNorm, batched MHCCoeffs) |
| `benchmark_optimizations.py` | Benchmark runner: micro-benchmarks per kernel + macro-benchmark (full fwd+bwd tokens/sec) |
| `nsys_profile.py` | Nsight Systems profiler: generates `.nsys-rep` files for baseline vs optimized side-by-side comparison |
| `ncu_profile.py` | Nsight Compute profiler: generates `.ncu-rep` files with per-kernel hardware counters (occupancy, throughput, stalls) |

## Prerequisites

```bash
cd experiments/tests/Test_14_gsa_only_liger_kernels_1000steps-OngoingRun3
uv sync   # ensure .venv/ has all deps
```

The init model must exist at `results/init/model_init.pt`. If not, run `./run.sh` first (or `FORCE_REWRITE_INIT=1 ./run.sh`).

## Quick Start

```bash
cd experiments/tests/Test_14_gsa_only_liger_kernels_1000steps-OngoingRun3

# Full benchmark (micro + macro)
PYTHONPATH="/workspace/LLM:${PYTHONPATH:-}" \
  .venv/bin/python scripts/benchmark_optimizations.py \
  --config configs/test14_gsa_only_liger_kernels_1000steps.yaml
```

## Run Options

### Micro-benchmarks only (no model load, fast)

```bash
PYTHONPATH="/workspace/LLM:${PYTHONPATH:-}" \
  .venv/bin/python scripts/benchmark_optimizations.py \
  --config configs/test14_gsa_only_liger_kernels_1000steps.yaml \
  --micro-only
```

### Macro-benchmark only (full model fwd+bwd)

```bash
PYTHONPATH="/workspace/LLM:${PYTHONPATH:-}" \
  .venv/bin/python scripts/benchmark_optimizations.py \
  --config configs/test14_gsa_only_liger_kernels_1000steps.yaml \
  --macro-only
```

### Override batch size

```bash
# Use smaller batch if GPU has less VRAM
PYTHONPATH="/workspace/LLM:${PYTHONPATH:-}" \
  .venv/bin/python scripts/benchmark_optimizations.py \
  --config configs/test14_gsa_only_liger_kernels_1000steps.yaml \
  --batch-size 32
```

### Control iterations

```bash
# More iterations for stable measurements
PYTHONPATH="/workspace/LLM:${PYTHONPATH:-}" \
  .venv/bin/python scripts/benchmark_optimizations.py \
  --config configs/test14_gsa_only_liger_kernels_1000steps.yaml \
  --warmup 10 --iters 50
```

### torch.compile mode

```bash
# Options: default, reduce-overhead (default), max-autotune
PYTHONPATH="/workspace/LLM:${PYTHONPATH:-}" \
  .venv/bin/python scripts/benchmark_optimizations.py \
  --config configs/test14_gsa_only_liger_kernels_1000steps.yaml \
  --compile-mode max-autotune
```

## What It Measures

### Micro-benchmarks (per-kernel)

| # | Kernel | Optimization | Expected Speedup |
|---|--------|-------------|-----------------|
| 1 | RoPE rotation | 4+ PyTorch ops → 1 Triton kernel | 5-8x |
| 2 | SiLU * mul | 2 ops → 1 fused Triton kernel | ~1x (already fast) |
| 3 | RMSNorm+SwishGate | norm + silu + gate → 1 Triton kernel | 2-4x |
| 4 | RMSNorm autotune | @triton.autotune for warp/stage config | 1-1.1x |
| 5 | Batched MHCCoeffs | 3 small matmuls → 1 batched Linear | 10-25x |

### Macro-benchmark (full model)

Runs the full 1.65B parameter model forward + backward and reports:
- **ms/step**: wall-clock time per training step
- **tokens/sec**: `batch_size × seq_len / (ms/step / 1000)`
- **vs baseline**: percentage improvement

Configurations tested:
- **Baseline**: current implementation (no changes)
- **+ Kernel fusions**: all optimized Triton kernels applied
- **+ torch.compile**: graph capture + automatic fusion on top
- **Sync elimination**: impact estimate for removing unnecessary `cuda.synchronize()`

## 6 Optimization Axes

1. **Reduce CUDA launch kernels** — BatchedMHCCoeffs merges 3→1 cuBLAS calls (×16 per step = 32 fewer launches)
2. **Fuse small kernels** — RoPE, SiLU*mul, RMSNorm+SwishGate each replace multi-op PyTorch chains
3. **Reduce device-to-device GPU memory movement** — fused kernels eliminate intermediate tensor allocations
4. **Reduce CUDA synchronize** — remove redundant `cuda.synchronize()` after non_blocking data transfers
5. **torch.compile** — graph capture, operator fusion, reduced Python overhead
6. **Triton autotune** — finds optimal `(num_warps, num_stages)` per kernel per hardware

## Nsight Systems Profiling

Generate `.nsys-rep` files for visual inspection of CUDA kernel timelines, memory transfers, and NVTX-annotated forward/backward passes.

### Profile both (recommended)

Launches `nsys profile` automatically for baseline and optimized:

```bash
cd experiments/tests/Test_14_gsa_only_liger_kernels_1000steps-OngoingRun3

PYTHONPATH="/workspace/LLM:${PYTHONPATH:-}" \
  .venv/bin/python scripts/nsys_profile.py \
  --config configs/test14_gsa_only_liger_kernels_1000steps.yaml \
  --mode both
```

Generates:
- `results/nsys_baseline.nsys-rep`
- `results/nsys_optimized.nsys-rep`

### Profile individually (manual nsys)

```bash
# Baseline
nsys profile -t cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --force-overwrite true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    -o results/nsys_baseline \
    .venv/bin/python scripts/nsys_profile.py \
    --config configs/test14_gsa_only_liger_kernels_1000steps.yaml \
    --mode baseline

# Optimized
nsys profile -t cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --force-overwrite true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    -o results/nsys_optimized \
    .venv/bin/python scripts/nsys_profile.py \
    --config configs/test14_gsa_only_liger_kernels_1000steps.yaml \
    --mode optimized
```

### Nsys options

```bash
# Fewer profiled steps (faster)
--steps 3

# Smaller batch for memory-constrained GPU
--batch-size 32

# More warmup before profiling
--warmup 5
```

### Viewing results

```bash
# Open in Nsight Systems GUI
nsys-ui results/nsys_baseline.nsys-rep
nsys-ui results/nsys_optimized.nsys-rep

# Or generate text-based stats summary
nsys stats results/nsys_baseline.nsys-rep
nsys stats results/nsys_optimized.nsys-rep
```

### What to look for in the profiles

| Metric | Baseline vs Optimized |
|--------|----------------------|
| CUDA kernel count per step | Fewer launches (fused kernels + batched MHC) |
| Kernel duration | Shorter for fused ops (RoPE, RMSNorm+SwishGate) |
| Memory operations | Fewer D2D copies (eliminated intermediate tensors) |
| GPU idle gaps | Reduced (fewer kernel launches = less dispatch overhead) |
| NVTX ranges | `forward` / `backward` / `zero_grad` per step |

## Nsight Compute (ncu) Profiling

Per-kernel hardware counter analysis: occupancy, memory throughput, compute throughput, warp stalls, register pressure, cache hit rates.

### Prerequisites: GPU performance counter permissions

ncu requires access to NVIDIA hardware performance counters, which are restricted by default. Without this, you'll see:

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
```

**Fix (one-time, resets on reboot):**

```bash
sudo sh -c 'echo 1 > /proc/driver/nvidia/params/RmProfilingAdminOnly'
```

This sets the counter access mode to "admin+user" for the current boot session. Alternatively, run the full ncu command under `sudo`.

To make it persistent across reboots:

```bash
# Add to /etc/rc.local or a systemd oneshot service
sudo tee /etc/rc.local <<'EOF'
#!/bin/bash
echo 1 > /proc/driver/nvidia/params/RmProfilingAdminOnly
exit 0
EOF
sudo chmod +x /etc/rc.local
```

### Profile both (recommended)

```bash
cd experiments/tests/Test_14_gsa_only_liger_kernels_1000steps-OngoingRun3

PYTHONPATH="/workspace/LLM:${PYTHONPATH:-}" \
  .venv/bin/python scripts/ncu_profile.py \
  --config configs/test14_gsa_only_liger_kernels_1000steps.yaml \
  --mode both
```

Generates:
- `results/ncu_baseline.ncu-rep`
- `results/ncu_optimized.ncu-rep`

### Profile individually (manual ncu)

```bash
# Baseline
ncu --set full --replay-mode kernel \
    --target-processes all \
    --launch-skip-before-match 0 \
    --nvtx --nvtx-include "baseline/" \
    --force-overwrite \
    -o results/ncu_baseline \
    .venv/bin/python scripts/ncu_profile.py \
    --config configs/test14_gsa_only_liger_kernels_1000steps.yaml \
    --mode baseline

# Optimized
ncu --set full --replay-mode kernel \
    --target-processes all \
    --launch-skip-before-match 0 \
    --nvtx --nvtx-include "optimized/" \
    --force-overwrite \
    -o results/ncu_optimized \
    .venv/bin/python scripts/ncu_profile.py \
    --config configs/test14_gsa_only_liger_kernels_1000steps.yaml \
    --mode optimized
```

### ncu options

```bash
# Profile only specific kernels (faster)
--kernel-filter "fused_rope|rmsnorm|silu"

# Lighter metric set (faster collection)
--metrics basic

# Roofline analysis
--metrics roofline

# Source-level metrics (requires debug info)
--metrics source

# Smaller batch for faster profiling
--batch-size 4
```

### Viewing results

```bash
# Open in Nsight Compute GUI
ncu-ui results/ncu_baseline.ncu-rep
ncu-ui results/ncu_optimized.ncu-rep

# Side-by-side comparison
ncu-ui --page details results/ncu_baseline.ncu-rep results/ncu_optimized.ncu-rep
```

### What to look for in ncu profiles

| Metric | What it tells you |
|--------|------------------|
| SM [%] | Compute utilization — are SMs busy or stalled? |
| Memory [%] | Memory bandwidth utilization (% of peak HBM BW) |
| Achieved Occupancy | Active warps / max warps — scheduling efficiency |
| L1/L2 Hit Rate | Cache effectiveness — fused kernels should improve this |
| DRAM Throughput (GB/s) | Global memory bandwidth consumed |
| Registers/Thread | Register pressure — affects occupancy |
| Warp Stall Reasons | Why warps wait (memory, sync, compute) |

### nsys vs ncu: when to use which

| | nsys | ncu |
|---|------|-----|
| **Scope** | Full timeline (all kernels, memcpy, CPU) | Per-kernel deep dive |
| **Speed** | Fast (no replay) | Slow (kernel replay for counters) |
| **Use for** | Kernel launch counts, idle gaps, overall flow | Occupancy, throughput, bottleneck analysis |
| **Output** | `.nsys-rep` | `.ncu-rep` |

## Results (2026-02-24)

**Model:** 1.65B params (8 layers: 6 DeltaNet + 2 GSA), hidden=4096, Kronecker embeddings (8192→4096)

### Per-optimization correctness isolation

| Optimization | max_diff | mean_diff | loss_diff | Status |
|---|---|---|---|---|
| RoPE only | 2.6875 | 0.2793 | 0.0000 | PASS |
| RMSNorm+SwishGate only | 0.0000 | 0.0000 | 0.0000 | PASS |
| BatchedMHCCoeffs only | 2.9375 | 0.2441 | 896.0000 | **FAIL** |

> **Note:** BatchedMHCCoeffs produces significant loss diff (896) in isolation — numerical issue with batched matmul implementation needs investigation.

### Full model benchmark (forward + backward)

```
Configuration                ms/step   tokens/sec  vs baseline
------------------------------------------------------------------------
Baseline                    4890.6ms       6,700            —
+ Kernel fusions            4642.7ms       7,058        +5.3%
------------------------------------------------------------------------
GPU memory: 5.5GB allocated, 39.5GB reserved, 38.6GB peak
```

> **Note:** Combined kernel fusions failed correctness check (max_diff=3.03, loss_diff=1408, 7.64%). The +5.3% throughput gain is real but results are numerically incorrect. BatchedMHCCoeffs is the likely culprit based on isolation tests.

## Output Example

```
========================================================================
  KERNEL MICRO-BENCHMARKS
========================================================================
Operation               Baseline (ms)  Optimized (ms)  Speedup
------------------------------------------------------------------------
RoPE                       75.945          9.688         7.84x
SiLU*mul                    7.463          8.916         0.84x
RMSNorm+SwishGate          79.645         43.789         1.82x
RMSNorm autotune            8.249          8.275         1.00x
MHCCoeffs batch            25.372          1.064        23.84x

========================================================================
  FULL MODEL BENCHMARK (forward + backward)
========================================================================
Configuration           ms/step    tokens/sec   vs baseline
------------------------------------------------------------------------
Baseline                1523.4ms     21,532         —
+ Kernel fusions        1342.1ms     24,456       +13.6%
+ torch.compile         1198.7ms     27,372       +27.1%
```
