# Kernel-Level Profiling Guide

This document describes the enhanced profiling system that now provides **granular kernel-level timing** with real-time reporting, designed for optimization analysis without impacting training throughput.

## Overview

### Four Levels of Profiling

1. **Step-level** (coarse) — forward, backward, optimizer, dataloader
2. **Layer-level** — per LightningDecoderLayer, per MTP block
3. **Kernel-level** — individual kernels like GSA, DeltaNet, indexer, RMSNorm
4. **Sub-kernel level** (new) — granular operations within kernels (matmul, softmax, topk, reduce, etc.)

### New Features

✅ **Real-time JSONL Writing** — Step results appended immediately after completion  
✅ **Async Background Writer** — Optional background thread (zero throughput impact)  
✅ **Hierarchical Regions** — Track nested operations (e.g., `gsa.sparse_attn.matmul`)  
✅ **Per-call Timing** — Automatic averaging of repeated operations  
✅ **Zero Overhead** — Profiling is a strict no-op when disabled  

## Quick Start

### 1. Enable Profiling in train.py

```python
from src.profiler import StepProfiler

# Create profiler with async writing (recommended)
profiler = StepProfiler(
    rank=local_rank,
    profile_steps={10, 11, 12},  # Only profile steps 10–12
    output_dir="results/run",
    enable_async_write=True,      # NEW: async writing (zero impact)
)
profiler.activate()
profiler.register_model(model_engine.module)
```

### 2. Automatic Step Recording

The training loop automatically handles step boundaries:

```python
profiler.start_step(global_step, tokens=tokens_this_step)
# --- training forward/backward ---
profiler.end_step(tokens=tokens_this_step)
```

### 3. Real-time Reports

During training, JSONL results are written incrementally:

```
results/run/profile.jsonl  ← One line per profiled step (appended in real-time)
results/run/profile_report.txt  ← Summary after training finishes
```

## Kernel Instrumentation

### Adding Profiling to Kernels

#### For high-level operations (layer/kernel boundary):

```python
from src.profiler import time_region

with time_region("gsa.sparse_attn"):
    output = triton_sparse_attention(...)
```

#### For sub-kernel operations (minute-level breakdown):

```python
from src.profiler import kernel_region

with kernel_region("sparse_attn.matmul"):
    scores = triton_matmul(q, k)

with kernel_region("sparse_attn.softmax"):
    attn_weights = softmax(scores)

with kernel_region("sparse_attn.output"):
    output = attn_weights @ v
```

### Already Instrumented Kernels

The following kernels have been updated with granular profiling:

1. **triton_sparse_attn_v2.py**
   - `sparse_attn_v2.fwd_total`
   - `sparse_attn_v2.fwd_kernel`
   - `sparse_attn_v2.bwd_total`
   - `sparse_attn_v2.bwd_dq`
   - `sparse_attn_v2.bwd_dkdv`

2. **triton_indexer.py**
   - `indexer_total`
   - `indexer_kernel`
   - `indexer_convert`

3. **triton_rmsnorm.py**
   - `rmsnorm_fwd_kernel`
   - `rmsnorm_bwd_kernel`

### Adding to Other Kernels

To instrument a kernel file, follow this pattern:

**Step 1: Import profiling helpers**

```python
# At the top of the file
try:
    from ..profiler import kernel_region
except ImportError:
    # Fallback: no-op context manager when profiler unavailable
    from contextlib import contextmanager
    @contextmanager
    def kernel_region(name: str):
        yield
```

**Step 2: Wrap operations**

```python
def my_kernel_function(...):
    with kernel_region("my_kernel.total"):
        with kernel_region("my_kernel.alloc"):
            # Allocate output tensors
            out = torch.empty(...)
        
        with kernel_region("my_kernel.compute"):
            # Main computation
            my_triton_kernel[grid](...)
        
        with kernel_region("my_kernel.convert"):
            # Type conversion
            out = out.to(dtype)
    
    return out
```

## Profiler Report Format

### Text Report (profile_report.txt)

```
══════════════════════════════════════════════════════════════════════════════
  STEP PROFILER REPORT — Granular Kernel Analysis
  (3 step(s) averaged, 4096 tokens/step)
══════════════════════════════════════════════════════════════════════════════

── Step Phases ──────────────────────────────────────────────────────
  Region                           ms        %step
  ─────────────────────────────────────────────────
  dataloader                     12.5        2.1%
  forward                       425.3       71.4%
  backward                      150.2       25.2%
  optim_step                      2.5        0.4%
  step_total                    596.4      100.0%

── Granular Kernel Operations (avg per call) ────────────────────────
  Operation                                            per-call ms    calls
  ────────────────────────────────────────────────────────────────────────
  gsa.sparse_attn.fwd_kernel                              18.532      24
  gsa.indexer.kernel                                      12.155      24
  rmsnorm_fwd_kernel                                       3.241      48
  gsa.sparse_attn.bwd_dkdv                                15.823      24
  deltanet.fwd                                            22.104      24

── All Regions (sorted by avg ms) ───────────────────────────────────
  Region                                          avg ms     calls  per-call
  ────────────────────────────────────────────────────────────────────────
  sparse_attn_v2.fwd_kernel                       18.532        24    0.7722
  indexer_kernel                                  12.155        24    0.5065
  rmsnorm_fwd_kernel                               3.241        48    0.0676
  ...

  Estimated throughput: 19,453 tok/sec
══════════════════════════════════════════════════════════════════════════════
```

### JSONL Report (profile.jsonl)

Each line is one profiled step:

```json
{"step": 10, "tokens": 4096, "timestamp": 1708697234.123, "forward": 425.3, "backward": 150.2, "sparse_attn_v2.fwd_kernel": 18.532, "sparse_attn_v2.fwd_kernel__count": 8, "sparse_attn_v2.fwd_kernel__avg": 2.3165, ...}
{"step": 11, "tokens": 4096, "timestamp": 1708697245.567, ...}
{"step": 12, "tokens": 4096, "timestamp": 1708697256.891, ...}
```

**Fields**:
- `step`, `tokens` — step metadata
- `timestamp` — wall-clock when step started
- `<region_name>` — total time in ms
- `<region_name>__count` — number of times region was recorded (if > 1)
- `<region_name>__avg` — average per call (if recorded multiple times)

## Configuration Options

Create the profiler with various options:

```python
profiler = StepProfiler(
    rank=local_rank,                    # Only rank-0 writes reports
    profile_steps={10, 11, 12, ...},    # Set of step numbers to profile
    output_dir="results/run",           # Where to write JSONL/report
    enable_async_write=True,            # (NEW) Async JSONL writing
)
```

### Async Writing

**Enabled (default):**
- Background thread writes JSONL in parallel
- Zero impact on training throughput
- Recommended for all configurations

**Disabled:**
```python
profiler = StepProfiler(..., enable_async_write=False)
```
- Synchronous JSONL write after each step
- Minimal overhead (< 1ms) but blocks training
- Useful for debugging or if background thread has issues

## Usage Examples

### Example 1: Profile Last 3 Steps

```python
# in main.py or train.py
if ENABLE_PROFILING:
    N_TRAIN_STEPS = 1000
    profile_steps = {N_TRAIN_STEPS - 3, N_TRAIN_STEPS - 2, N_TRAIN_STEPS - 1}
    profiler = StepProfiler(
        rank=local_rank,
        profile_steps=profile_steps,
        enable_async_write=True,
    )
    profiler.activate()
```

### Example 2: Access Results After Training

```python
# After training completes
if profiler:
    profiler.deactivate()
    profiler.write_report("results/run/profile_report.txt")
    
    # JSONL already written incrementally; this is optional flush
    profiler.write_jsonl()
```

### Example 3: Parse Results for Analysis

```python
import json
import pandas as pd

# Read JSONL
rows = []
with open("results/run/profile.jsonl") as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)

# Find slowest kernels
kernel_cols = [c for c in df.columns if '__avg' in c]
kernel_times = df[kernel_cols].mean().sort_values(ascending=False)
print(kernel_times)

# Compute throughput per step
df['tok_per_sec'] = df['tokens'] / (df['step_total'] / 1000.0)
print(f"Average throughput: {df['tok_per_sec'].mean():,.0f} tok/sec")
```

## Performance Impact

- **No profiling** → 0 ms overhead
- **Profiling disabled** → < 0.1 ms (global flag check)
- **Profiling enabled, sync JSONL** → ~1 ms per step (disk I/O)
- **Profiling enabled, async JSONL** → ~0.1 ms per step (queued write)

The async writer is the recommended default — it gives you real-time JSONL output without any throughput penalty.

## Optimization Workflow

1. **Profile a few representative steps** (e.g., steps 100–102)
2. **Identify bottleneck kernels** from the report (sort by total time)
3. **Drill into sub-kernel timings** (e.g., matmul vs. softmax within sparse attn)
4. **Optimize the kernel** using insights
5. **Re-profile to verify** improvement
6. **Repeat** for next bottleneck

Example workflow for GSA optimization:

```
profile_steps = {100, 101, 102}  # Quick turnaround
↓
Report shows: sparse_attn_v2.bwd_dkdv is 40% of backward time
↓
Re-instrument dkdv kernel with sub-kernel timing
↓
Find: dV accumulation (atomics) is 50% of dkdv time
↓
Optimize: Switch to atomic-free key-major dV kernel
↓
Re-profile: Verify 2× speedup achieved
```

## Zero-Overhead Guarantee

When profiling is disabled or a step is not in `profile_steps`:

```python
# These are NO-OPs (one global read + branch):
with time_region("name"):      # → if profiler is None: yield
    pass

with kernel_region("name"):    # → if profiler is None: yield
    pass
```

The Python function call overhead is ~1 µs per region, where a CUDA kernel launch is ~10–100 µs. Profiling is hidden in kernel launch overhead.

## Troubleshooting

### Q: JSONL file is empty or has no step records

A: Verify `profile_steps` includes actual step numbers. Profiler only records steps in the set.

```python
# Check what's being profiled
print(profiler.profile_steps)  # Should be non-empty
print(profiler._history)       # After deactivate(), should have records
```

### Q: Async writer is falling behind

A: JSONL writes should be very fast (< 5 ms per 100 KB file). If you see queue buildup:

```python
# Disable async for debugging
profiler = StepProfiler(..., enable_async_write=False)
```

Then profile again to see sync write overhead.

### Q: Can't import `kernel_region`

A: The `kernel_region` import will fail gracefully if profiler module is unavailable. Check:

```python
try:
    from ..profiler import kernel_region
except ImportError:
    print("Warning: profiler not available")
    kernel_region = lambda name: contextmanager(lambda: (yield))()
```

## Future Extensions

Possible additions to the profiling system:

- [ ] GPU memory tracking per kernel
- [ ] Roofline analysis (FLOPs vs. bandwidth)
- [ ] Kernel comparison across steps
- [ ] Interactive real-time dashboard
- [ ] Automatic bottleneck detection
- [ ] Integration with PyTorch Profiler
- [ ] Multi-GPU aggregation

