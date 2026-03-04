# Nsight Systems Profiling - Analysis & Findings

## 📊 Overview

This directory contains Nsight Systems profiling results and analysis for the 1B LLM training pipeline.

**Profiling Configuration:**
- **Profile Range:** Steps 10-20 (out of 100 total)
- **Model:** LLM-1B-Base (662M parameters)
- **GPU:** Tesla T4 (14.56 GB VRAM)
- **Batch Size:** 1
- **Sequence Length:** 512
- **Gradient Accumulation:** 1

**Output Files:**
- `seed_1b_timeline.nsys-rep` (2.2 MB) - Nsight Systems report
- `seed_1b_timeline.sqlite` (6.5 MB) - Database with detailed metrics
- `baseline.log` (21 KB) - Complete profiling run output

---

## 🎯 Quick Start

### Running Profiling

```bash
# Make script executable
chmod +x profile_1b_timeline.sh

# Run with sudo for GPU metrics (optional)
sudo -E ./profile_1b_timeline.sh
```

### Viewing Results

```bash
# Command-line statistics
nsys stats seed_1b_timeline.nsys-rep

# Transfer to local machine for GUI analysis
scp user@remote:/path/to/seed_1b_timeline.nsys-rep .

# Open in Nsight Systems GUI
# Launch Nsight Systems → File → Open → seed_1b_timeline.nsys-rep
```

---

## ⚠️ Critical Findings from baseline.log

### 🔴 Issue 1: Severe Training Throughput Degradation During Profiling

**Problem:** Training throughput drops dramatically during profiling capture.

**Evidence from baseline.log:**
```
Step  10/100 | Tok/s: 393   # Profiling starts
Step  20/100 | Tok/s: 59    # 85% throughput drop! ⚠️
Step  30/100 | Tok/s: 583   # Recovers after profiling
Step  40/100 | Tok/s: 581   # Back to normal
```

**Impact:**
- Normal throughput: **~580 tokens/sec**
- During profiling: **~59-393 tokens/sec** (10-68% of normal)
- **85% performance degradation** at step 20

**Root Cause:** Profiling overhead from Nsight Systems capturing detailed CUDA events, API calls, and kernel metrics.

**Workaround:** This is **expected behavior**. For production training, disable profiling after initial analysis.

---

### 🔴 Issue 2: PyTorch API Deprecation Warnings

**Problem:** Using deprecated PyTorch AMP APIs that will be removed in future versions.

**Evidence from baseline.log (lines 25-26, 45-46):**
```python
FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. 
               Please use `torch.amp.GradScaler('cuda', args...)` instead.

FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. 
               Please use `torch.amp.autocast('cuda', args...)` instead.
```

**Location in Code:**
- `training/train.py:271` - GradScaler initialization
- `training/train.py:359` - autocast context manager

**Impact:**
- ⚠️ Code will break in future PyTorch versions (likely PyTorch 2.5+)
- ✅ No current functional impact

**Required Fix:**
```python
# OLD (deprecated):
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler(enabled=self.use_amp)
with autocast(enabled=self.use_amp, dtype=self.amp_dtype):

# NEW (recommended):
from torch.amp import autocast, GradScaler
scaler = GradScaler('cuda', enabled=self.use_amp)
with autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
```

---

## 🟡 Performance Concerns

### Issue 3: Missing NVTX Instrumentation

**Problem:** No custom profiling markers detected in the timeline.

**Evidence from baseline.log (line 59):**
```
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: No data available.
```

**Impact:**
- Cannot identify specific training phases (forward/backward/optimizer) in timeline
- Difficult to pinpoint bottlenecks
- No custom profiling markers for debugging

**Recommendation:** Add NVTX annotations to the training loop:

```python
import torch.cuda.nvtx as nvtx

# In training loop
nvtx.range_push("forward_pass")
outputs = model(input_ids, labels=labels)
nvtx.range_pop()

nvtx.range_push("backward_pass")
loss.backward()
nvtx.range_pop()

nvtx.range_push("optimizer_step")
optimizer.step()
nvtx.range_pop()
```

---

### Issue 4: High CUDA Stream Synchronization Overhead

**Problem:** Excessive time spent in CUDA stream synchronization.

**Evidence from baseline.log (line 89):**
```
cudaStreamSynchronize: 45.4% of total CUDA API time
- 50 calls total
- Avg: 85.1 ms per call
- Max: 213.5 ms per call
- Total: 4.26 seconds
```

**Impact:**
- GPU may be idle waiting for synchronization
- Poor CPU-GPU overlap
- Potential performance bottleneck

**Analysis:**
- Average 85ms per sync is **very high** (should be <10ms ideally)
- Indicates potential GPU starvation or CPU bottlenecks
- May suggest data loading or preprocessing issues

**Investigation Steps:**
1. Check dataloader num_workers (increase for better CPU-GPU overlap)
2. Profile CPU activity to identify blocking operations
3. Consider async data loading and pinned memory

---

### Issue 5: Kernel Launch Overhead

**Problem:** High ratio of kernel launches to actual compute time.

**Evidence from baseline.log (line 88):**
```
cudaLaunchKernel: 54.3% of CUDA API time
- 41,670 kernel launches
- Avg: 122 μs per launch
```

**Impact:**
- Many small kernels being launched
- Launch overhead becomes significant
- Opportunity for kernel fusion

**Analysis:**
- Launching 41,670 kernels for just 10 training steps
- ~4,167 kernel launches per step
- Indicates fragmented computation

**Potential Optimizations:**
- Evaluate `torch.compile()` for automatic kernel fusion
- Use custom fused CUDA kernels for repeated patterns
- Consider operator fusion in model architecture

---

## 📊 Performance Baseline Metrics

### GPU Kernel Time Distribution

From `baseline.log` lines 99-156:

| Kernel Type | GPU Time | % of Total | Count | Optimization Potential |
|-------------|----------|------------|-------|------------------------|
| **GEMM operations (cuBLAS)** | 6.92s | 64.0% | 6,510 | ✅ Expected for transformers |
| **Multi-tensor operations** | 1.57s | 14.5% | 2,640 | ⚠️ Could be fused/batched |
| **Element-wise kernels** | 1.31s | 12.1% | 2,920 | ⚠️ High fragmentation |
| **Softmax forward/backward** | 194ms | 1.8% | 20 | ✅ Acceptable |
| **Memory copies** | 284ms | 2.6% | 5,310 | ✅ Acceptable |
| **Other** | 538ms | 5.0% | - | - |

**Total GPU Compute Time:** 10.82 seconds (for 10 training steps)

**Key Observations:**
1. ✅ **64% GEMM time is normal** for transformer models (matrix multiplication dominates)
2. ⚠️ **14.5% multi-tensor ops** suggests parameter updates could be batched better
3. ⚠️ **12.1% elementwise kernels** with 2,920 launches indicates kernel fragmentation
4. ✅ **Softmax is only 1.8%** - attention mechanism is efficient

**Top 3 Most Expensive Kernels:**
1. `magma_sgemmEx_kernel` (no transpose): 2.60s (24.0%)
2. `magma_sgemmEx_kernel` (transpose A): 2.29s (21.2%)
3. `magma_sgemmEx_kernel` (transpose B): 2.03s (18.8%)

**Total:** These 3 GEMM variants account for **64% of all GPU time** - this is expected and optimal.

---

### Memory Transfer Efficiency

From `baseline.log` lines 160-172:

| Transfer Type | Time | % of Total | Size | Count |
|---------------|------|------------|------|-------|
| Device-to-Device | 22.7 ms | 99.8% | 2.57 GB | 500 |
| Host-to-Device | 11.9 μs | 0.1% | 0.082 MB | 20 |
| Device-to-Host | 35.7 μs | 0.2% | <0.001 MB | 30 |

**Analysis:**
- ✅ **Excellent:** 99.8% of transfers are GPU-internal (D2D)
- ✅ **Minimal CPU↔GPU transfers** (<0.1 MB total)
- ✅ **No transfer bottlenecks** detected
- ✅ Data properly residing on GPU

**Average Transfer Sizes:**
- D2D: 5.14 MB per transfer (good batch size)
- H2D: 4 KB per transfer (minimal overhead)
- D2H: <1 KB per transfer (just metrics)

---

### CUDA API Time Distribution

From `baseline.log` lines 86-93:

| CUDA API | Time | % of Total | Calls | Avg Time |
|----------|------|------------|-------|----------|
| `cudaLaunchKernel` | 5.09s | 54.3% | 41,670 | 122 μs |
| `cudaStreamSynchronize` | 4.26s | 45.4% | 50 | 85.1 ms |
| `cudaMemcpyAsync` | 28.8ms | 0.3% | 550 | 52.3 μs |
| Other | 52ms | <0.1% | 12 | - |

**Total CUDA API Time:** 9.37 seconds

---

### OS Runtime (CPU) Breakdown

From `baseline.log` lines 62-82:

Top CPU activities during training:

1. **`poll`** - 40.2% (30.0s) - Waiting for I/O events
2. **`sem_wait`** - 35.2% (26.2s) - Thread synchronization
3. **`pthread_cond_wait`** - 13.9% (10.4s) - Condition variable waits
4. **`pthread_cond_timedwait`** - 10.7% (8.0s) - Timed waits

**Analysis:**
- CPU spends **89.3%** of time waiting (poll + sem_wait + cond_wait)
- This is **normal** for GPU-bound training
- Indicates GPU is the bottleneck (as expected)

---

## 🎯 Performance Summary

### Training Metrics from baseline.log

**Configuration:**
```
Model:               LLM-1B-Base (662M parameters)
GPU:                 Tesla T4 (14.56 GB VRAM)
Batch size:          1
Sequence length:     512
Gradient accumulation: 1
Precision:           BFloat16 mixed precision
```

**Performance:**
```
Normal throughput:   ~580 tokens/sec
Total steps:         100
Total time:          182.1 seconds
Time per step:       ~1.82 seconds
Final loss:          11.0757
Tokens processed:    51,200 (100 steps × 512 seq_len)
```

**Profiling Overhead:**
```
Profiled steps:      10-20 (10 steps captured)
Profiling time:      ~10 seconds
Output size:         2.2 MB (.nsys-rep)
Throughput impact:   -85% during capture
```

---

## 🔧 Recommended Actions

### Priority 1: Critical (Must Fix)

- [ ] **Update PyTorch AMP API calls** to non-deprecated versions
  - Files: `training/train.py` lines 271, 359
  - Impact: Will break in future PyTorch versions
  
- [ ] **Profile with larger batch sizes** on more capable GPUs (A100/H100)
  - Current: Batch size 1 on Tesla T4 (memory constrained)
  - Recommended: Batch size 8-16 for realistic performance analysis

### Priority 2: Performance Optimization

- [ ] **Add NVTX markers** for better profiling insights
  - Impact: Easier identification of bottlenecks in timeline
  - Effort: Low (add 6-8 markers in training loop)

- [ ] **Investigate high cudaStreamSynchronize times** (85ms avg)
  - Check dataloader configuration (num_workers, pin_memory)
  - Profile CPU activity during training
  - Consider async data loading

- [ ] **Consider kernel fusion** for elementwise operations
  - 2,920 small kernel launches detected
  - Try `torch.compile()` for automatic fusion
  - Measure impact on throughput

### Priority 3: Long-term Optimization

- [ ] **Evaluate torch.compile()** for kernel fusion
  - Expected benefit: 10-30% speedup from fusion
  - Test on PyTorch 2.0+

- [ ] **Consider gradient checkpointing** for larger batches
  - Trade compute for memory
  - Enable larger batch sizes on Tesla T4

- [ ] **Profile with different batch sizes** to find sweet spot
  - Test: 1, 2, 4, 8 (if memory allows)
  -找 optimal throughput/memory tradeoff

---

## 📁 Files in This Directory

| File | Description | Size |
|------|-------------|------|
| `profile_1b_timeline.sh` | Profiling script | - |
| `seed_1b_timeline.nsys-rep` | Nsight Systems report | 2.2 MB |
| `seed_1b_timeline.sqlite` | Profiling database | 6.5 MB |
| `baseline.log` | Complete profiling output | 21 KB |
| `README.md` | This file | - |
| `PROFILING_SUCCESS.md` | Initial profiling guide | - |

---

## 🚀 Next Steps

1. **Review Timeline in Nsight Systems GUI**
   - Transfer `.nsys-rep` file to local machine
   - Open in Nsight Systems
   - Look for GPU idle gaps ("bubbles")
   - Identify kernel launch patterns

2. **Fix Deprecation Warnings**
   - Update AMP API calls in `training/train.py`
   - Test with latest PyTorch version

3. **Add NVTX Instrumentation**
   - Mark forward/backward/optimizer phases
   - Re-profile with markers
   - Analyze phase timings

4. **Optimize Hotspots**
   - Focus on cudaStreamSynchronize overhead
   - Consider kernel fusion opportunities
   - Test with `torch.compile()`

---

## 📊 System Information

**From baseline.log warnings (lines 19-23):**
```
WARNING: CPU IP/backtrace sampling not supported
WARNING: CPU context switch tracing not supported
```

**Impact:** Limited CPU profiling capabilities on this system. GPU profiling data is unaffected and complete.

**Nsight Systems Version:** 2025.5.2 (or later)

---

## 📚 Additional Resources

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [CUDA Profiler API](https://docs.nvidia.com/cuda/profiler-users-guide/index.html)
- [NVTX Documentation](https://nvidia.github.io/NVTX/doxygen-cpp/index.html)

---

**Last Updated:** 2026-02-02  
**Profiling Run:** baseline.log (100 steps, profiled steps 10-20)  
**Model:** LLM-1B-Base (662M parameters)
