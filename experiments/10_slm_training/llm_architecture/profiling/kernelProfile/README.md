# Nsight Compute Kernel Profiling

## 📊 Overview

This directory contains Nsight Compute kernel-level profiling scripts for deep-dive performance analysis of individual CUDA kernels in the 1B LLM training pipeline.

**Profiling Level:** Per-kernel detailed metrics (compute vs memory bottlenecks, occupancy, warp stalls)

---

## 🚀 Quick Start

### Prerequisites

**Install Nsight Compute:**
```bash
# From project root
./install_profiling_tools.sh

# Or manually:
sudo apt-get install -y nsight-compute-2025.4.1

# Add to PATH if needed:
export PATH=$PATH:/opt/nvidia/nsight-compute/2025.4.1
```

**Verify Installation:**
```bash
ncu --version
# Should show: NVIDIA (R) Nsight Compute Command Line Profiler
```

---

## 📁 Profiling Scripts

### 1. Full Kernel Profiling (Comprehensive)

**Script:** `profile_all_kernels.sh`

**What it does:**
- Profiles **ALL** kernels launched during steps 10-11 (2 steps)
- Captures comprehensive metrics for each kernel
- Generates detailed `.ncu-rep` file for GUI analysis

**When to use:**
- Initial profiling to identify hotspot kernels
- Comprehensive baseline analysis
- When you don't know which kernels are slow

**Runtime:** ⚠️ **30-60 minutes** (very slow!)

**Configuration:**
```bash
Profile Steps: 10-11 (2 steps only)
Batch Size: 1
Sequence Length: 512
Expected Kernels: ~8,334 (4,167 per step × 2)
```

**Run:**
```bash
chmod +x profile_all_kernels.sh
source profile_all_kernels.sh
```

---

### 2. Focused Kernel Profiling (Targeted)

**Script:** `profile_focused_kernels.sh`

**What it does:**
- Profiles **only filtered** kernels matching regex pattern
- Default filter: `attention|matmul|layernorm|gelu|softmax|rope`
- Much faster than full profiling

**When to use:**
- After identifying hotspots from full profiling
- Deep-dive on specific operations (attention, matmul, etc.)
- Iterative optimization workflow

**Runtime:** ~10-20 minutes (filtered)

**Configuration:**
```bash
Profile Steps: 10-12 (3 steps)
Kernel Filter: attention|matmul|layernorm|gelu|softmax|rope
Batch Size: 1
Sequence Length: 512
```

**Run:**
```bash
chmod +x profile_focused_kernels.sh
source profile_focused_kernels.sh
```

**Customize Filter:**
Edit the script and change:
```bash
# Focus on GEMM operations only
KERNEL_PATTERN="gemm|sgemm|matmul|cublas"

# Focus on attention mechanism
KERNEL_PATTERN="attention|softmax|rope"

# Focus on activations
KERNEL_PATTERN="gelu|silu|relu|tanh"
```

---

## 📊 Expected Kernel Distribution

Based on Nsight Systems baseline profiling (`nsightSystemProfile/baseline.log`):

### Top Kernels by GPU Time

| Kernel Family | GPU Time | % of Total | Launches | Optimization Priority |
|---------------|----------|------------|----------|----------------------|
| **GEMM (cuBLAS)** | 6.92s | 64.0% | 6,510 | ⚠️ **HIGH** - Biggest impact |
| **Multi-tensor ops** | 1.57s | 14.5% | 2,640 | 🟡 Medium - Fusion opportunity |
| **Element-wise** | 1.31s | 12.1% | 2,920 | 🟡 Medium - Fragmented |
| **Softmax** | 194ms | 1.8% | 20 | ✅ Low - Already efficient |
| **Memory copies** | 284ms | 2.6% | 5,310 | ✅ Low - Acceptable |

**Total:** 10.82s for 10 training steps (per baseline profiling)

---

## 🎯 What to Profile

### Priority 1: GEMM Operations (64% of GPU time)

**Expected kernels:**
- `magma_sgemmEx_kernel` variations
- `cublasSgemm` calls
- Matrix multiplication in attention and FFN

**What to look for:**
- Compute vs memory bound (Speed of Light metrics)
- Tensor Core utilization
- Achieved occupancy vs theoretical

**Optimization potential:** Medium (cuBLAS is already highly optimized)

---

### Priority 2: Multi-Tensor Operations (14.5% of GPU time)

**Expected kernels:**
- `multi_tensor_apply_kernel` 
- Adam optimizer updates
- Gradient operations

**What to look for:**
- Can these be fused?
- Memory bandwidth utilization
- Launch overhead

**Optimization potential:** High (many small kernels can be batched)

---

### Priority 3: Element-wise Operations (12.1% of GPU time)

**Expected kernels:**
- `vectorized_elementwise_kernel`
- `unrolled_elementwise_kernel`
- Activation functions (SiLU, GELU)

**What to look for:**
- Memory bandwidth bound
- Kernel fusion opportunities
- Launch overhead (2,920 launches!)

**Optimization potential:** High (too many small kernels)

---

## 📈 Metrics to Analyze

### Speed of Light (SOL)

Indicates whether kernel is compute or memory bound:

```
Compute (SM) Throughput:  75%  ← Compute utilization
Memory Throughput:        45%  ← Memory bandwidth utilization
```

**Interpretation:**
- Memory > Compute: **Memory bound** → Optimize memory access patterns
- Compute > Memory: **Compute bound** → Optimize arithmetic instructions
- Both high (>80%): **Well balanced** ✅
- Both low (<50%): **Launch overhead** or **dependency stalls**

---

### Occupancy

Percentage of active warps vs theoretical maximum:

```
Achieved Occupancy:     65.2%
Theoretical Occupancy:  100%
Limiting Factor:        Shared Memory
```

**What to optimize:**
- Shared memory usage
- Register pressure
- Thread block size

**Good occupancy:** >60% for compute-bound, >40% for memory-bound

---

### Warp Stall Reasons

Why warps are waiting instead of executing:

```
Memory Throttle:        45%  ← Waiting for memory
Execution Dependency:   25%  ← Waiting for previous instructions  
Synchronization:        15%  ← __syncthreads() barriers
Not Selected:           15%  ← Scheduler didn't pick this warp
```

**Optimization targets:**
- High memory throttle → Improve memory coalescing
- High dependency → Increase instruction-level parallelism
- High sync → Reduce synchronization points

---

## 📁 Output Files

After running profiling, expect:

```
kernelProfile/
├── seed_1b_kernels.ncu-rep          # Full profiling report
├── seed_1b_focused_kernels.ncu-rep  # Focused profiling report
├── kernel_metrics.csv                # Exported CSV metrics
└── README.md                         # This file
```

---

## 🔍 Analysis Workflow

### Step 1: Run Full Profiling (Initial Survey)

```bash
source profile_all_kernels.sh
# Wait ~30-60 minutes
```

**What you get:**
- Complete kernel inventory
- Identify top 10 hotspot kernels
- Understanding of kernel distribution

---

### Step 2: Export Metrics

```bash
ncu --import seed_1b_kernels.ncu-rep --csv > kernel_metrics.csv
```

**Analyze CSV:**
```bash
# Sort by duration to find top kernels
sort -t',' -k5 -rn kernel_metrics.csv | head -20

# Filter by kernel name
grep "gemm" kernel_metrics.csv
```

---

### Step 3: Transfer for GUI Analysis

```bash
# From local machine:
scp user@remote:/path/to/seed_1b_kernels.ncu-rep .

# Open in Nsight Compute GUI
# File → Open → seed_1b_kernels.ncu-rep
```

**GUI Features:**
- Interactive timeline
- Detailed metrics per kernel
- Source code correlation
- Optimization suggestions

---

### Step 4: Deep-Dive on Hotspots

Based on findings, update focused profiling:

```bash
# Edit profile_focused_kernels.sh
KERNEL_PATTERN="<your-hotspot-kernels>"

# Re-run focused profiling
source profile_focused_kernels.sh
# Wait ~10-20 minutes
```

---

## 🎯 Expected Findings

Based on the Nsight Systems baseline, you should expect to find:

### 1. GEMM Kernels Dominate (64%)

**Typical findings:**
- Well-optimized cuBLAS kernels
- High compute throughput (>70%)
- Using Tensor Cores (if available)
- Already efficient, limited optimization potential

**Potential optimizations:**
- Tune matrix dimensions for Tensor Core alignment
- Check if BF16 is being used effectively
- Verify no unnecessary data type conversions

---

### 2. Fragmented Element-wise Kernels (12.1%)

**Typical findings:**
- 2,920 small kernel launches
- Memory bandwidth bound
- Low occupancy due to small grid sizes
- High launch overhead

**Optimization opportunities:**
- Kernel fusion (combine multiple element-wise ops)
- Use `torch.compile()` for automatic fusion
- Custom fused CUDA kernels

---

### 3. Multi-Tensor Operations (14.5%)

**Typical findings:**
- Adam optimizer doing many small updates
- Each parameter gets separate kernel launch
- Could be batched/fused

**Optimization opportunities:**
- Use fused optimizer kernels
- Batch parameter updates
- Consider apex fused optimizers

---

## 🚨 Known Issues

### Issue 1: Profiling is Very Slow

**Problem:** Full kernel profiling takes 30-60 minutes for just 2 training steps.

**Cause:** Nsight Compute captures detailed metrics per kernel invocation.

**Workarounds:**
- Use focused profiling with kernel filters
- Profile even fewer steps (1 step instead of 2)
- Profile only specific kernel launches with `--launch-skip` / `--launch-count`

---

### Issue 2: OOM Errors on Tesla T4

**Problem:** CUDA out of memory errors during profiling.

**Solution:** Scripts are already configured for Tesla T4:
- Batch size: 1
- Sequence length: 512
- Gradient accumulation: 1

If still getting OOM, further reduce sequence length:
```bash
SEQ_LENGTH=256  # In profiling scripts
```

---

### Issue 3: No GPU Metrics on Some GPUs

**Problem:** Some Tesla GPUs (T4, K80) don't support all profiling metrics.

**Impact:** Some sections may show "Not Supported"

**Workaround:** Focus on metrics that are available (SOL, occupancy, duration)

---

## 📚 Metrics Reference

### Key Metrics to Review

| Metric | What It Measures | Good Value | Bad Value |
|--------|------------------|------------|-----------|
| **Duration** | Time kernel ran | Baseline | 2-10x baseline |
| **SM Throughput** | Compute utilization | >70% | <30% |
| **Memory Throughput** | Memory bandwidth use | Depends | >95% (bandwidth limit) |
| **Achieved Occupancy** | Active warps % | >60% | <30% |
| **L2 Cache Hit Rate** | Cache effectiveness | >80% | <20% |
| **Global Load Efficiency** | Coalesced loads | >80% | <50% |
| **Global Store Efficiency** | Coalesced stores | >80% | <50% |

---

## 🔧 Customization

### Profiling Different Steps

```bash
# Edit scripts and change:
PROFILE_STEPS="5-7"    # Profile earlier steps
PROFILE_STEPS="20-22"  # Profile later steps (more stable gradients)
```

### Adding More Analysis Sections

```bash
# In profiling script, add sections:
--section InstructionStats \
--section LaunchStats \
--section Scheduler \
```

**Available sections:**
- `SpeedOfLight` - Compute/memory throughput
- `MemoryWorkloadAnalysis` - Memory access patterns
- `ComputeWorkloadAnalysis` - Instruction mix
- `Occupancy` - Warp occupancy factors
- `InstructionStats` - Instruction counts
- `LaunchStats` - Kernel launch info
- `Scheduler` - Warp scheduler efficiency

---

## 🎓 Learning Resources

- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
- [CUDA Kernel Profiling Best Practices](https://docs.nvidia.com/cuda/profiler-users-guide/index.html)
- [Understanding GPU Performance](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
- [Optimizing CUDA Applications](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

---

## 📝 Next Steps After Profiling

1. **Identify Top 5 Kernels** by duration
2. **Classify Each:**
   - Compute bound vs memory bound
   - Well-optimized vs optimization potential
3. **Prioritize:**
   - High impact (>5% of total time)
   - High optimization potential
4. **Optimize:**
   - Kernel fusion for element-wise ops
   - Memory access patterns for memory-bound
   - Try `torch.compile()` first (easy wins)
5. **Re-profile:**
   - Measure improvement
   - Iterate

---

**Profiling Configuration:**
- GPU: Tesla T4 (14.56 GB)
- Batch Size: 1
- Sequence Length: 512
- Model: LLM-1B-Base (662M parameters)

**Last Updated:** 2026-02-02
