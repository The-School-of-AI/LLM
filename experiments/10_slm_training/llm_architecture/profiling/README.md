# LLM Training Profiling Guide

This directory contains profiling tools and guides for analyzing and optimizing the 1B LLM training pipeline performance.

## 📋 Prerequisites

Before using the profiling scripts, install NVIDIA profiling tools:

### Quick Installation (Recommended)

From the project root:
```bash
cd llm_architecture
chmod +x install_profiling_tools.sh
./install_profiling_tools.sh
```

This installs:
- Nsight Systems 2025.5.2 (timeline profiling)
- Nsight Compute 2025.4.1 (kernel profiling)

### Manual Installation

```bash
sudo apt-get install nsight-systems-2025.5.2
sudo apt-get install nsight-compute-2025.4.1
```

### Verify Installation

```bash
nsys --version
ncu --version
```

📖 **See [`SYSTEM_REQUIREMENTS.md`](../SYSTEM_REQUIREMENTS.md)** for detailed installation options and troubleshooting.

---

## 📂 Available Profiling Tools

### [Nsight Systems](./nsightSystemProfile/)

GPU timeline profiling to identify system-level bottlenecks:

- **What it profiles**: Full system timeline (CPU, GPU, memory, libraries)
- **Best for**: Identifying gaps, stalls, CPU-GPU sync issues, multi-GPU overlap
- **Output**: `.nsys-rep` timeline trace files
- **Analysis**: Visual timeline in Nsight Systems GUI

**Quick Start:**
```bash
cd nsightSystemProfile
./profile_1b_timeline.sh
```

📖 **[Full Guide](./nsightSystemProfile/README.md)** - Detailed analysis checklist, troubleshooting, and remote-to-local workflow

---

### [Nsight Compute](./kernelProfile/)

Deep kernel-level profiling for instruction-level analysis:

- **What it profiles**: Individual CUDA kernel performance metrics
- **Best for**: Speed of Light analysis, memory/compute bottlenecks, occupancy issues
- **Output**: `.ncu-rep` kernel report files
- **Analysis**: Per-kernel metrics in Nsight Compute GUI

**Quick Start:**
```bash
cd kernelProfile

# Full profiling (slow, comprehensive)
./profile_all_kernels.sh

# Focused profiling (fast, specific kernels)
./profile_focused_kernels.sh
```

📖 **[Full Guide](./kernelProfile/README.md)** - Kernel analysis checklist, SOL metrics, optimization recommendations

---

## 🎯 Which Profiling Tool to Use?

| Goal | Tool | When to Use |
|------|------|-------------|
| **Find GPU idle time** | Nsight Systems | GPU utilization gaps, data loading bottlenecks |
| **Analyze kernel performance** | Nsight Compute | Compute vs memory bound, warp stalls |
| **Multi-GPU debugging** | Nsight Systems | NCCL communication overlap, DDP issues |
| **Memory bandwidth** | Both | Systems: transfers, Compute: coalescing |
| **Kernel fusion opportunities** | Nsight Systems | Many small kernels in timeline |
| **Occupancy optimization** | Nsight Compute | Register/shared memory limits |
| **Cache efficiency** | Nsight Compute | L1/L2 hit rates, memory patterns |

---

## 🔧 Recommended Profiling Workflow

**Phase 1: System-Level Analysis (Nsight Systems)**
1. **Baseline profile** - Run timeline profiling on current setup
2. **Identify bottleneck type** - GPU idle? Kernel overhead? Multi-GPU issues?
3. **Fix system-level issues** - Data loading, kernel fusion, DDP config

**Phase 2: Kernel-Level Optimization (Nsight Compute)**
4. **Profile top kernels** - Deep-dive into slowest kernels from Phase 1
5. **Analyze metrics** - SOL, occupancy, memory/compute bound
6. **Apply kernel optimizations** - Improve coalescing, use tensor cores, etc.

**Phase 3: Validation**
7. **Re-profile** - Run both tools to verify improvements
8. **Compare** - Side-by-side before/after analysis

---

## 📝 Common Optimizations

Based on profiling findings:

### System-Level (from Nsight Systems)

#### GPU Utilization Gaps
- **Issue**: GPU idle while waiting for data
- **Solution**: Increase dataloader workers, use pinned memory, prefetch batches

#### Kernel Launch Overhead
- **Issue**: Many small kernels with high launch latency
- **Solution**: Use `torch.compile()` for kernel fusion

#### CPU-GPU Sync Points
- **Issue**: Blocking synchronization stalls pipeline
- **Solution**: Remove `.cpu()`, `.item()`, use async operations

#### Multi-GPU Communication
- **Issue**: NCCL blocking computation
- **Solution**: Tune DDP bucket configuration, verify gradient overlap

### Kernel-Level (from Nsight Compute)

#### Compute-Bound Kernels
- **Issue**: High SM utilization, low memory throughput
- **Solution**: Use Tensor Cores (BF16/FP16), optimize instruction mix

#### Memory-Bound Kernels
- **Issue**: Low SM utilization, high memory throughput
- **Solution**: Improve coalescing, increase cache hits, reduce transactions

#### Low Occupancy
- **Issue**: <50% occupancy, warp stalls
- **Solution**: Reduce register usage, tune block size, optimize shared memory

#### Poor Cache Performance
- **Issue**: Low L1/L2 hit rates
- **Solution**: Improve data locality, tile operations, reuse data

---

## 📚 Additional Resources

- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

---

## 🚫 .gitignore

Profile trace files are large binary files (100MB-1GB+) and should not be committed:

```gitignore
*.nsys-rep
*.qdrep
*.sqlite
```

This is already configured in subdirectory `.gitignore` files.
