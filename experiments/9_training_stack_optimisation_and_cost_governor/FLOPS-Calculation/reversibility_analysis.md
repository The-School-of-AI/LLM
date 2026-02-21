# Reversible Training: Memory Impact Analysis

## 70B MoE (DeltaNet+GSA) — 8× H200 GPUs (141 GiB/GPU)

Compares GPU memory at different micro batch sizes, with and without reversible training.

> **Paper**: Gal et al., *"Reversing Large Language Models for Efficient Training
> and Fine-Tuning"* ([arXiv:2512.02056v2](https://arxiv.org/abs/2512.02056))

### BF16 Precision

| Micro Batch | Standard (GiB) | Reversible (GiB) | Δ Savings (GiB) | Std Fits? | Rev Fits? |
|:---:|---:|---:|---:|:---:|:---:|
| 1 | 126.9 | 125.4 | 1.5 | ✅ | ✅ |
| 2 | 128.5 | 125.4 | 3.1 | ✅ | ✅ |
| 4 | 131.6 | 125.4 | 6.2 | ✅ | ✅ |
| 8 | 137.9 | 125.5 | 12.4 | ✅ | ✅ |
| 16 | 150.4 | 125.6 | 24.8 | ❌ OOM | ✅ |
| 32 | 175.4 | 125.9 | 49.5 | ❌ OOM | ✅ |
| 64 | 225.4 | 126.4 | 99.0 | ❌ OOM | ✅ |
| 128 | 325.4 | 127.4 | 198.0 | ❌ OOM | ✅ |
| 256 | 525.4 | 129.4 | 396.0 | ❌ OOM | ✅ |
| 512 | 925.4 | 133.4 | 792.0 | ❌ OOM | ✅ |
| 1024 | 1725.4 | 141.4 | 1584.0 | ❌ OOM | ❌ OOM |
| 2048 | 3325.4 | 157.4 | 3168.0 | ❌ OOM | ❌ OOM |

### FP8 Precision

| Micro Batch | Standard (GiB) | Reversible (GiB) | Δ Savings (GiB) | Std Fits? | Rev Fits? |
|:---:|---:|---:|---:|:---:|:---:|
| 1 | 113.6 | 112.1 | 1.5 | ✅ | ✅ |
| 2 | 115.2 | 112.1 | 3.1 | ✅ | ✅ |
| 4 | 118.3 | 112.1 | 6.2 | ✅ | ✅ |
| 8 | 124.5 | 112.2 | 12.4 | ✅ | ✅ |
| 16 | 137.0 | 112.3 | 24.8 | ✅ | ✅ |
| 32 | 162.0 | 112.5 | 49.5 | ❌ OOM | ✅ |
| 64 | 212.0 | 113.0 | 99.0 | ❌ OOM | ✅ |
| 128 | 312.0 | 114.0 | 198.0 | ❌ OOM | ✅ |
| 256 | 512.0 | 116.0 | 396.0 | ❌ OOM | ✅ |
| 512 | 912.0 | 120.0 | 792.0 | ❌ OOM | ✅ |
| 1024 | 1712.0 | 128.0 | 1584.0 | ❌ OOM | ✅ |
| 2048 | 3312.0 | 144.0 | 3168.0 | ❌ OOM | ❌ OOM |

### Key Findings

| Metric | BF16 | FP8 |
|--------|------|-----|
| Max batch (Standard) | **8** | **16** |
| Max batch (Reversible) | **512** | **1024** |
| Batch increase | **64×** | **64×** |

### How Reversibility Works

| Component | Standard | Reversible |
|-----------|----------|------------|
| Activation Memory | `B × S × H × L × 10 × bytes` (O(layers)) | `2 × B × S × H × bytes` (O(1)) |
| FLOPs Overhead | 1.0× | 1.33× (recompute fwd during bwd) |
| Memory Bottleneck | Activations + Weights | Weights only |

Standard training stores activations for all layers — memory grows with batch size AND depth.
Reversible training reconstructs activations during backward using invertible dynamics
(midpoint/leapfrog), storing only 2 hidden-state tensors regardless of depth.
The activation cost becomes negligible; the bottleneck shifts to model weights + optimizer states.
