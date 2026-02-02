# 🚀 LLM Architecture Throughput Benchmark Report

> **Date:** February 2, 2026  
> **Purpose:** Compare inference and training throughput across different 1B parameter LLM architecture variants

---

## 📊 Executive Summary

This benchmark evaluates **four architecture variants** of a 1B-class LLM to understand throughput characteristics and infrastructure compatibility.

Because the test machine is an **Intel XPU with ~8.44 GB reported device memory**, one of the variants (`1b_full.yaml`) cannot be run at the full 1B-ish scale on this device (it expands to a much larger parameter count due to enabling multiple advanced features together). To keep this useful for infra bring-up and architectural comparisons, this report includes:

- **Tiny profile (recommended for infra testing):** same architectures, scaled down to fit and run *both inference + training* reliably.
- **Full-size run (where it fits):** the 1B configs that successfully ran on this device.

| Variant | YAML | Attention Type | Notes |
|---------|---------------|------------|----------|
| **Base (GQA)** | `configs/1b_base.yaml` | Grouped Query Attention | Baseline |
| **GSA** | `configs/1b_gsa.yaml` | Gated Sparse Attention | Sparse/gated attention |
| **DeepSeek (MLA)** | `configs/1b_deepseek.yaml` | DeepSeek Sparse (MLA) | KV compression style |
| **Full** | `configs/1b_full.yaml` | GSA + YaRN + mHC + MTP | Too large at full scale on this XPU; benchmarked with tiny profile |

### 🏆 Key Findings (Full-Size, Seq 256)

| Metric | Winner | Value |
|--------|--------|-------|
| **Fastest Inference (prefill)** | Base (GQA) | 818 tokens/s |
| **Fastest Training (fwd+bwd)** | Base (GQA) | 291 tokens/s |
| **Lowest Memory (Inference)** | DeepSeek | 2.36 GB |
| **Lowest Memory (Training)** | DeepSeek | 5.12 GB |
| **Smallest Model** | DeepSeek | 1.15B params |

---

## 🖥️ Hardware Configuration

### GPU Specifications

This snapshot is taken from the `device_info` embedded in the benchmark JSON artifacts.

| Specification | Value |
|---|---|
| Device | Intel(R) Arc(TM) Graphics |
| Backend | XPU via Level-Zero |
| Reported Memory (bytes) | 8444891136 |
| Reported Memory (GB) | 8.44 |
| Execution Units (EUs) | 112 |
| Subslices | 14 |
| Compute Units | 112 |
| FP16 | True |
| FP64 | True |
| Driver Version | 1.6.33184 |
| Runtime | Intel(R) oneAPI Unified Runtime over Level-Zero |

### NVIDIA “Rough Equivalent” (Expectation Setting)

| This System | Rough NVIDIA Class |
|-----------------------------|-------------------|
| Intel Arc Graphics (XPU), ~8.44 GB reported memory | **Entry-level to midrange laptop GPU class** (often compared to GTX 1650 / RTX 3050 Mobile for some workloads) |

> **Note:** Mapping Intel XPU ↔ NVIDIA is inherently approximate and workload-dependent. Treat this as a *sanity/expectation* guide, not a spec-equivalence claim.

---

## ⚙️ Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| **Batch Size** | 1 |
| **Sequence Lengths** | 128, 256 |
| **Data Type** | float16 |
| **Warmup Iterations** | 2 |
| **Benchmark Iterations** | 5 |
| **Framework** | PyTorch XPU backend (Level-Zero). Intel Extension for PyTorch (IPEX) is optional. |

**Important terminology:**
- **"Inference Throughput"** = prompt processing / prefill throughput (a single forward pass over a full input sequence). It does **not** measure autoregressive text generation (decode) with KV-cache reuse.
- **"Training Throughput"** = forward + backward pass. Does **not** include optimizer step or gradient accumulation.

**Statistical note:** Each measurement is the mean of 5 timed iterations after 2 warmup runs. Variance/std-dev is not reported; treat numbers as indicative, not definitive.

### Output Artifacts

- Tiny profile results (comparable across all 4 variants):
  - `results/1b_base_benchmark_tiny.json`
  - `results/1b_gsa_benchmark_tiny.json`
  - `results/1b_deepseek_benchmark_tiny.json`
  - `results/1b_full_benchmark_tiny.json`
- Full-size results (only where it fit):
  - `results/1b_base_benchmark.json`
  - `results/1b_gsa_benchmark.json` *(canonical)*
  - `results/1b_gsa_benchmark_fp16.json` *(earlier run, kept for reference)*
  - `results/1b_deepseek_benchmark.json`

### All Outputs (JSON) Summary

This is a single table summarizing **every JSON output** currently present in `results/`, including the **config name/path** and key model switches used for that run.

| Output | Config | Config Name | Profile | Device | Dtype | Batch | Params | Key config (attn/pos/conn/mtp, hidden, layers) | Seq lens | Inference tok/s (avg) | Train tok/s (avg) | Peak mem GB (max) |
|---|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|
| 1b_base_benchmark.json | configs/1b_base.yaml | LLM-1B-Base | full | xpu | float16 | 1 | 1.166B | grouped_query/rope/residual/False, h=2048, L=24 | 128,256 | 731 | 257 | 5.18 |
| 1b_base_benchmark_tiny.json | configs/1b_base.yaml | LLM-1B-Base (tiny) | tiny | xpu | float16 | 1 | 0.049B | grouped_query/rope/residual/False, h=512, L=6 | 128,256 | 13,040 | 3,650 | 0.33 |
| 1b_deepseek_benchmark.json | configs/1b_deepseek.yaml | LLM-1B-DeepSeek | full | xpu | float16 | 1 | 1.154B | deepseek_sparse/rope/residual/False, h=2048, L=24 | 128,256 | 691 | 246 | 5.13 |
| 1b_deepseek_benchmark_tiny.json | configs/1b_deepseek.yaml | LLM-1B-DeepSeek (tiny) | tiny | xpu | float16 | 1 | 0.050B | deepseek_sparse/rope/residual/False, h=512, L=6 | 128,256 | 11,045 | 3,225 | 0.34 |
| 1b_full_benchmark_tiny.json | configs/1b_full.yaml | LLM-1B-Full (tiny) | tiny | xpu | float16 | 1 | 0.091B | gated_sparse/yarn/mhc/True, h=512, L=6 | 128,256 | 4,194 | 860 | 0.54 |
| 1b_gsa_benchmark.json | configs/1b_gsa.yaml | LLM-1B-GSA | full | xpu | float16 | 1 | 1.219B | gated_sparse/rope/residual/False, h=2048, L=24 | 128,256 | 487 | 209 | 5.29 |
| 1b_gsa_benchmark_fp16.json | configs/1b_gsa.yaml | LLM-1B-GSA | full | xpu | float16 | 1 | 1.219B | gated_sparse/rope/residual/False, h=2048, L=24 | 128,256 | 516 | 130 | 5.29 |
| 1b_gsa_benchmark_tiny.json | configs/1b_gsa.yaml | LLM-1B-GSA (tiny) | tiny | xpu | float16 | 1 | 0.050B | gated_sparse/rope/residual/False, h=512, L=6 | 128,256 | 7,145 | 2,858 | 0.33 |

---

## 📈 Throughput Results (Tiny Profile — Recommended)

These results are the most apples-to-apples comparison across **all four** architectures on the current XPU.

### Model Sizes (Tiny Profile)

| Variant | Params | Hidden | Layers | Attn | Pos | Conn | MTP |
|---|---:|---:|---:|---|---|---|---|
| Base (GQA) | 0.049B | 512 | 6 | grouped_query | rope | residual | False |
| DeepSeek (MLA) | 0.050B | 512 | 6 | deepseek_sparse | rope | residual | False |
| GSA | 0.050B | 512 | 6 | gated_sparse | rope | residual | False |
| Full | 0.091B | 512 | 6 | gated_sparse | yarn | mhc | True |

### Inference Throughput (tokens/s)

| Variant | Seq 128 | Seq 256 | Avg |
|--------|--------:|--------:|----:|
| Base (GQA) | 10,470 | 15,611 | 13,041 |
| DeepSeek (MLA) | 7,778 | 14,312 | 11,045 |
| GSA | 6,323 | 7,968 | 7,145 |
| Full | 3,469 | 4,918 | 4,194 |

### Training Throughput (tokens/s)

| Variant | Seq 128 | Seq 256 | Avg |
|--------|--------:|--------:|----:|
| Base (GQA) | 2,989 | 4,311 | 3,650 |
| DeepSeek (MLA) | 2,606 | 3,844 | 3,225 |
| GSA | 2,395 | 3,322 | 2,858 |
| Full | 781 | 938 | 860 |

---

## 📈 Throughput Results (Full-Size Where Supported)

These are the “true” full-size runs for the models that fit on this machine.

### Inference Throughput (tokens/s)

### Tokens per Second (Higher is Better)

| Architecture | Seq 128 | Seq 256 | Avg Tokens/s |
|--------------|---------|---------|--------------|
| 🥇 **Base (GQA)** | 645 | **818** | **731** |
| 🥈 **DeepSeek (MLA)** | **616** | 767 | 691 |
| 🥉 **GSA** | 441 | 533 | 487 |

```
Inference Throughput (tokens/s) - Sequence Length 256
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Base (GQA)    ████████████████████████████████████████ 818
DeepSeek MLA  █████████████████████████████████████    767
GSA           ██████████████████████████               533
              0        200       400       600       800
```

### Latency (ms per sample) - Lower is Better

| Architecture | Seq 128 | Seq 256 |
|--------------|---------|---------|
| 🥇 **Base (GQA)** | **199 ms** | **313 ms** |
| 🥈 **DeepSeek (MLA)** | 208 ms | 334 ms |
| 🥉 **GSA** | 290 ms | 480 ms |

### Memory Usage (Inference)

| Architecture | Seq 128 | Seq 256 |
|--------------|---------|---------|
| 🥇 **DeepSeek (MLA)** | **2.36 GB** | **2.37 GB** |
| 🥈 **Base (GQA)** | 2.39 GB | 2.40 GB |
| 🥉 **GSA** | 2.49 GB | 2.51 GB |

---

## 🏋️ Training Throughput Results

### Tokens per Second (Higher is Better)

| Architecture | Seq 128 | Seq 256 | Avg Tokens/s |
|--------------|---------|---------|--------------|
| 🥇 **Base (GQA)** | **223** | **291** | **257** |
| 🥈 **DeepSeek (MLA)** | 213 | 279 | 246 |
| 🥉 **GSA** | 178 | 240 | 209 |

```
Training Throughput (tokens/s) - Sequence Length 256
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Base (GQA)    ████████████████████████████████████████ 291
DeepSeek MLA  ██████████████████████████████████████   279
GSA           █████████████████████████████████        240
              0         75        150       225       300
```

### Latency (ms per sample) - Lower is Better

| Architecture | Seq 128 | Seq 256 |
|--------------|---------|---------|
| 🥇 **Base (GQA)** | **575 ms** | **880 ms** |
| 🥈 **DeepSeek (MLA)** | 602 ms | 916 ms |
| 🥉 **GSA** | 720 ms | 1068 ms |

### Memory Usage (Training)

| Architecture | Seq 128 | Seq 256 |
|--------------|---------|---------|
| 🥇 **DeepSeek (MLA)** | **5.12 GB** | **5.13 GB** |
| 🥈 **Base (GQA)** | 5.17 GB | 5.18 GB |
| 🥉 **GSA** | 5.28 GB | 5.29 GB |

---

## 🔬 Architecture Deep Dive

### 1. Base Model (Grouped Query Attention - GQA)

| Specification | Value |
|--------------|-------|
| **Attention Type** | Grouped Query Attention |
| **Parameters** | 1,166,379,008 (1.17B) |
| **Hidden Size** | 2048 |
| **Layers** | 24 |
| **Attention Heads** | 16 |
| **KV Heads** | 4 (4:1 ratio) |
| **Position Encoding** | RoPE |
| **FFN Type** | SwiGLU |

**Strengths:**
- ✅ Fastest inference and training throughput
- ✅ Well-optimized for standard hardware
- ✅ Production-ready architecture (similar to LLaMA 3, Qwen3)

**Trade-offs:**
- ⚠️ Standard KV cache size (no compression)

---

### 2. Gated Sparse Attention (GSA)

| Specification | Value |
|--------------|-------|
| **Attention Type** | Gated Sparse Attention |
| **Parameters** | 1,219,069,952 (1.22B) |
| **Hidden Size** | 2048 |
| **Layers** | 24 |
| **GSA Slots** | 64 |
| **GSA Top-K** | 32 |
| **Position Encoding** | RoPE |
| **FFN Type** | SwiGLU |

**Strengths:**
- ✅ Learned sparse attention patterns
- ✅ Potentially better for very long sequences
- ✅ Novel research architecture

**Trade-offs:**
- ⚠️ ~34% slower inference than Base
- ⚠️ ~4.5% more parameters
- ⚠️ Higher memory usage

---

### 3. DeepSeek MLA (Multi-head Latent Attention)

| Specification | Value |
|--------------|-------|
| **Attention Type** | DeepSeek Sparse (MLA) |
| **Parameters** | 1,153,808,384 (1.15B) |
| **Hidden Size** | 2048 |
| **Layers** | 24 |
| **Compressed KV Dim** | 512 |
| **Position Encoding** | RoPE |
| **FFN Type** | SwiGLU |

**Strengths:**
- ✅ Smallest model size (1.15B vs 1.17B-1.22B)
- ✅ Lowest memory usage (both inference and training)
- ✅ Significant KV cache compression potential (theoretical, not measured here)
- ✅ Strong inference performance (only 5% slower than Base)

**Trade-offs:**
- ⚠️ Slightly slower than Base for training

---

## 📊 Comparative Analysis

### Throughput vs Parameters (Full-Size)

| Architecture | Parameters | Inference (tok/s) | Training (tok/s) | Throughput/Param* |
|--------------|------------|-------------------|------------------|------------------:|
| Base (GQA) | 1.17B | 731 | 257 | **845** |
| DeepSeek | 1.15B | 691 | 246 | 814 |
| GSA | 1.22B | 487 | 209 | 570 |

*Throughput/Param = (Inference + Training) / Params in billions. Higher is better.

### Memory Efficiency Ratio (Full-Size Training)

| Architecture | Params | Train Memory | GB per Billion Params |
|--------------|--------|--------------|----------------------:|
| GSA | 1.22B | 5.29 GB | **4.34** |
| Base | 1.17B | 5.18 GB | 4.43 |
| DeepSeek | 1.15B | 5.13 GB | 4.46 |

*Lower GB/B = more memory-efficient per parameter. DeepSeek has lowest **absolute** memory; GSA is most efficient **per param**.*

---

## 🎯 Recommendations

### For Production Deployment
**👉 Use Base (GQA)**
- Fastest throughput for both inference and training
- Well-tested architecture with broad hardware support
- Similar to production models (LLaMA 3, Qwen3, Mistral)

### For Memory-Constrained Environments
**👉 Use DeepSeek (MLA)**
- Lowest memory footprint
- Smallest parameter count
- Only 5% slower inference than Base
- Ideal for edge deployment

### For Research/Experimentation
**👉 Use GSA**
- Novel sparse attention patterns
- Good for studying attention mechanisms
- May excel on specific tasks requiring sparse attention

---

## 🔄 How to Reproduce

### Run Individual Benchmarks

```bash
# Activate environment
conda activate torch-xpu

# Navigate to project
cd D:\Educational\ERA_v4\Capstone\Testing\LLM\experiments\10_slm_training\llm_architecture

# Run benchmarks
python benchmark_throughput.py configs/1b_base.yaml --device xpu --output results/1b_base_benchmark.json --dtype float16 --batch-size 1 --seq-lengths 128,256

python benchmark_throughput.py configs/1b_gsa.yaml --device xpu --output results/1b_gsa_benchmark.json --dtype float16 --batch-size 1 --seq-lengths 128,256

python benchmark_throughput.py configs/1b_deepseek.yaml --device xpu --output results/1b_deepseek_benchmark.json --dtype float16 --batch-size 1 --seq-lengths 128,256
```

### Available Options

```bash
python benchmark_throughput.py <config.yaml> [OPTIONS]

Options:
  --device {cuda,xpu,cpu}    Device to use (auto-detect if not set)
  --output, -o PATH          Output JSON file for results
  --batch-size INT           Batch size (default: 4)
  --seq-lengths STR          Comma-separated sequence lengths (default: 128,256,512)
  --dtype {float32,float16,bfloat16}  Data type (default: float32)
  --warmup INT               Warmup iterations (default: 3)
  --iters INT                Benchmark iterations (default: 10)
  --inference-only           Skip training benchmark
  --compile                  Use torch.compile (PyTorch 2.0+)
```

---

## 📁 Files Structure

```
results/
├── 1b_base_benchmark.json      # Base GQA results
├── 1b_gsa_benchmark.json       # Gated Sparse Attention results
├── 1b_deepseek_benchmark.json  # DeepSeek MLA results
└── BENCHMARK_REPORT.md         # This report

configs/
├── 1b_base.yaml                # Base GQA configuration
├── 1b_gsa.yaml                 # GSA configuration
├── 1b_deepseek.yaml            # DeepSeek MLA configuration
├── 1b_mhc.yaml                 # Manifold Hyper-Connections
├── 1b_mtp.yaml                 # Multi-Token Prediction
├── 1b_yarn.yaml                # YaRN extended context
└── 1b_full.yaml                # All features combined
```

---

## ⚠️ Limitations & Notes

1. **Integrated GPU**: Results are from an Intel Arc iGPU with shared memory. Dedicated GPUs (RTX 3090, A100) will show significantly different characteristics.

2. **Small Batch Size**: Batch size of 1 was used due to memory constraints. Larger batches on dedicated GPUs would improve throughput.

3. **Short Sequences**: Only 128 and 256 token sequences were tested. Longer sequences may show different relative performance.

4. **FP16 Only**: BF16 is not supported on this hardware. Results may differ with BF16 on compatible hardware.

5. **`1b_full.yaml` at full scale is too large for this XPU**: enabling GSA + YaRN + mHC + MTP together expands parameters dramatically and caused `UR_RESULT_ERROR_DEVICE_LOST` during model move to device. Use `--profile tiny` / `--profile small` to benchmark infra and architectural behavior safely.

---

## 📝 Conclusion

The **Base (GQA)** architecture provides the best overall performance for both inference and training on this hardware configuration. **DeepSeek (MLA)** offers compelling memory efficiency with minimal performance trade-off, making it ideal for memory-constrained deployments. **GSA** provides novel sparse attention capabilities but with notable throughput overhead.

For production workloads on standard hardware, **GQA remains the recommended choice**. For research or edge deployment scenarios, **DeepSeek MLA** presents an attractive alternative.

---

*Report generated by LLM Architecture Benchmark Suite v1.0*
