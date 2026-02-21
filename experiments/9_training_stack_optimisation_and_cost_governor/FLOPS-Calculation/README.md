# ERA V4 Training Compute & Cost Governor

This tool calculates the estimated training time, FLOPs, and cloud costs for Large Language Models (LLMs). It is designed to model **Sparse Mixture-of-Experts (MoE)** architectures and supports "Attention-Aware" compute estimation.

## Features
- **Attention-Aware FLOPs**: Accounts for both linear (FFN/Projections) and quadratic (Attention) compute costs.
- **MoE Support**: accurately models Active vs. Total parameters for sparse models.
- **Cost Governor**: Estimates total cloud training costs based on configurable GPU pricing.
- **Quantization Filtering**: Filter outputs by precision (BF16, FP8, NVFP4).
- **Flexible Configs**: Supports both flat JSON and nested YAML-style JSON.

## Usage
1. **Pick a config**:
   - `config.json` is a **starter template** (single stage).
   - Team presets live under `configs/` (recommended).
2. **Run**:
   ```bash
   python3 compute.py
   ```
   Or specify a preset:
   ```bash
   python3 compute.py --config configs/moe_team8/moe_team8_all_stages.json
   ```

### Sample Configs
Preset-ready configs are available under:
- `experiments/9_training_stack_optimisation_and_cost_governor/FLOPS-Calculation/configs/1b_presets/`
- `experiments/9_training_stack_optimisation_and_cost_governor/FLOPS-Calculation/configs/moe_team8/`

Run any preset with:
```bash
python3 compute.py --config configs/1b_presets/1b_deepseek_gsa.json
```

#### 1B Presets (when to use which)
| File | Use when you want... |
|------|-----------------------|
| `1b_base.json` | Baseline 1B with GQA + YaRN (default). |
| `1b_deepseek_gsa.json` | DeepSeek GSA (recommended sparse attention). |
| `1b_gsa.json` | Original GSA (non‑DeepSeek) variant. |
| `1b_deepseek_mla.json` | DeepSeek MLA (KV compression). |
| `1b_mhc.json` | Manifold hyper‑connections (mHC) variant. |
| `1b_yarn.json` | Extended context (32K) using YaRN. |
| `1b_mtp.json` | Multi‑token prediction (extra LM heads). |
| `1b_full.json` | All features enabled (GSA + MHC + MTP + 32K). |

#### Team‑8 MoE configs
| File | Use when you want... |
|------|----------------------|
| `stage1_1b_dense.json` | Stage‑1 only (1B dense, recurrence + MTP GSA). |
| `stage2_3b_moe.json` | Stage‑2 only (3B MoE, recurrence + MTP GSA). |
| `stage3_8b_moe.json` | Stage‑3 only (8B MoE, recurrence + MTP GSA). |
| `stage4_70b_moe.json` | Stage‑4 only (70B MoE). |
| `moe_team8_all_stages.json` | Combined 1B→70B plan (all stages). |

### Config Formats and Precedence
The calculator accepts **two shapes**. You can use either in `architecture`:

**1) Flat (legacy/simple):**
```json
{
  "architecture": {
    "hidden_size": 2048,
    "num_layers": 16,
    "num_heads": 16,
    "num_kv_heads": 4,
    "attention_type": "gqa"
  }
}
```

**2) Nested (YAML-style, matches team configs):**
```json
{
  "architecture": {
    "hidden_size": 2048,
    "num_layers": 16,
    "attention": {
      "attention_type": "grouped_query",
      "num_attention_heads": 16,
      "num_key_value_heads": 4
    },
    "router": { "top_k": 2 },
    "expert": { "intermediate_size": 512 },
    "head": { "use_multi_token_prediction": false }
  }
}
```

### Attention Types (Quick Reference)

The calculator supports **5 attention types** with optional ratio/parameter notation:

| Type | Format | Examples | Description |
|------|--------|----------|-------------|
| **mha** | `mha` | `"mha"`, `"normal"` | Standard Multi-Head Attention |
| **gqa** | `gqa:Q:KV` | `"gqa"`, `"gqa:4:1"`, `"gqa:8:1"` | Grouped Query Attention |
| **gsa** | `gsa:k` | `"gsa"`, `"gsa:512"`, `"gsa:2048"` | Gated Sparse Attention |
| **dsa** | `dsa:rank` | `"dsa"`, `"dsa:256"`, `"dsa:512"` | DeepSeek MLA (KV compression) |
| **hybrid** | `type1-type2:r1:r2` | `"gqa-gsa:4:1"` | Hybrid (4 type1 layers per 1 type2) |

#### GQA Ratio Notation

For GQA, the ratio format is `gqa:Q:KV` where **Q heads share KV heads**:

| Notation | Meaning | KV Ratio | Example |
|----------|---------|----------|---------|
| `"gqa:4:1"` | 4 Q heads share 1 KV head | 0.25 | Llama-2 style |
| `"gqa:8:1"` | 8 Q heads share 1 KV head | 0.125 | More memory efficient |
| `"gqa:2:1"` | 2 Q heads share 1 KV head | 0.5 | Less aggressive |
| `"gqa"` | Uses `num_kv_heads` from config | varies | Legacy compatibility |

#### GSA Sparse Tokens

For GSA, the format is `gsa:k` where **k = number of tokens to attend to**:

| Notation | Meaning | Typical Use |
|----------|---------|-------------|
| `"gsa:512"` | Attend to 512 tokens | Short context (4K) |
| `"gsa:2048"` | Attend to 2048 tokens | Medium context (32K) |
| `"gsa:4096"` | Attend to 4096 tokens | Long context (128K) |
| `"gsa"` | Uses `sparse_k_tokens` from config | Legacy compatibility |

#### DSA/MLA Compression Rank

For DSA (DeepSeek MLA), the format is `dsa:rank` where **rank = KV LoRA compression rank**:

| Notation | Meaning | Typical Use |
|----------|---------|-------------|
| `"dsa:256"` | Compress KV to rank 256 | Moderate compression |
| `"dsa:512"` | Compress KV to rank 512 | Light compression |
| `"dsa"` | Uses `mla_kv_lora_rank` from config | Legacy compatibility |

#### Hybrid Attention (Layer Mixing)

For models using different attention types across layers (e.g., DeepSeek-V3 style):

| Format | Example | Description |
|--------|---------|-------------|
| `type1-type2:r1:r2` | `"gqa-gsa:4:1"` | 4 GQA layers for every 1 GSA layer |
| `type1-type2:r1:r2:k` | `"gqa-gsa:4:1:512"` | Hybrid with custom sparse k=512 |

**How it works:**
- `"gqa-gsa:4:1"` = 80% GQA layers, 20% GSA layers
- FLOPs are weighted: `0.8 × GQA_FLOPs + 0.2 × GSA_FLOPs`
- Memory includes GSA-specific parameters (gates, indexer)

**Effect on Training:**
| Metric | `gqa:4:1` (Pure GQA) | `gqa-gsa:4:1` (Hybrid) |
|--------|----------------------|------------------------|
| FLOPs | Higher | **Lower** (sparse attention) |
| Params | Baseline | Slightly higher (GSA overhead) |
| Memory | Baseline | Slightly higher |

#### Config Examples

```json
// Standard attention (32 heads, no KV sharing)
"attention_type": "mha"

// GQA with 32 Q heads sharing 4 KV heads (8:1 ratio)
"attention_type": "gqa:8:1"

// GSA with k=512 sparse tokens
"attention_type": "gsa:512"

// DeepSeek MLA with rank=256 compression
"attention_type": "dsa:256"

// Hybrid: 4 GQA layers per 1 GSA layer (DeepSeek-V3 style)
"attention_type": "gqa-gsa:4:1"

// Hybrid with custom sparse k
"attention_type": "gqa-gsa:4:1:1024"

// Legacy format (still works)
"attention_type": "grouped_query",
"num_kv_heads": 4
```

**Precedence rule:** If a value exists both at the top level and inside a nested block,
the **top-level value wins**. Otherwise the nested value is used.

### Growth Mode (Opt‑In)
By default, **tokens are NOT reallocated**. The calculator uses the token counts you provide.
To enable growth reallocation, set:
```json
"growth": { "mode": "paper" }
```
If you omit `growth` or set `mode` to `none`, no reallocation occurs.

## Output Snapshots
We keep simple text snapshots of recent runs:
- `last_run.txt`: baseline run for `flops_config.json`
- `last_run_growth.txt`: growth/expansion run for `flops_config_growth.json`
- `last_run_qwen3style.txt` and `last_run_growth_qwen3style.txt`: exploratory Qwen3-style MoE (many experts, top-k=8, smaller FFN) to approximate a 70B total parameter model

These are **reference outputs only** and should be regenerated after config changes.

## Configuration Reference (DeepSpeed-Compatible)

This calculator uses a config format that mirrors **DeepSpeed's JSON structure**. If you're familiar with DeepSpeed configs, you'll feel right at home.

---

### 🔧 Hardware Settings

```json
"hardware": {
  "num_gpus": 8,                    // Number of GPUs in your cluster
  "price_per_gpu_hour": 2.50,       // Cost per GPU-hour in USD
  "mfu": 0.30,                      // Model FLOPs Utilization (0.0-1.0)
  "quantization": "all",            // "all", "bf16", "fp8", or "nvfp4"
  
  "tflops_per_gpu": {               // Peak TFLOPS per GPU by precision
    "bf16": 989.0,                  // H100 SXM = 989 TFLOPS BF16
    "fp8": 1979.0,                  // H100 SXM = 1979 TFLOPS FP8
    "nvfp4": 3500.0                 // H100 SXM = ~3500 TFLOPS FP4
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_gpus` | int | 8 | Total GPUs in cluster |
| `price_per_gpu_hour` | float | 2.50 | Cost per GPU-hour (USD) |
| `mfu` | float | 0.30 | Model FLOPs Utilization (30% typical) |
| `quantization` | str | "all" | Precision filter: "all", "bf16", "fp8", "nvfp4" |

---

### ⚡ ZeRO Optimization (DeepSpeed-style)

```json
"hardware": {
  "zero_stage": 2,                  // ZeRO Stage: 0, 2, 3, or "infinity"
  
  // ZeRO efficiency factors (how much overhead each stage adds)
  "zero_efficiency": {
    "zero0": 1.0,                   // No sharding overhead
    "zero2": 0.95,                  // 5% overhead for gradient sharding
    "zero3": 0.70,                  // 30% overhead for full sharding
    "zero_infinity": 0.25           // 75% overhead for CPU/NVMe offload
  }
}
```

| ZeRO Stage | What's Sharded | Typical Efficiency | When to Use |
|------------|----------------|-------------------|-------------|
| **0** | Nothing | 100% | Small models, single GPU |
| **2** | Optimizer + Gradients | 95% | **Recommended default** |
| **3** | Optimizer + Gradients + Weights | 70% | Large models (>10B params) |
| **infinity** | Everything + CPU/NVMe offload | 25% | Very large models (>70B) |

---

### 💾 CPU Offload (ZeRO-Infinity)

```json
"hardware": {
  "zero_stage": 3,                  // Must be 3 for offload
  "cpu_offload": true,              // Enable CPU offloading
  
  "cpu_offload_config": {
    "offload_optimizer": true,      // Offload optimizer states to CPU
    "offload_params": true,         // Offload parameters to CPU
    "offload_gradients": true,      // Offload gradients to CPU
    "gpu_buffer_gb": 4.0            // GPU memory reserved for buffers
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cpu_offload` | bool | false | Enable CPU offloading |
| `offload_optimizer` | bool | true | Offload optimizer states (8 bytes/param) |
| `offload_params` | bool | true | Offload model parameters |
| `offload_gradients` | bool | true | Offload gradients |
| `gpu_buffer_gb` | float | 4.0 | GPU memory reserved for compute buffers |

> **Memory Formula:**
> - GPU Memory = `gpu_buffer_gb` + activations + any non-offloaded state
> - CPU Memory = total offloaded state ÷ num_gpus

---

### 🧠 Activation Checkpointing

Activation checkpointing trades compute for memory by recomputing activations during backward pass.

**Option 1: Architecture-level (legacy)**
```json
"architecture": {
  "training": {
    "activation_checkpointing": true,        // Enable checkpointing
    "activation_checkpointing_factor": 0.5,  // % of layers to checkpoint
    "activation_precision": "bf16",          // "bf16", "fp16", "fp32"
    "activation_multiplier": 10.0,           // Hidden vectors per token per layer (see docs)
    "include_activation_memory": true        // Include in memory estimate
  }
}
```

**Option 2: Root-level DeepSpeed format (recommended)**
```json
{
  "activation_checkpointing": {
    "partition_activations": true,    // Shard activations across GPUs
    "cpu_checkpointing": false,       // Offload checkpoints to CPU
    "checkpoint_factor": 0.5          // 1.0=none, 0.5=half, 0.1=aggressive
  },
  "train_micro_batch_size_per_gpu": 16,
  "gradient_accumulation_steps": 8
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `partition_activations` | bool | false | Shard activation memory across data-parallel GPUs |
| `cpu_checkpointing` | bool | false | Offload checkpoints to CPU (slower but saves GPU memory) |
| `checkpoint_factor` | float | 1.0 | Fraction of activations to keep (lower = more savings) |
| `train_micro_batch_size_per_gpu` | int | 1 | Micro batch size (affects activation memory) |
| `gradient_accumulation_steps` | int | 1 | Steps before optimizer update |

> **Memory Savings:** Checkpointing can reduce activation memory by 60-80% at the cost of ~33% more compute.
> 
> **partition_activations:** When enabled, divides activation memory by `num_gpus`. Use with data parallelism.
>
> ⚠️ **Reversible models:** `partition_activations` is **ignored** when `reversible: true`. Reversible training uses a custom `ReversibleMidpointStack` that manages its own states per GPU — they cannot be partitioned by DeepSpeed.

#### Understanding `activation_multiplier`

The `activation_multiplier` controls how many "hidden vectors per token per layer" are stored for the backward pass. The formula used is:

```
activation_bytes = batch × seq × hidden × layers × activation_multiplier × bytes × ckpt_factor
```

**How we calculated the default value (10.0):**

For a SwiGLU transformer layer with GQA, the activations stored for backward pass are:

| Activation | Shape | Units (÷ hidden) |
|------------|-------|------------------|
| Layer input (residual) | `B × S × H` | 1 |
| Post-LayerNorm (pre-attn) | `B × S × H` | 1 |
| Q projection output | `B × S × H` | 1 |
| K projection output | `B × S × kv_dim` | kv_ratio (e.g., 0.25) |
| V projection output | `B × S × kv_dim` | kv_ratio (e.g., 0.25) |
| Attention output | `B × S × H` | 1 |
| Post-LayerNorm (pre-FFN) | `B × S × H` | 1 |
| Gate projection (FFN) | `B × S × I` | intermediate/hidden (e.g., 4) |
| Up projection (FFN) | `B × S × I` | intermediate/hidden (e.g., 4) |

**Typical totals:**

| Scenario | Multiplier | When to use |
|----------|------------|-------------|
| With FlashAttention | **10-15** | Modern training (recommended) |
| Without FlashAttention | **30-50** | Adds `heads × seq / hidden` for attention scores |

**Example calculation (GQA 4:1, intermediate=4×hidden, FlashAttention):**
```
1 + 1 + 1 + 0.25 + 0.25 + 1 + 1 + 4 + 4 = 13.5 → round to 10-15
```

**Without FlashAttention (must store O(seq²) attention scores):**
```
13.5 + (16 heads × 4096 seq / 2048 hidden) = 13.5 + 32 = 45.5 → use 30-50
```

**Configuration examples:**
```json
// With FlashAttention (default, recommended)
"training": {
  "activation_multiplier": 10.0
}

// Without FlashAttention (stores attention scores)
"training": {
  "activation_multiplier": 40.0
}

// Conservative estimate for safety
"training": {
  "activation_multiplier": 15.0
}
```

---

### 🎯 Precision Settings

```json
"architecture": {
  "precision": {
    "weight_precision": "bf16",              // "bf16", "fp16", "fp32", "fp8", "auto"
    "gradient_precision": "fp32",            // "bf16", "fp16", "fp32"
    "optimizer_precision": "fp32",           // Optimizer state precision
    "optimizer_states_count": 2,             // Adam = 2 states (m, v)
    "optimizer_state_multiplier": 1.0,       // Extra state overhead
    
    "master_weights": false,                 // Keep FP32 master weights
    "master_weights_precision": "fp32"       // Master weights precision
  }
}
```

| Precision | Bytes/Param | When to Use |
|-----------|-------------|-------------|
| **bf16** | 2 | Default for training |
| **fp16** | 2 | Alternative mixed precision |
| **fp32** | 4 | Optimizer states, master weights |
| **fp8** | 1 | H100+ inference/training |

---

### 🔀 Parallelism Configuration

```json
"hardware": {
  "parallelism": {
    "tensor_parallel_size": 1,       // TP: Split layers across GPUs
    "pipeline_parallel_size": 1,     // PP: Split model into stages
    "expert_parallel_size": 1,       // EP: Distribute MoE experts
    "data_parallel_size": null       // DP: Auto = num_gpus / (TP×PP×EP)
  }
}
```

| Parallelism | Use When |
|-------------|----------|
| **Data Parallel (DP)** | Always (default) |
| **Tensor Parallel (TP)** | Model too large for single GPU |
| **Pipeline Parallel (PP)** | Very deep models (>100 layers) |
| **Expert Parallel (EP)** | MoE with many experts |

---

### 📡 Communication Settings (Advanced)

```json
"hardware": {
  "communication": {
    "dp_bandwidth_gbps": 200,        // Data parallel bandwidth (NVLink/IB)
    "dp_latency_ms": 0.5,            // DP communication latency
    "dp_comm_multiplier": 1.0,       // DP overhead multiplier
    
    "ep_bandwidth_gbps": 200,        // Expert parallel bandwidth
    "ep_latency_ms": 0.5,            // EP communication latency
    "ep_comm_multiplier": 1.0,       // EP overhead multiplier
    
    "offload_bandwidth_gbps": 256,   // CPU offload bandwidth (PCIe)
    "offload_latency_ms": 1.0,       // Offload latency
    "offload_bytes_per_step": null   // Bytes offloaded per step
  },
  
  "performance": {
    "use_explicit_comm_model": false, // Use detailed comm model
    "compute_mfu": 0.30               // Pure compute MFU
  }
}
```

> **Note:** Set `use_explicit_comm_model: true` to model communication overhead explicitly instead of using the multiplicative efficiency model.

---

### 🏗️ Architecture Settings

```json
"stages": [{
  "name": "1B Dense",
  "total_tokens": 20000000000,        // 20B tokens
  
  "architecture": {
    // Core dimensions
    "vocab_size": 128000,
    "hidden_size": 2048,
    "intermediate_size": 4096,        // FFN hidden size (typically 4× hidden)
    "num_layers": 16,
    "sequence_length": 4096,
    
    // Attention
    "num_heads": 16,
    "num_kv_heads": 4,                // GQA: fewer KV heads
    "attention_type": "gqa:4:1",      // See Attention Types section
    
    // Embeddings
    "tie_embeddings": true,           // Share input/output embeddings
    
    // MoE (if applicable)
    "num_routed_experts": 8,
    "num_shared_experts": 1,
    "top_k": 2,
    "moe_layer_frequency": 1          // MoE every N layers
  }
}]
```

---

### 🧪 Training Settings

```json
"architecture": {
  "training": {
    "micro_batch_size": 1,                   // Batch size per GPU
    "gradient_accumulation_steps": 1,        // Steps before optimizer update
    "activation_precision": "bf16",          // Activation tensor precision
    "activation_multiplier": 10.0,           // Hidden vectors per token per layer
    "activation_checkpointing": false,       // Enable checkpointing
    "activation_checkpointing_factor": 1.0,  // Fraction to checkpoint
    "include_activation_memory": false       // Include in memory calc
  }
}
```

---

### 📋 Complete Example Config

Here's a complete config showing all options:

```json
{
  "hardware": {
    "num_gpus": 8,
    "price_per_gpu_hour": 2.50,
    "mfu": 0.30,
    "quantization": "all",
    "zero_stage": 2,
    "cpu_offload": false,
    "tflops_per_gpu": {
      "bf16": 989.0,
      "fp8": 1979.0
    }
  },
  "stages": [{
    "name": "1B Base",
    "total_tokens": 20000000000,
    "architecture": {
      "vocab_size": 128000,
      "hidden_size": 2048,
      "intermediate_size": 4096,
      "num_layers": 16,
      "num_heads": 16,
      "attention_type": "gqa:4:1",
      "sequence_length": 4096,
      "tie_embeddings": true,
      "precision": {
        "weight_precision": "bf16",
        "optimizer_precision": "fp32"
      },
      "training": {
        "micro_batch_size": 1,
        "activation_checkpointing": false
      }
    }
  }]
}
```

---

### 🚀 Quick Reference: Common Setups

| Use Case | `zero_stage` | `cpu_offload` | `activation_checkpointing` |
|----------|--------------|---------------|---------------------------|
| **1B model, 8 GPUs** | 2 | false | false |
| **7B model, 8 GPUs** | 2 | false | true |
| **13B model, 8 GPUs** | 3 | false | true |
| **70B model, 8 GPUs** | 3 | true | true |
| **70B model, 64 GPUs** | 2 | false | true |

---

### Architecture
Define your training stages. All parameters (Layers, Hidden Size, Experts, etc.) must be specified.
```json
"stages": [
  {
    "name": "70B MoE",
    "total_tokens": 240000000000,
    "architecture": {
      "vocab_size": 64000,
      "hidden_size": 2176,
      "num_layers": 24,
      "num_experts": 80,                             // Total Experts
      "top_k_experts": 2,                            // Active Experts per token
      "num_moe_layers": 24,                          // Optional: MoE layers out of total layers
      "tie_embeddings": true,                        // Optional: LM head tied to embeddings
      "target_total_params": 70000000000,            // Optional: solve to hit total params
      "target_params_per_expert": 8000000000,        // Optional: target total/experts
      "solve_for": "num_experts_from_per_expert",    // Optional: "num_experts" or "num_experts_from_per_expert"
      "sequence_length": 4096
    }
  }
]
```

#### Training / Batch (optional)
You can optionally model **gradient accumulation and activation memory**:
```json
"architecture": {
  "training": {
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "activation_precision": "bf16",
    "activation_multiplier": 10.0,
    "activation_checkpointing": false,
    "activation_checkpointing_factor": 1.0,
    "activation_bytes_per_element": null,
    "include_activation_memory": false
  }
}
```
Notes:
- FLOPs are still token‑based, so GA does **not** change total FLOPs.
- Activation memory is included **only** when `include_activation_memory=true`.
- If `activation_checkpointing=true` and no factor is provided, a default **0.5** multiplier is used.

## Methodology

### 1. FLOPs Calculation
We use an **Attention-Aware** formula that scales with sequence length:

$$
\text{Total FLOPs} = \text{Linear Term} + \text{Attention Term}
$$

**Per-Sequence FLOPs:**
```python
Linear_Term    = 6 * Sequence_Length * Active_NonEmbedding_Params
Attention_Term = 12 * Num_Layers * Hidden_Size * (Sequence_Length^2)
Total_Per_Seq  = Linear_Term + Attention_Term
```
*Total Training FLOPs* is then scaled by the total number of sequences (`Total_Tokens / Sequence_Length`).

#### DeepSeek Sparse Attention (DSA)

For models using sparse attention, the attention term changes from O(L^2) to O(Lk):

**Dense Attention (default):**
```python
Attention_Term = 12 * Num_Layers * Hidden_Size * (Sequence_Length^2)
```

**Sparse Attention (DSA enabled):**
```python
# 1. Lightning Indexer: Fast token selection (still O(L^2) but optimized)
Indexer_FLOPs = (2 * Sequence_Length^2 * Indexer_Heads * Indexer_Dim) / FP8_Speedup

# 2. Sparse Core: Only attend to k selected tokens (O(Lk))
Sparse_Core = 2 * Sequence_Length * k * Head_Dim * Num_Heads   # QK^T
            + 3 * Sequence_Length * k * Num_Heads              # Softmax
            + 2 * Sequence_Length * k * Head_Dim * Num_Heads   # Attn-V

# 3. MLA Projections: Compress KV cache (optional)
MLA_FLOPs = 2 * Sequence_Length * Hidden_Size * KV_Rank * 2    # if enabled

Attention_Term = Num_Layers * (Indexer_FLOPs + Sparse_Core + MLA_FLOPs)
```

**Configuration:**
```json
{
  "architecture": {
    "use_sparse_attention": true,
    "sparse_k_tokens": 2048,       // Number of tokens to attend to (k)
    "indexer_heads": 4,            // Lightning indexer heads (2-4)
    "indexer_dim": 1024,           // Indexer dimension (H/8 typical)
    "mla_kv_lora_rank": 512        // MLA compression rank (0=disabled)
  }
}
```

**Key Points:**
- **Lightning Indexer**: Scores all tokens using FP8 precision and fewer heads for speed
- **Sparse k**: Typically set to L/32 to L/64 (e.g., 2048 for 128K context)
- **MLA**: Compresses key-value cache to reduce memory bandwidth
- **Reduction**: For 4096 context with k=2048, attention FLOPs reduce by ~6-8x

Notes:
- `Active_NonEmbedding_Params` excludes embeddings, but includes the output logits projection even if embeddings are tied.
- If `include_softmax_flops=true`, a small extra softmax term is added.
- If `attention_window` or `attention_sparsity` is set, the attention term uses the reduced effective context.
- `recompute_multiplier` scales total FLOPs for activation checkpointing/recompute.
- `attention_kernel_multiplier` (or `flash_attention_multiplier`) scales the attention term to model faster kernels.
- `quantization_flops_multiplier` scales total FLOPs to model quantization overheads.

### 2. Parameter Counting
- **Dense Models**: `Active Params` = `Total Params`.
- **MoE Models**:
    - `Total Params`: Includes weights from **all** experts.
    - `Active Params`: Includes weights from only the **Top-K** experts selected per token.
- **Embeddings** are counted for parameter totals but excluded from linear FLOPs.
- **Null expert** logic still includes router compute by default (FFN skipped only).
- **MoE capacity** (`moe_capacity_factor`) inflates compute (not parameter counts) to model expert padding/overflow.
- **MoE routing overhead** can be added via `moe_routing_overhead_ratio` (ratio of MoE expert compute)
  or `moe_routing_flops_per_token` (absolute per-token per-layer overhead).

### Optional Architecture Knobs
These keys are optional and default to current behavior if omitted:
- `num_kv_heads`: Enable GQA/MQA parameter counting (defaults to `num_heads`).
- `attention_type`: `gqa`, `gsa`, `deepseek_gsa`, or `deepseek_mla` (MLA uses KV compression).
- `ffn_type`: One of `swiglu`, `geglu`, `glu`, `gelu`, `relu` (defaults to `swiglu`).
- `ffn_multiplier`: Numeric override for FFN matrix count (e.g., 2 or 3).
- `moe_intermediate_size`: Expert FFN intermediate size (defaults to `intermediate_size`).
- `num_routed_experts`: Alias for `num_experts`.
- `num_shared_experts`: Always-active experts per MoE layer.
- `num_null_experts`: Null experts per MoE layer (params are optional and tiny).
- `moe_layer_frequency`: MoE every N layers (used if `num_moe_layers` omitted).
- `lm_head_multiplier`: Multiplies LM head params/FLOPs (e.g., MTP-style multi-heads).
- `router_type`: If set to `gsa`/`gsa_router`, adds extra router params based on `router_dim` and `num_router_heads`.
- `router_dim`, `num_router_heads`: Router projection dimensions (used when `router_type` is GSA-style).
- `moe_capacity_factor`: >1 models MoE padding/overflow compute.
- `attention_window`: Sliding-window attention size.
- `attention_sparsity`: Fraction (0,1] of tokens attended per token (ignored if `attention_window` set).
- `attention_flops_multiplier`: Scales attention FLOPs for kernel speedups/overheads.
- `linear_flops_multiplier`: Scales linear FLOPs (e.g., fused kernels).
- `recompute_multiplier`: Activation checkpointing multiplier on total FLOPs.
- `attention_kernel_multiplier`: Scales attention term for FlashAttention-like kernels.
- `flash_attention_multiplier`: Alias for `attention_kernel_multiplier`.
- `quantization_flops_multiplier`: Scales total FLOPs for quantization overheads.
- `moe_routing_overhead_ratio`: Adds overhead as a ratio of MoE expert compute.
- `moe_routing_flops_per_token`: Adds fixed overhead per-token per-layer.

#### Nested Config Compatibility (YAML-style)
If you pass `attention`, `router`, `expert`, or `head` objects (like your YAMLs), the calculator reads common fields from those too:
- `attention.attention_type`, `attention.num_attention_heads`, `attention.num_key_value_heads`
- `attention.gsa_*`, `attention.indexer_*`, `attention.mla_kv_lora_rank`, `attention.ds_compressed_dim`
- `router.top_k`, `router.top_k_max`, `router.use_adaptive_top_k`, `router.data_sparsity`
- `router.router_type`, `router.num_router_heads`, `router.router_dim`
- `expert.intermediate_size`
- `head.use_multi_token_prediction`, `head.num_prediction_heads`

#### Gated Sparse Attention (GSA)
Use `attention_type: "gsa:k"` where `k` is the number of tokens to attend to:
- `"gsa:32"` - Very sparse (cheap)
- `"gsa:128"` - Moderate sparse (recommended)
- `"gsa:512"` - Less sparse

#### DeepSeek MLA
Set `attention_type: deepseek_mla` and provide either:
- `mla_kv_lora_rank`, or
- `ds_compressed_dim` (alias)
to model KV compression in attention FLOPs.

#### YaRN Position Encoding
YaRN (Yet another RoPE extensioN) has **zero computational overhead** compared to standard RoPE.
The cost of training at longer contexts is automatically captured via `sequence_length` in the O(S²) attention formula.
No additional configuration is needed for YaRN FLOPs accounting.

#### DeltaNet (Gated Linear Attention)
DeltaNet is a linear attention mechanism with **O(S × d²)** complexity instead of O(S² × d) for standard attention.
This makes it dramatically more efficient for long sequences (256k+ tokens).

**Standalone usage:**
```json
{
  "attention_type": "deltanet"
}
```
Aliases: `deltanet`, `delta`, `linear`, `gated_linear`, `gated_deltanet`

**Hybrid DeltaNet-GSA format:**
```json
{
  "attention_type": "deltanet-gsa:3:1:512"
}
```
- `3:1` = 75% DeltaNet layers, 25% GSA layers
- `512` = GSA sparse k tokens

**Complexity comparison:**
| Attention Type | Complexity | 256k Context (d=128) |
|---------------|------------|----------------------|
| Dense | O(S² × d) | Very expensive |
| GSA | O(S × k) + O(S²_indexer) | Fast |
| **DeltaNet** | **O(S × d²)** | **Fastest at long seq** |

Crossover: When S > d², DeltaNet wins. For head_dim=128, crossover at ~16k tokens.

**DeltaNet config options:**
```json
{
  "deltanet": {
    "num_heads": 32,
    "head_dim": 128,
    "conv_size": 4
  }
}
```

#### Kronecker Product Embeddings (KPE)

Instead of a standard `vocab × hidden` lookup table, KPE represents each token as a fixed Kronecker product of character-level and position-level factors, followed by a trainable projection.

**Standard embeddings:** `vocab × hidden` params (e.g., 131072 × 4096 = **537M params**)
**Kronecker embeddings:** `D × hidden + hidden` params (e.g., 8192 × 4096 + 4096 = **33.6M params** — **16× smaller**)

**Configuration:**
```json
{
  "embedding_type": "kronecker",
  "kronecker_config": {
    "char_dim": 256,       // Character embedding dimension
    "pos_dim": 32,         // Position embedding dimension
    "D": 8192              // Product feature dimension (char_dim × pos_dim)
  },
  "tie_embeddings": false  // KPE cannot be tied (forced to false)
}
```

**Parameters:**
| Component | Formula | Example (D=8192, H=4096) |
|-----------|---------|--------------------------|
| `pf_to_model` | D × H | 33,554,432 |
| `embed_norm` (RMSNorm) | H | 4,096 |
| **Total** | **D × H + H** | **33.6M** |

**FLOPs:** Unlike standard lookup (negligible FLOPs), KPE adds a matrix multiply per token:
```
FLOPs = 6 × seq_len × D × hidden  (fwd + bwd-grad + bwd-weight)
```

> **Note:** KPE forces `tie_embeddings: false` since the embedding and LM head have different dimensions.

#### Multi-Head Composition (mHC)

mHC enables **reversible** multi-stream architectures. The hidden state is split into `n_streams` parallel streams, and each sublayer (attention/FFN) uses learnable mixing coefficients to combine streams before and after processing.

**Key benefit:** With reversibility, activations can be **reconstructed from outputs** during backward pass, reducing activation memory from O(layers) to O(1). This is separate from gradient checkpointing.

**Configuration:**
```json
{
  "n_streams": 4    // Number of parallel streams (typically 4)
}
```

**Parameters per sublayer** (2 sublayers per layer: attention + FFN):
| Component | Formula | Example (H=4096, S=4) |
|-----------|---------|------------------------|
| φ_pre (pre-mixing) | S×H × S | 65,536 |
| φ_post (post-mixing) | S×H × S | 65,536 |
| φ_res (residual mixing) | S×H × S² | 262,144 |
| Biases | S + S + S² | 24 |
| Alphas | 3 | 3 |
| RMSNorm | S × H | 16,384 |
| **Per sublayer** | | **409,627** |
| **Per layer** (×2) | | **819,254** |

Where `S = n_streams` and `d_in = S × hidden`.

**Total mHC params** = `layers × 2 × mhc_per_sublayer` (included in both total and active params).

#### Reversible Training (Midpoint/Leapfrog Method)

Reversible training eliminates the need to store intermediate activations for all layers by using a mathematically invertible recurrence (midpoint/leapfrog integration). During the backward pass, activations are recomputed from the final state rather than retrieved from memory.

**Config:**
```json
"architecture": {
  "reversible": true
}
```

**Effects on Calculations:**

| Component | Standard | Reversible |
|-----------|----------|------------|
| Activation Memory | `B × S × H × L × 10 × bytes × ckpt` (O(layers)) | `2 × B × S × n_streams × H × bytes` (O(1)) |
| FLOPs Multiplier | 1.0× (or 1.33× with checkpointing) | 1.33× (recompute during backward) |
| Checkpoint Factor | Configurable | Ignored (not needed) |
| partition_activations | Divides by num_gpus | **Ignored** (custom stack, not DeepSpeed-managed) |

> **n_streams multiplier:** The reversible states are `(B, T, n_streams, H)` not `(B, T, H)`. With `n_streams=4` (mHC), states are **4× larger** than a naive hidden-only estimate.

**Memory Savings:** Activation memory becomes independent of model depth. Only two hidden state tensors (`p_prev`, `p_cur`) are stored regardless of the number of layers. For deep models this can be a **10-20× reduction** in activation memory.

**FLOPs Overhead:** Each layer's forward pass is recomputed once during backward (standard: 6× matmul → reversible: 8× matmul, ratio = 4/3 ≈ 1.33). Override with `reversible_recompute_overhead` in the training config if using partial recomputation.

**Throughput Impact:** The freed memory allows larger `micro_batch_size`, improving GPU utilization. This manifests as higher MFU — adjust `mfu` in the hardware config (e.g., 0.30 → 0.35–0.40) to model this effect.

#### Logits Memory & Chunked Cross-Entropy

For large-vocabulary models, the `lm_head` output tensor `(B, T, vocab_size)` can be the **dominant memory cost** — often exceeding the entire model + optimizer + activations combined. This is especially severe for models with MTP (Multi-Token Prediction), where two logit tensors co-exist simultaneously.

**Example at seq_len=128K, batch=2, vocab=131072 (bf16):**
- NTP logits: `2 × 128000 × 131072 × 2 bytes` = **63.5 GB**
- MTP logits: another **63.5 GB**
- **Peak: ~127 GB** just for logits!

**Fix: Chunked Cross-Entropy Loss**

Instead of materializing the full logits tensor, compute the loss in chunks of `chunk_size` tokens at a time:

```json
"architecture": {
  "chunked_ce_loss_size": 1024
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `chunked_ce_loss_size` | int | 0 | Tokens per chunk for cross-entropy. 0 = full materialization (no chunking). |

**Memory impact:**

| Setting | Logits Memory (seq=128K, B=2, V=131K, bf16) |
|---------|----------------------------------------------|
| `0` (no chunking) | **127 GB** (NTP + MTP) |
| `1024` | **1 GB** |
| `4096` | **4 GB** |

> ⚠️ This config tells `compute.py` how to **estimate** memory. You must also implement chunked CE in the actual training code — the standard approach used by LLaMA, Mistral, and other production LLMs with large vocabularies.

#### Kronecker Embedding Buffer

When `embedding_type: "kronecker"` is set, the model maintains a non-parameter buffer of size `vocab_size × D` (where D is the Kronecker projection dimension, typically 8192). This is always in GPU memory:

```
buffer_bytes = vocab_size × kronecker_D × bytes_per_element
// Example: 131072 × 8192 × 2 = ~2 GB (bf16)
```

This is automatically included in the memory estimate when `embedding_type: "kronecker"` is detected.

#### Memory Stream Recurrence

Memory Stream Recurrence extends the model's effective context to **infinite length** by maintaining a persistent memory state across segments. One of the mHC streams (typically stream 3) is designated as the "memory stream," which accumulates information via a gated exponential moving average across training segments.

**Config:**
```json
"architecture": {
  "recurrence": {
    "enabled": true,
    "stream_idx": 3
  }
}
```

**Parameters added** (per model, ~12K total — negligible):
| Component | Formula | Example (H=4096) |
|-----------|---------|-------------------|
| `lambda_r_raw` | 1 | 1 |
| `memory_ln` (LayerNorm) | 2 × H | 8,192 |
| `memory_gate_proj` (Linear) | H + 1 | 4,097 |
| **Total** | **2H + H + 2** | **12,290** |

**How it works:**
1. The model processes segments of `sequence_length` tokens
2. After each segment, the memory stream state is passed to the next segment
3. `lambda_r = sigmoid(lambda_r_raw)` controls the decay rate (learned)
4. `memory_gate_proj` projects the current memory stream to a gating signal
5. The memory state is updated via: `M_new = lambda_r * M_old + (1 - lambda_r) * gate * current_stream`

> **Note:** Recurrence parameters are included in both total and active parameter counts. They are independent of MoE — even dense models use recurrence when enabled.

#### MTP Attention Type

By default, the MTP (Multi-Token Prediction) block uses DeltaNet attention. However, the recurrence model variants use **GSA** for MTP instead, since MTP runs only once per step and GSA provides better gradient quality at negligible extra cost.

**Config:**
```json
"architecture": {
  "mtp_attention_type": "gsa"
}
```

| Value | Description |
|-------|-------------|
| `"deltanet"` (default) | MTP block uses DeltaNet attention params |
| `"gsa"` | MTP block uses GSA attention params (recommended for recurrence models) |

This affects the parameter count of the MTP block — GSA attention has different projection counts and includes indexer parameters.

#### MoE Upcycling Cost Calculation

When converting a Dense model to MoE, you need to shrink FFN weights to create experts. The calculator supports three upcycling methods:

| Method | FLOPs | Description |
|--------|-------|-------------|
| `slicing` | 0 | Memory copy only (take first N columns) |
| `random_projection` | O(H × src × tgt) | Project via random matrix |
| `svd` | O(min² × max) | Truncated SVD decomposition |

**Usage:**
```bash
# Upcycling comparison only (clean output)
python3 compute.py --config configs/moe_team8/test_upcycling_methods.json --upcycling-only
```

**Configuration:**
```json
{
  "upcycling": {
    "method": "svd",
    "source_intermediate_size": 4096,
    "target_intermediate_size": 1024,
    "num_experts_to_create": 20
  }
}
```

**Trade-offs:** Slicing is fastest but may lose info. Random projection is balanced. SVD is slowest but preserves max variance.

### 3. Cost Calculation

```python
Cost = (Total_FLOPs / Effective_Cluster_PFLOPS) * Price_Per_GPU_Hour * Num_GPUs
```

### 3.1 Precision & Memory Assumptions
By default:
- **BF16 / FP8 / NVFP4** affect **model weights only** (bytes per parameter) and peak TFLOPS.
- **Optimizer states** and **gradients** are assumed **FP32**.

Default bytes per parameter:
- Weights: BF16=2, FP8=1, NVFP4=0.5 (from `quantization`)
- Optimizer states (Adam): 2 states × FP32 = **8 bytes**
- Gradients: FP32 = **4 bytes**

This matches the code in `calculate_memory_per_gpu`.

#### Mixed Precision Overrides
You can override these with optional keys inside `architecture`:

```json
{
  "architecture": {
    "weight_precision": "auto",
    "optimizer_precision": "fp32",
    "optimizer_states_count": 2,
    "optimizer_state_multiplier": 1.0,
    "gradient_precision": "fp32",
    "master_weights": true,
    "master_weights_precision": "fp32"
  }
}
```

`weight_precision: "auto"` means **use the selected quantization** (BF16/FP8/NVFP4)
for model weights instead of hard-coding a precision.

Additional optional overrides:
- `weight_bytes_per_param` (float, overrides weight precision)
- `optimizer_state_bytes_per_param` (float, total bytes across all optimizer states)
- `gradient_bytes_per_param` (float, overrides gradient precision)

These can also be placed under a nested block:
```json
"precision": { ... }
```

### 4. Null Expert Probability
The `null_expert_prob` defines the fraction of tokens that skip the MoE layer (e.g., due to low router confidence or auxiliary-free load balancing).
- **How to determine**:
  1. **Design Choice**: Set a target (e.g., `0.1` for 10% drop rate) to enforce efficiency.
  2. **Profiling**: Run a small ~1B model benchmark, log the router's assignment distribution, and use the observed drop rate.

### 5. Growth / Expansion Mode
In `compute_flops_growth.py`, token allocation uses the same per-token FLOPs formula as
the main calculator, so growth budgets are consistent with attention-aware compute.
By default, `preserve_total_tokens=true` rescales the allocated tokens so the final sum
matches the original total token budget.

## References

1. [DeepSeek-V3: Scaling Open-Source Language Models](https://arxiv.org/abs/2512.02556)
2. [MoE Architecture Specification](https://github.com/The-School-of-AI/LLM/blob/p8/8.5-70B-MoE-Large-Configuration-(The-Explosion)/experiments/8_moe_architecture/moe_arch_spec.md)
3. [Chinchilla Scaling Laws: Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
4. [DeepSpeed ZeRO: Memory Optimization for Large-Scale Training](https://arxiv.org/abs/1910.02054)
