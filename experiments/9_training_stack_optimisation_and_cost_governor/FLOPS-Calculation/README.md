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
|------|-----------------------|
| `stage1_1b_dense.json` | Stage‑1 only (dense). |
| `stage2_3b_moe.json` | Stage‑2 only (3B MoE). |
| `stage3_8b_moe.json` | Stage‑3 only (8B MoE). |
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

## Configuration Format

### Hardware & Cost
```json
"hardware": {
  "num_gpus": 8,
  "mfu": 0.30,                // Model FLOPs Utilization (e.g., 30%)
  "price_per_gpu_hour": 2.50, // Cost per GPU/Hour in USD
  "quantization": "all",      // Options: "all", "bf16", "fp8"
  "tflops_per_gpu": { ... }   // Peak TFLOPS for each precision
}
```

#### Parallelism & Communication (optional)
If you want to model **communication overhead explicitly**, you can provide:
```json
"hardware": {
  "parallelism": {
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "expert_parallel_size": 1,
    "data_parallel_size": null   // auto = num_gpus / (tp*pp*ep)
  },
  "communication": {
    "dp_bandwidth_gbps": 200,
    "dp_latency_ms": 0.5,
    "dp_comm_multiplier": 1.0,
    "ep_bandwidth_gbps": 200,
    "ep_latency_ms": 0.5,
    "ep_comm_multiplier": 1.0,
    "offload_bandwidth_gbps": 256,
    "offload_latency_ms": 1.0,
    "offload_bytes_per_step": null
  },
  "performance": {
    "use_explicit_comm_model": false,
    "compute_mfu": 0.30
  }
}
```
Defaults reflect a **typical IB 200G cluster** plus **PCIe Gen4 offload**. Adjust for your actual fabric.
When `use_explicit_comm_model=true`, the calculator:
- uses `compute_mfu` for **pure compute**
- **adds explicit DP/EP/offload communication time** on top
Otherwise, it uses the simpler multiplicative efficiency model:
```
effective_mfu = mfu × zero_efficiency × scaling_efficiency
```

#### CPU Offload (ZeRO‑Infinity)
When `hardware.cpu_offload=true`, you can choose **what gets offloaded** and how much **GPU buffer** to reserve:
```json
"hardware": {
  "cpu_offload": true,
  "cpu_offload_config": {
    "offload_params": true,
    "offload_optimizer": true,
    "offload_gradients": true,
    "gpu_buffer_gb": 4.0
  }
}
```
Notes:
- Offload moves **parameters, optimizer states, and/or gradients** to CPU/NVMe.
- GPU memory becomes **buffer + any non‑offloaded state**.
- CPU memory is reported **per GPU** (total offloaded state ÷ number of GPUs).
- Throughput penalty is modeled via `zero_efficiency.zero_infinity`.

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
    "activation_multiplier": 2.0,
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

#### Gated Sparse Attention (GSA) / DeepSeek GSA
When `attention_type` is `gsa` or `deepseek_gsa`, the calculator uses sparse attention (O(Lk)) and supports:
- `gsa_k_base`, `gsa_k_min`, `gsa_k_max`: Base and clamp bounds for adaptive k.
- `gsa_k_tokens`: Explicit k (overrides base/min/max).
- `gsa_use_adaptive_k`: Enable adaptive k (uses base/min/max).
- `gsa_num_indexer_heads`, `gsa_indexer_dim`: Lightning indexer config.
- `gsa_use_value_gate`, `gsa_use_output_gate`: Adds G2/G1 gate projection params.
- `indexer_fp8_speedup`: FP8 speedup factor for indexer (default 2.0).

#### DeepSeek MLA
Set `attention_type: deepseek_mla` and provide either:
- `mla_kv_lora_rank`, or
- `ds_compressed_dim` (alias)
to model KV compression in attention FLOPs.

#### YaRN (Position) Overhead
If you are using YaRN from the start and want to model any extra overhead beyond
sequence length, set:
```json
"position": {
  "position_type": "yarn",
  "yarn_flops_multiplier": 1.0
}
```
`yarn_flops_multiplier` scales the **attention** FLOPs term when `position_type` is `yarn`.

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
