# ERA V4 Training Compute & Cost Governor

This tool calculates the estimated training time, FLOPs, and cloud costs for Large Language Models (LLMs). It is designed to model **Sparse Mixture-of-Experts (MoE)** architectures and supports "Attention-Aware" compute estimation.

## Features
- **Attention-Aware FLOPs**: Accounts for both linear (FFN/Projections) and quadratic (Attention) compute costs.
- **MoE Support**: accurately models Active vs. Total parameters for sparse models.
- **Cost Governor**: Estimates total cloud training costs based on configurable GPU pricing.
- **Quantization Filtering**: Filter outputs by precision (BF16, FP8, NVFP4).
- **Strict Configuration**: All parameters are defined in `flops_config.json`.

## Usage
1. **Configure**: Edit `flops_config.json` to match your hardware and model specs.
2. **Run**:
   ```bash
   python3 compute_flops.py
   ```

## Configuration (`flops_config.json`)

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
      "num_experts": 80,      // Total Experts
      "top_k_experts": 2,     // Active Experts per token
      "sequence_length": 4096
    }
  }
]
```

## Methodology

### 1. FLOPs Calculation
We use an **Attention-Aware** formula that scales with sequence length:

$$
\text{Total FLOPs} = \text{Linear Term} + \text{Attention Term}
$$

**Per-Sequence FLOPs:**
```python
Linear_Term    = 6 * Sequence_Length * Active_Params
Attention_Term = 12 * Num_Layers * Hidden_Size * (Sequence_Length^2)
Total_Per_Seq  = Linear_Term + Attention_Term
```
*Total Training FLOPs* is then scaled by the total number of sequences (`Total_Tokens / Sequence_Length`).

### 2. Parameter Counting
- **Dense Models**: `Active Params` = `Total Params`.
- **MoE Models**:
    - `Total Params`: Includes weights from **all** experts.
    - `Active Params`: Includes weights from only the **Top-K** experts selected per token.

### 3. Cost Calculation
```python
Cost = (Total_FLOPs / Effective_Cluster_PFLOPS) * Price_Per_GPU_Hour * Num_GPUs
```
