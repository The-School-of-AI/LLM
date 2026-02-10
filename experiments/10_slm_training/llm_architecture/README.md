# LLM Architecture

A modular, production-ready LLM codebase supporting:
- DeepSeek-GSA architecture variants
- Test_Code-compatible 1B Reference architecture (`1b-reference`) with hybrid DeltaNet + GSA, reversible midpoint integration, mHC-v2, and optional Kronecker embeddings.

## Model Specifications

### 1B-DeepSeek-GSA (Default Configuration)

| Parameter | Value |
|-----------|-------|
| **Total Parameters** | **~1.27B** (with GPT-2 tokenizer) / **~1.39B** (with 128K vocab) |
| Hidden Size | 2,048 |
| Num Layers | 16 |
| Attention Heads | 16 (4 KV heads, GQA 4:1 ratio) |
| Head Dimension | 128 |
| FFN Intermediate Size | 8192 (SwiGLU) |
| Vocab Size | 50,257 (GPT-2) / 128,000 (preset default) |
| Max Position Embeddings | 128,000 (YaRN extended) |
| Precision | bfloat16 |

### 1B-Reference (Test_Code/model_1b.py Compatible)

| Parameter | Value |
|-----------|-------|
| Backbone Layers | 8 |
| MTP Layers | 1 |
| Total Computational Layers | 9 |
| Hidden Size | 4096 |
| Delta/GSA Split | 75% / 25% (6 / 2 layers) |
| DeltaNet Heads | 32 V / 16 QK, head_dim=128 |
| GSA Heads | 16, head_dim=256, d_idx=32 |
| Gate Dim | 384 |
| FFN (Shared / Dense) | 2048 |
| FFN (Routed, future MoE) | 1024 |
| mHC Variant | `mhc_v2` (`alpha=0.1`) |
| Integration | Reversible Midpoint (`step=0.25`, `a=0.5`) |
| Embedding Types | `standard`, `kronecker` |
| Preset | `1b-reference` |

### Parameter Breakdown (Per Layer)

| Component | Parameters | Computation |
|-----------|------------|-------------|
| Q Projection | 4,194,304 | 2048 x 2048 |
| K Projection | 1,048,576 | 2048 x 512 |
| V Projection | 1,048,576 | 2048 x 512 |
| O Projection | 4,194,304 | 2048 x 2048 |
| **Attention Subtotal** | **10,485,760** | |
| Indexer qw_proj (fused) | 532,740 | 2048 x 260 + 260 |
| Indexer k_proj | 131,072 | 2048 x 64 |
| Indexer bias | 4 | 4 heads |
| **Indexer Subtotal** | **663,816** | |
| Value Gate (G2) | 1,049,088 | (2048 x 512) + 512 |
| Output Gate (G1) | 4,196,352 | (2048 x 2048) + 2048 |
| **Gates Subtotal** | **5,245,440** | |
| FFN gate_proj | 1,67,77,216 | 2048 x 8192 |
| FFN up_proj | 1,67,77,216 | 2048 x 8192 |
| FFN down_proj | 1,67,77,216 | 8192 x 2048 |
| **FFN Subtotal** | **5,03,31,648** | |
| RMSNorm (2x) | 4,096 | 2 x 2048 |
| **Per-Layer Total** | **~66.7M** | |
| **All 16 Layers** | **~1.07B** | |

### Global Parameters

| Component | Params (50K vocab) | Params (128K vocab) |
|-----------|-------------------|---------------------|
| Token Embedding | 102.9M | 262.1M |
| Final RMSNorm | 2,048 | 2,048 |
| LM Head (untied) | 102.9M | 262.1M |
| **Global Total** | **~205.9M** | **~524.3M** |

## Features

- **Gated Sparse Attention (GSA)**: O(L*k) complexity instead of O(L^2) for long sequences
- **Gated DeltaNet + Reference GSA Hybrid**: 75/25 layer split for long-context efficiency + sparse quality
- **Reversible Midpoint Stack**: Memory-efficient integration for reference architecture
- **mHC-v2 Connections**: Norm-inside hyper-connections matching Test_Code behavior
- **Kronecker Embeddings (Optional)**: Byte-level PF embeddings with projection to model dimension
- **Triton Kernel Optimization**: Fused RMSNorm and sparse attention kernels for reduced kernel launches
- **Extended Context**: YaRN position embeddings supporting 4K to 256K context
- **Adaptive Sparsity**: Variance-based top-k selection (inverse relationship: high confidence = fewer tokens)
- **Dual Gating**: G1 (output gate) + G2 (value gate) for training stability
- **Cross-Platform**: CUDA (with Triton), MPS (Apple Silicon), and CPU support
- **Mixed Precision**: bfloat16/float16 autocast with gradient scaling

## Quick Start

```bash
# Install dependencies
pip install torch>=2.0.0 transformers datasets pyyaml

# Install Triton for optimized long-context training (CUDA only)
pip install triton

# Train with YAML config (recommended)
python training/train_wikitext2_gpt2.py --config configs/1b_deepseek_gsa.yaml

# Train with YAML config and CLI overrides
python training/train_wikitext2_gpt2.py \
    --config configs/1b_deepseek_gsa.yaml \
    --device cuda \
    --batch-size 4 \
    --seq-length 4096 \
    --max-steps 1000

# Train with preset
python training/train_wikitext2_gpt2.py \
    --preset 1b-deepseek-gsa \
    --device cuda \
    --seq-length 8192 \
    --batch-size 2 \
    --max-steps 100

# Train on Apple Silicon (MPS) with smaller k values for memory
python training/train_wikitext2_gpt2.py \
    --preset 1b-deepseek-gsa \
    --device mps \
    --seq-length 256 \
    --batch-size 1 \
    --gsa-k-base 128 \
    --gsa-k-max 256

# Disable Triton kernels (use PyTorch fallback)
python training/train_wikitext2_gpt2.py \
    --preset 1b-deepseek-gsa \
    --device cuda \
    --no-triton

# run training with torch compile
python training/train_wikitext2_gpt2.py --preset 1b-deepseek-gsa --use-torch-compile --torch-compile-mode max-autotune --device cuda \
    --seq-length 8192 \
    --batch-size 2 \
    --max-steps 100 

# Reference architecture smoke test (standard embedding)
python training/test_wikitext2_reference_gpt2.py \
    --config configs/1b_reference.yaml \
    --tokenizer gpt2 \
    --embedding-type standard \
    --seq-length 1024 \
    --batch-size 1 \
    --max-steps 20 \
    --strict-arch

# Reference architecture with Kronecker embedding
python training/test_wikitext2_reference_gpt2.py \
    --config configs/1b_reference.yaml \
    --tokenizer gpt2 \
    --embedding-type kronecker \
    --seq-length 1024 \
    --batch-size 1 \
    --max-steps 20 \
    --strict-arch

# Reference architecture + torch.compile (recommended compile mode)
python training/test_wikitext2_reference_gpt2.py \
    --config configs/1b_reference.yaml \
    --tokenizer gpt2 \
    --embedding-type standard \
    --seq-length 4096 \
    --batch-size 1 \
    --max-steps 100 \
    --strict-arch \
    --use-torch-compile \
    --torch-compile-mode max-autotune-no-cudagraphs

# Force Flash/Efficient SDPA backend for compile stability
python training/test_wikitext2_reference_gpt2.py \
    --config configs/1b_reference.yaml \
    --tokenizer gpt2 \
    --embedding-type standard \
    --seq-length 4096 \
    --batch-size 1 \
    --max-steps 100 \
    --strict-arch \
    --use-torch-compile \
    --gsa-backend flash

# Force Triton sparse kernel path (eager mode)
python training/test_wikitext2_reference_gpt2.py \
    --config configs/1b_reference.yaml \
    --tokenizer gpt2 \
    --embedding-type standard \
    --seq-length 4096 \
    --batch-size 1 \
    --max-steps 100 \
    --strict-arch \
    --gsa-backend triton

# Long-sequence memory tip (CUDA allocator)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 \
python training/test_wikitext2_reference_gpt2.py --config configs/1b_reference.yaml --tokenizer gpt2 --embedding-type standard --seq-length 4096 --batch-size 1 --max-steps 100 --strict-arch
```

## Architecture Overview

### Forward Pass Flow

```
Input IDs [batch, seq_len]
    |
    v
Token Embedding ──> [batch, seq_len, 2048]
    |
    v
Position IDs ──> torch.arange(0, seq_len)
    |
    v
Attention Mask ──> None for GSA (handles causality internally)
                   Full causal mask for GQA/standard attention
    |
    v
┌──────────────────────────────────────────────────────────────┐
│                  Transformer Block x 16                       │
│                                                               │
│  hidden_states ──> RMSNorm (Triton-fused or PyTorch)         │
│       |                                                       │
│       v                                                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │           DeepSeek Gated Sparse Attention               │  │
│  │                                                         │  │
│  │  Step 1: QKV Projections                                │  │
│  │    Q = W_q(h)  [B, S, 16, 128]                         │  │
│  │    K = W_k(h)  [B, S, 4, 128]   (GQA: 4 KV heads)     │  │
│  │    V = W_v(h)  [B, S, 4, 128]                          │  │
│  │                                                         │  │
│  │  Step 2: Value Gate (G2)                                │  │
│  │    V = V * sigmoid(W_g2(h))   suppress noisy values     │  │
│  │                                                         │  │
│  │  Step 3: YaRN Rotary Position Embedding                 │  │
│  │    Q, K = apply_rotary(Q, K, cos, sin)                  │  │
│  │                                                         │  │
│  │  Step 4: KV Cache (inference only)                      │  │
│  │    K = cat(past_K, K)                                   │  │
│  │    V = cat(past_V, V)                                   │  │
│  │                                                         │  │
│  │  Step 5: Gated Lightning Indexer                        │  │
│  │    qw = W_qw(h)            fused q + weight projection  │  │
│  │    q_idx, weights = split(qw)                           │  │
│  │    k_idx = W_k_idx(h)                                   │  │
│  │    scores = sigmoid(q_idx @ k_idx^T / sqrt(d))          │  │
│  │    scores = scores * sigmoid(weights)                   │  │
│  │    Apply causal mask (future tokens = -inf)             │  │
│  │                                                         │  │
│  │  Step 6: Adaptive Top-K Selection                       │  │
│  │    k_per_query = f(variance)  in [k_min, k_max]         │  │
│  │    HIGH variance = confident = FEWER tokens (inverse)   │  │
│  │    LOW variance = uncertain = MORE tokens               │  │
│  │    indices = topk(scores, k)                            │  │
│  │                                                         │  │
│  │  Step 7: Sparse Attention  O(L*k) not O(L^2)           │  │
│  │    K_sel = gather(K, indices)                           │  │
│  │    V_sel = gather(V, indices)                           │  │
│  │    attn = softmax(Q @ K_sel^T / sqrt(d)) @ V_sel       │  │
│  │    (Triton kernel -> PyTorch chunked -> legacy)         │  │
│  │                                                         │  │
│  │  Step 8: Output Gate (G1)                               │  │
│  │    attn = attn * sigmoid(W_g1(h))                       │  │
│  │                                                         │  │
│  │  Step 9: Output Projection                              │  │
│  │    output = W_o(attn)                                   │  │
│  └────────────────────────────────────────────────────────┘  │
│       |                                                       │
│       + ──> Residual Connection                               │
│       |                                                       │
│       v                                                       │
│  hidden_states ──> RMSNorm                                   │
│       |                                                       │
│       v                                                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                    SwiGLU FFN                           │  │
│  │    gate = SiLU(h @ W_gate)                              │  │
│  │    out  = (gate * (h @ W_up)) @ W_down                  │  │
│  └────────────────────────────────────────────────────────┘  │
│       |                                                       │
│       + ──> Residual Connection                               │
│                                                               │
└──────────────────────────────────────────────────────────────┘
    |
    v
Final RMSNorm ──> [batch, seq_len, 2048]
    |
    v
LM Head (untied) ──> [batch, seq_len, vocab_size]
    |
    v
Cross-Entropy Loss (shifted by 1 position for next-token prediction)
```

### GSA Attention Kernel Dispatch

```
Sparse Attention Dispatch Priority:
  1. Triton Kernel     (CUDA + triton installed + enabled + eager mode)
  2. Flash/Efficient SDPA (compile-friendly path, uses SDPA backend dispatch)
  3. PyTorch Chunked   (optimized sparse gather fallback)
  4. Dense SDPA        (last-resort correctness fallback)

Reference GSA backend controls:
  -> gsa_sparse_backend: auto|triton|pytorch|flash|dense
  -> gsa_use_triton_kernels: enable/disable Triton
  -> gsa_triton_min_seq_len: threshold for Triton in auto mode
  -> gsa_prefer_flash: prefer Flash/Efficient SDPA kernels on CUDA
  -> gsa_sdpa_chunk_size: query chunk size for sparse SDPA gather (lower saves memory)
```

## Model Configurations

### Configuration Methods

The pipeline supports two configuration methods:

**1. YAML Config Files (Recommended)**

YAML files in `configs/` contain both model architecture and training parameters:

```bash
python training/train_wikitext2_gpt2.py --config configs/1b_deepseek_gsa.yaml
python training/train_wikitext2_gpt2.py --config configs/1b_deepseek_gsa.yaml --seq-length 512 --max-steps 500
```

**2. Python Presets**

Programmatic presets for quick experimentation:

```bash
python training/train_wikitext2_gpt2.py --preset 1b-deepseek-gsa --max-steps 5000
```

CLI arguments always override config file values.

### Available Presets

| Preset | Attention | Position | Description |
|--------|-----------|----------|-------------|
| `1b-base` | GQA | YaRN | Standard grouped query attention |
| `1b-deepseek-gsa` | DeepSeek GSA | YaRN | Gated sparse attention (recommended) |
| `1b-deepseek` | DeepSeek MLA | YaRN | Multi-head latent attention with KV compression |
| `1b-mhc` | GQA | YaRN | Manifold hyper-connections |
| `1b-yarn` | GQA | YaRN | Extended context (32K) |
| `1b-mtp` | GQA | YaRN | Multi-token prediction |
| `1b-full` | GSA | YaRN | All features enabled |
| `1b-reference` | DeltaNet + Reference GSA | YaRN | Test_Code-compatible reference architecture |

### DeepSeek GSA Configuration Parameters

```python
# Indexer
gsa_indexer_dim = 64              # d_I: low-dim indexer projection
gsa_num_indexer_heads = 4         # H_I: number of indexer heads
gsa_indexer_activation = "sigmoid" # activation for gated scores

# Adaptive Sparsity
gsa_k_base = 512                  # base selection budget (tokens per query)
gsa_k_min = 64                    # minimum tokens to attend
gsa_k_max = 1024                  # maximum tokens to attend
gsa_use_adaptive_k = True         # enable variance-based adaptive k
gsa_adaptive_k_method = "variance" # method: variance, entropy, learned

# Dual Gating
gsa_use_value_gate = True         # G2: applied to V before attention
gsa_use_output_gate = True        # G1: applied to output after attention
gsa_gate_activation = "sigmoid"
gsa_gate_bias_init = 0.5          # sigmoid(0.5) ~ 0.62

# Triton Optimization
gsa_use_triton_kernels = True     # use Triton kernels when available
gsa_sparse_backend = "auto"       # auto, triton, pytorch, flash, dense
gsa_triton_min_seq_len = 512      # threshold for auto->triton
gsa_prefer_flash = True           # prefer Flash/Efficient SDPA on CUDA
gsa_sdpa_chunk_size = 16          # lower for long-context memory safety
```

### YAML Config Structure

```yaml
model_name: "LLM-1B-DeepSeek-GSA"

# Core architecture
vocab_size: 128000
hidden_size: 2048
num_hidden_layers: 16
max_position_embeddings: 4096

# Attention
attention:
  attention_type: "deepseek_gsa"
  num_attention_heads: 16
  num_key_value_heads: 4
  head_dim: 128

# Position embedding
position:
  position_type: "yarn"
  yarn_original_max_position: 8192
  yarn_scale: 8.0

# Feed-forward
ffn:
  ffn_type: "swiglu"
  intermediate_size: 8192

# Training (optional, overridable via CLI)
training:
  max_steps: 10000
  batch_size: 2
  learning_rate: 3.0e-4
  warmup_steps: 500
```

## Directory Structure

```
llm_architecture/
├── config/
│   └── model_config.py              # Configuration dataclasses and presets
├── configs/                         # YAML configuration files
│   ├── 1b_deepseek_gsa.yaml        # DeepSeek GSA (recommended)
│   ├── 1b_base.yaml                # Base GQA model
│   ├── 1b_gsa.yaml                 # Original GSA
│   ├── 1b_deepseek.yaml            # DeepSeek MLA
│   ├── 1b_mhc.yaml                 # Manifold hyper-connections
│   ├── 1b_mtp.yaml                 # Multi-token prediction
│   ├── 1b_reference.yaml           # Test_Code-compatible reference architecture
│   ├── 1b_yarn.yaml                # Extended context (32K)
│   └── 1b_full.yaml                # All features enabled
├── components/
│   ├── attention/
│   │   ├── deepseek_gsa.py          # DeepSeek GSA (recommended)
│   │   ├── gated_sparse_attention.py # Original GSA
│   │   ├── gated_deltanet.py        # DeltaNet O(N) linear attention
│   │   ├── grouped_query_attention.py # GQA + causal mask utilities
│   │   ├── deepseek_sparse_attention.py # DeepSeek MLA
│   │   └── reference_gsa.py         # Reference GSA used by 1b-reference
│   ├── embeddings/
│   │   ├── token_embedding.py       # Token embedding layer
│   │   ├── rotary_embedding.py      # RoPE with dynamic cache extension
│   │   ├── yarn_embedding.py        # YaRN extended context RoPE
│   │   └── kronecker_embedding.py   # Kronecker embedding support
│   ├── ffn/
│   │   ├── swiglu_ffn.py            # SwiGLU feed-forward network
│   │   └── moe_ffn.py               # Shared+routed MoE FFN for reference stack
│   ├── normalization/
│   │   └── rms_norm.py              # RMSNorm (PyTorch)
│   ├── connections/
│   │   ├── mhc.py                   # Residual & Manifold Hyper-Connections
│   │   └── mhc_v2.py                # mHC-v2 used by reference model
│   ├── integration/
│   │   └── reversible_midpoint.py   # Reversible midpoint stack
│   ├── heads/
│   │   └── multi_token_head.py      # LM Head + Multi-Token Prediction
│   └── kernels/
│       ├── triton_normalization.py   # Fused RMSNorm + residual (Triton)
│       ├── triton_sparse_attn.py     # Sparse attention kernel (Triton + PyTorch)
│       └── triton_indexer.py         # Indexer computation kernel
├── layers/
│   ├── transformer_block.py         # TransformerBlock + TransformerBlockList
│   └── lightning_decoder.py         # Reference decoder + full-transformer MTP block
├── models/
│   ├── llm.py                       # Model factory + default LLM model
│   └── reference_llm.py             # Test_Code-compatible reference model
├── training/
│   ├── train.py                     # Trainer class (AMP, gradient accumulation)
│   ├── train_wikitext2_gpt2.py      # Generic WikiText-2 training script
│   └── test_wikitext2_reference_gpt2.py # Reference architecture training/test script
├── profiling/                       # NSight Systems / NCU profiling scripts
├── configs/                         # YAML model configurations
├── benchmark_throughput.py          # Throughput benchmarking
└── visualize_benchmarks.py          # Benchmark visualization
```

## Training

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to YAML config file | None |
| `--preset` | Model preset (if no --config) | `1b-base` |
| `--tokenizer` | HuggingFace tokenizer | `gpt2` |
| `--max-steps` | Maximum training steps | 200 |
| `--batch-size` | Batch size per device | 2 |
| `--gradient-accumulation` | Gradient accumulation steps | 1 |
| `--seq-length` | Sequence length | 256 |
| `--learning-rate` | Peak learning rate | 3e-4 |
| `--warmup-steps` | LR warmup steps | 20 |
| `--device` | Device: auto, cuda, mps, cpu | auto |
| `--no-triton` | Disable Triton kernels | False |
| `--no-amp` | Disable mixed precision | False |
| `--gsa-k-base` | Override GSA k_base | None |
| `--gsa-k-max` | Override GSA k_max | None |
| `--persistent-workers` | Keep DataLoader workers alive | False |
| `--use-torch-compile` | Enable torch.compile | False |
| `--torch-compile-mode` | Compile mode | `max-autotune-no-cudagraphs` |
| `--seed` | Random seed | 42 |

Reference script (`training/test_wikitext2_reference_gpt2.py`) adds:
- `--strict-arch`
- `--embedding-type {standard,kronecker}`
- `--disable-reversible-checkpoint` (not recommended at long sequence lengths)
- `--torch-compile-fullgraph`, `--torch-compile-dynamic`, `--torch-compile-backend`
- `--no-triton`, `--gsa-backend`, `--gsa-triton-min-seq-len`, `--no-flash-sdpa`, `--gsa-sdpa-chunk-size`

### Training Configuration

```
Optimizer:      AdamW (beta1=0.9, beta2=0.95, eps=1e-8)
LR Schedule:    Cosine decay with linear warmup
Weight Decay:   0.1
Gradient Clip:  1.0
AMP:            bfloat16 (CUDA), float16 (MPS)
```

### Device Support

| Device | AMP Dtype | Triton | Notes |
|--------|-----------|--------|-------|
| CUDA | bfloat16 | Supported | Full support with Triton kernel fusion |
| MPS | float16 | Not available | Apple Silicon, use smaller k values |
| CPU | None | Not available | Fallback only |

### Memory Scaling (GSA k values)

| k_base | k_max | Approx VRAM (seq=4096) | Recommended For |
|--------|-------|----------------------|-----------------|
| 128 | 256 | ~4 GB | MPS / 8GB GPU |
| 256 | 512 | ~8 GB | 16GB GPU |
| 512 | 1024 | ~16 GB | 40GB+ GPU (default) |
| 1024 | 2048 | ~32 GB | 80GB GPU |

### Programmatic Training

```python
from config.model_config import get_preset_config
from models.llm import create_model_from_config
from training.train import Trainer, TrainingConfig

config = get_preset_config("1b-deepseek-gsa")
config.attention.gsa_k_base = 512
config.attention.gsa_k_max = 1024

model = create_model_from_config(config)

training_config = TrainingConfig(
    max_steps=10000,
    batch_size=4,
    gradient_accumulation_steps=4,
    seq_length=2048,
    learning_rate=3e-4,
    warmup_steps=500,
    device="cuda",
    use_amp=True,
)

trainer = Trainer(model, dataloader, training_config, model_config)
trainer.train()
```

### Reference Architecture Commands

```bash
# Baseline reference run (standard embedding)
python training/test_wikitext2_reference_gpt2.py \
  --config configs/1b_reference.yaml \
  --tokenizer gpt2 \
  --embedding-type standard \
  --seq-length 4096 \
  --batch-size 1 \
  --num-workers 0 \
  --max-steps 100 \
  --strict-arch

# Reference + Kronecker embedding
python training/test_wikitext2_reference_gpt2.py \
  --config configs/1b_reference.yaml \
  --tokenizer gpt2 \
  --embedding-type kronecker \
  --seq-length 2048 \
  --batch-size 1 \
  --num-workers 0 \
  --max-steps 100 \
  --strict-arch

# Reference + compile (safe mode for MTP)
python training/test_wikitext2_reference_gpt2.py \
  --config configs/1b_reference.yaml \
  --tokenizer gpt2 \
  --embedding-type standard \
  --seq-length 4096 \
  --batch-size 1 \
  --num-workers 0 \
  --max-steps 100 \
  --strict-arch \
  --use-torch-compile \
  --torch-compile-mode max-autotune-no-cudagraphs
```

Notes:
- Avoid `--disable-reversible-checkpoint` for long contexts (2048+), or memory usage can spike and cause OOM.
- For MTP runs, `max-autotune-no-cudagraphs` is preferred over `max-autotune`.

## API Reference

### LLM Model

```python
class LLM(nn.Module):
    def forward(
        self,
        input_ids: torch.LongTensor,          # [batch, seq_len]
        attention_mask: Optional[torch.Tensor], # [batch, seq_len] or None
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[Tuple],       # KV cache for inference
        labels: Optional[torch.LongTensor],     # Target IDs for loss
        use_cache: bool = False,
    ) -> LLMOutput

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.LongTensor
```

### Configuration

```python
from config.model_config import (
    ModelConfig,
    AttentionConfig,
    PositionConfig,
    FFNConfig,
    ConnectionConfig,
    HeadConfig,
    get_preset_config,
    PRESET_CONFIGS,
)

# From YAML
config = ModelConfig.from_dict(yaml_data)

# From preset
config = get_preset_config("1b-deepseek-gsa")

# Modify
config.attention.gsa_k_base = 512
config.max_position_embeddings = 131072  # 128K
```

## Performance

### Benchmarking

To run the throughput benchmark for the full architecture:

```bash
python benchmark_throughput.py configs/1b_full.yaml --seq-lengths 512,1024,2048
```

### Latest Benchmark Report

```text
================================================================================
🚀 SOTA LLM THROUGHPUT BENCHMARK
================================================================================
Device: cuda (NVIDIA A100-SXM4-80GB)
PyTorch: 2.9.0+cu126
CUDA: 12.6 | Memory: 85.2GB
Dtype: bfloat16 | Batch: 4 | Sequences: [512, 1024, 2048]
================================================================================

================================================================================
📋 CONFIG: 1b_full.yaml
================================================================================
Model: LLM-1B-Full
  Hidden: 2048 | Layers: 24
  Attention: deepseek_gsa | Heads: 16/4 | HeadDim: 128
  FFN: swiglu (5504) | Position: yarn
  Connection: mhc | MTP: True

🔧 Loading model...
Initialized LLM-1B-Full
  Parameters: 1.74B
  Attention: deepseek_gsa
  Connection: mhc
  Position: yarn
  MTP: True

📊 Parameters: 1,738,509,776 (1.739B)
  Embedding: 103,022,592 | Attention: 398,390,088 | FFN: 816,513,672

────────────────────────────────────────────────────────────────────────────────
📈 INFERENCE (Forward Pass)
────────────────────────────────────────────────────────────────────────────────
Seq      Tok/s        Samp/s     Lat(ms)      P95(ms)    Mem(GB)    TFLOPS  
────────────────────────────────────────────────────────────────────────────────
512      7,031        13.73      291.30       293.69     4.15       20.81   
1024     5,285        5.16       775.02       777.45     4.38       15.64   
2048     3,193        1.56       2565.27      2566.87    4.83       9.45    

────────────────────────────────────────────────────────────────────────────────
🏋️ TRAINING (Forward + Backward)
────────────────────────────────────────────────────────────────────────────────
Seq      Tok/s        Samp/s     Lat(ms)      P95(ms)    Mem(GB)    TFLOPS  
────────────────────────────────────────────────────────────────────────────────
512      5,941        11.60      344.70       365.76     23.46      52.76   
1024     4,677        4.57       875.74       893.76     43.02      41.54   
2048     OOM

────────────────────────────────────────────────────────────────────────────────
💡 INSIGHTS
────────────────────────────────────────────────────────────────────────────────
Bottleneck: SUB_OPTIMAL: MFU of 18.4% - optimization opportunities exist
Memory: GSA k_base=2048: sparse attention for long sequences
Throughput: Seq scaling: 21% throughput drop from 512 to 1024 tokens
Architecture: DeepSeek GSA: variance adaptive k

────────────────────────────────────────────────────────────────────────────────
📊 SEQUENCE SCALING ANALYSIS
────────────────────────────────────────────────────────────────────────────────
  Memory Scaling: Quadratic (O(n²)) (exponent: 0.11)
  Throughput Scaling Exponent: -0.57
  Scaling Efficiency: 63.72%

────────────────────────────────────────────────────────────────────────────────
🏗️ ARCHITECTURE BREAKDOWN
────────────────────────────────────────────────────────────────────────────────
  Attention: deepseek_gsa (KV reduction: 4.0x)
  Position: yarn (max context: 32768)
  Connection: mhc
  MTP: Enabled (4 tokens)
  Triton Kernels: Enabled (k=2048)
  Param Distribution: Embed 5.9% | Attn 22.9% | FFN 47.0% | Head 24.2%

================================================================================
🏁 BENCHMARK COMPLETE
================================================================================
```

### Design Decisions

- **Untied embeddings**: Input and output embeddings are separate (following DeepSeek V3) for better quality at scale
- **GSA skips causal mask**: GSA/DeepSeek-GSA attention handles causality internally via the indexer, avoiding the full N x N causal mask allocation
- **Fused indexer projection**: q_proj + weight_proj fused into single `qw_proj` linear layer for fewer kernel launches
- **Chunked indexer for long sequences**: When seq_q x seq_kv > 16M, processes in chunks of 256 to avoid OOM
- **Triton RMSNorm**: Fuses variance + rsqrt + multiply + residual into a single kernel, reducing memory bandwidth by ~50%

## References

- [Gated Sparse Attention (GSA)](https://arxiv.org/abs/2601.15305v1) - Core attention mechanism
- [DeepSeek V3](https://arxiv.org/abs/2512.02556v1) - Architecture and training insights
- [YaRN](https://arxiv.org/abs/2309.00071) - Context length extension
- [Manifold Hyper-Connections](https://arxiv.org/abs/2512.24880) - mHC connections
- [Triton](https://triton-lang.org/) - GPU kernel optimization
