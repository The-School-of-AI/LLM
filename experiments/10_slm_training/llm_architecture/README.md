# LLM Architecture

A modular, production-ready 1B parameter Large Language Model with state-of-the-art attention mechanisms including DeepSeek-style Gated Sparse Attention.

## Features

- **Multiple Attention Mechanisms**: GQA, Gated Sparse Attention (GSA), DeepSeek GSA, DeepSeek MLA
- **Extended Context**: YaRN position embeddings for 4K → 32K+ context
- **Cross-Platform**: CUDA, MPS (Apple Silicon), and CPU support
- **Memory Efficient**: Adaptive sparse attention with configurable memory modes
- **Production Ready**: Mixed precision, gradient checkpointing, KV caching

## Quick Start

```bash
# Install dependencies
pip install torch>=2.0.0 transformers datasets pyyaml

# Train with YAML config (recommended)
python training/train_wikitext2_gpt2.py --config configs/1b_deepseek_gsa.yaml

# Train with YAML config and CLI overrides
python training/train_wikitext2_gpt2.py \
    --config configs/1b_base.yaml \
    --device cuda \
    --batch-size 4 \
    --max-steps 1000

# Train with preset (legacy mode)
python training/train_wikitext2_gpt2.py \
    --preset 1b-deepseek-gsa \
    --device mps \
    --seq-length 256 \
    --batch-size 1 \
    --max-steps 100
```

## Architecture

```
Input IDs
    │
    ▼
┌─────────────────────┐
│  Token Embedding    │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Transformer Block  │ × 16
│  ┌───────────────┐  │
│  │ RMSNorm       │  │
│  │ Attention     │  │  ← GQA / GSA / DeepSeek GSA
│  │ Connection    │  │  ← Residual / mHC
│  │ RMSNorm       │  │
│  │ SwiGLU FFN    │  │
│  │ Connection    │  │
│  └───────────────┘  │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Final RMSNorm      │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  LM Head            │
└─────────────────────┘
    │
    ▼
Logits
```

## Model Configurations

### Configuration Methods

The pipeline supports **two configuration methods**:

#### 1. YAML Config Files (Recommended)

YAML files in `configs/` contain both model architecture and training parameters:

```bash
# Use complete config from YAML
python training/train.py --config configs/1b_deepseek_gsa.yaml

# Override specific values via CLI
python training/train.py --config configs/1b_gsa.yaml --batch-size 8 --learning-rate 1e-4
```

#### 2. Python Presets (Legacy)

Programmatic presets for quick experimentation:

```bash
python training/train.py --preset 1b-base --max-steps 5000
```

### Available Configurations

| Preset | Attention | Position | Description |
|--------|-----------|----------|-------------|
| `1b-base` | GQA | YaRN | Standard model with grouped query attention |
| `1b-deepseek-gsa` | DeepSeek GSA | YaRN | **Recommended** - Gated sparse attention with memory optimization |
| `1b-gsa` | GSA | YaRN | Original gated sparse attention |
| `1b-deepseek` | DeepSeek MLA | YaRN | Multi-head latent attention with KV compression |
| `1b-mhc` | GQA | YaRN | Manifold hyper-connections |
| `1b-yarn` | GQA | YaRN | Extended context (32K) |
| `1b-mtp` | GQA | YaRN | Multi-token prediction |
| `1b-full` | GSA | YaRN | All features enabled |

### YAML Config Structure

Each YAML config file contains:

```yaml
# Model identification
model_name: "LLM-1B-Base"
model_version: "1.0.0"

# Core architecture
vocab_size: 50304
hidden_size: 2048
num_hidden_layers: 24
max_position_embeddings: 4096

# Component configs
attention:
  attention_type: "grouped_query"  # Options: grouped_query, gated_sparse, deepseek_gsa, deepseek_sparse
  num_attention_heads: 16
  num_key_value_heads: 4
  # ... more attention params

position:
  position_type: "rope"  # Options: rope, yarn
  # ... position params

ffn:
  ffn_type: "swiglu"  # Options: swiglu, gelu, moe
  # ... ffn params

connection:
  connection_type: "residual"  # Options: residual, mhc
  # ... connection params

head:
  use_multi_token_prediction: false
  # ... head params

# Training configuration (optional, can be overridden via CLI)
training:
  max_steps: 10000
  batch_size: 8
  gradient_accumulation_steps: 4
  seq_length: 1024
  learning_rate: 3.0e-4
  warmup_steps: 500
  device: "auto"
  checkpoint_dir: "./checkpoints/1b_base"
  experiment_name: "1b_base"
```

### Base Configuration

```python
from config.model_config import get_preset_config

config = get_preset_config("1b-base")
# Parameters:     ~0.57B
# Hidden Size:    2048
# Layers:         16
# Attention Heads: 16 (4 KV heads)
# Vocabulary:     128,000
# Max Context:    4,096
```

### DeepSeek GSA Configuration (Recommended)

```python
config = get_preset_config("1b-deepseek-gsa")
```

**Key Features:**
- Gated Lightning Indexer with proper 1/sqrt(d) scaling
- Adaptive Top-K selection with inverse variance relationship
- Dual gating: G1 (output gate) + G2 (value gate)
- Memory-efficient mode for MPS/limited VRAM
- Indexer key caching for efficient decoding

**Configuration Parameters:**

```python
# Indexer
gsa_indexer_dim = 64              # Indexer projection dimension
gsa_num_indexer_heads = 4         # Number of indexer heads
gsa_indexer_activation = "sigmoid" # Activation function

# Sparsity (Memory-optimized defaults for MPS)
gsa_k_base = 128                  # Base selection budget
gsa_k_min = 32                    # Minimum tokens to attend
gsa_k_max = 256                   # Maximum tokens to attend
gsa_use_adaptive_k = True         # Enable adaptive k selection
gsa_adaptive_k_method = "variance" # Method: variance, entropy, learned

# Gating
gsa_use_value_gate = True         # G2: Applied after V projection
gsa_use_output_gate = True        # G1: Applied after attention
gsa_gate_activation = "sigmoid"
gsa_gate_bias_init = 0.5
```

**For CUDA with more VRAM**, increase k values:
```python
config.attention.gsa_k_base = 512
config.attention.gsa_k_max = 1024
```

## Components

### Directory Structure

```
llm_architecture/
├── config/
│   └── model_config.py           # Configuration classes and presets
├── configs/                      # YAML configuration files
│   ├── 1b_base.yaml             # Base GQA model
│   ├── 1b_deepseek_gsa.yaml     # DeepSeek GSA (recommended)
│   ├── 1b_deepseek.yaml         # DeepSeek MLA
│   ├── 1b_gsa.yaml              # Original GSA
│   ├── 1b_mhc.yaml              # Manifold hyper-connections
│   ├── 1b_mtp.yaml              # Multi-token prediction
│   ├── 1b_yarn.yaml             # Extended context (32K)
│   └── 1b_full.yaml             # All features enabled
├── components/
│   ├── attention/
│   │   ├── grouped_query_attention.py    # GQA
│   │   ├── gated_sparse_attention.py     # Original GSA
│   │   ├── deepseek_gsa.py               # DeepSeek GSA (recommended)
│   │   └── deepseek_sparse_attention.py  # DeepSeek MLA
│   ├── embeddings/
│   │   ├── token_embedding.py
│   │   ├── rotary_embedding.py           # RoPE
│   │   └── yarn_embedding.py             # YaRN extended context
│   ├── ffn/
│   │   └── swiglu_ffn.py
│   ├── normalization/
│   │   └── rms_norm.py
│   ├── connections/
│   │   └── mhc.py                        # Residual & Manifold Hyper-Connections
│   └── heads/
│       └── multi_token_head.py
├── layers/
│   └── transformer_block.py
├── models/
│   └── llm.py
└── training/
    ├── train.py                  # Main training script
    └── train_wikitext2_gpt2.py   # WikiText-2 smoke test
```

### Attention Mechanisms

#### 1. Grouped Query Attention (GQA)
Standard attention with KV head sharing for memory efficiency.

```python
AttentionConfig(
    attention_type=AttentionType.GROUPED_QUERY,
    num_attention_heads=16,
    num_key_value_heads=4,  # 4:1 ratio
)
```

#### 2. DeepSeek Gated Sparse Attention
**Recommended for training.** Implements the GSA paper (arXiv:2601.15305v1) with corrections.

```python
AttentionConfig(
    attention_type=AttentionType.DEEPSEEK_GSA,
    gsa_indexer_dim=64,
    gsa_num_indexer_heads=4,
    gsa_k_base=128,
    gsa_k_min=32,
    gsa_k_max=256,
)
```

**How it works:**
1. **Gated Lightning Indexer**: Computes importance scores for each token pair
2. **Adaptive Top-K Selection**: Selects k most important tokens (k varies based on score variance)
3. **Sparse Attention**: Computes attention only over selected tokens
4. **Dual Gating**: G2 (value gate) + G1 (output gate) for stability

**Memory Modes:**
- `auto`: Automatically selects based on device
- `fast`: Batched processing (CUDA with 40GB+ VRAM)
- `memory_efficient`: Sequential processing (MPS, limited VRAM)

#### 3. DeepSeek MLA (Multi-head Latent Attention)
KV compression for reduced memory footprint.

```python
AttentionConfig(
    attention_type=AttentionType.DEEPSEEK_SPARSE,
    ds_compressed_dim=512,
    ds_rope_head_dim=32,
)
```

### Position Embeddings

#### YaRN (Yet another RoPE extensioN)
Extends context length beyond training length.

```python
PositionConfig(
    position_type=PositionEmbeddingType.YARN,
    yarn_original_max_position=4096,
    yarn_scale=8.0,          # 4K → 32K
    yarn_beta_fast=32.0,
    yarn_beta_slow=1.0,
    yarn_mscale=1.0,
)
```

### Connections

#### Residual (Default)
```python
ConnectionConfig(connection_type=ConnectionType.RESIDUAL)
```

#### Manifold Hyper-Connections (mHC)
From paper arXiv:2512.24880.

```python
ConnectionConfig(
    connection_type=ConnectionType.MHC,
    mhc_expansion_rate=4,
    mhc_alpha_init=0.01,
)
```

## Training

### Config-Driven Training (Recommended)

The recommended approach is to use YAML config files that contain both model and training configurations:

```bash
# Full training with YAML config
python training/train.py --config configs/1b_deepseek_gsa.yaml

# WikiText-2 smoke test with config
python training/train_wikitext2_gpt2.py --config configs/1b_base.yaml

# Override specific parameters
python training/train.py \
    --config configs/1b_gsa.yaml \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --device cuda
```

**CLI arguments always override config file values.**

### Available CLI Overrides

| Argument | Description |
|----------|-------------|
| `--config` | Path to YAML config file |
| `--preset` | Model preset (if --config not provided) |
| `--max-steps` | Maximum training steps |
| `--batch-size` | Batch size per device |
| `--gradient-accumulation` | Gradient accumulation steps |
| `--seq-length` | Sequence length |
| `--learning-rate` | Peak learning rate |
| `--warmup-steps` | LR warmup steps |
| `--device` | Device: auto, cuda, mps, cpu |
| `--experiment-name` | Experiment name for logging |
| `--checkpoint-dir` | Checkpoint directory |
| `--seed` | Random seed |
| `--log-interval` | Logging interval (steps) |
| `--save-interval` | Checkpoint interval (steps) |

### Basic Training (Programmatic)

```python
from config.model_config import get_preset_config
from models.llm import LLM
from training.train import Trainer, TrainingConfig

# Load configuration
model_config = get_preset_config("1b-deepseek-gsa")

# Create model
model = LLM(model_config)

# Training configuration
training_config = TrainingConfig(
    max_steps=10000,
    batch_size=4,
    gradient_accumulation_steps=4,
    seq_length=1024,
    learning_rate=3e-4,
    warmup_steps=500,
    device="auto",  # cuda, mps, or cpu
    use_amp=True,
)

# Train
trainer = Trainer(model, dataloader, training_config, model_config)
trainer.train()
```

### Command Line Training

```bash
# YAML config mode (recommended)
python training/train_wikitext2_gpt2.py \
    --config configs/1b_deepseek_gsa.yaml \
    --device auto

# YAML config with overrides
python training/train.py \
    --config configs/1b_base.yaml \
    --max-steps 10000 \
    --batch-size 8 \
    --gradient-accumulation 4

# Preset mode (legacy)
python training/train_wikitext2_gpt2.py \
    --preset 1b-deepseek-gsa \
    --device auto \
    --seq-length 512 \
    --batch-size 2 \
    --max-steps 1000 \
    --learning-rate 3e-4
```

### Device Support

| Device | AMP Support | Notes |
|--------|-------------|-------|
| CUDA | bfloat16, float16 | Full support, GradScaler for float16 |
| MPS | float16 | Apple Silicon, memory-efficient attention mode |
| CPU | None | Fallback, no mixed precision |

## API Reference

### LLM Model

```python
class LLM(nn.Module):
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple] = None,
        labels: Optional[torch.LongTensor] = None,
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

# Method 1: Load from YAML (recommended)
config = ModelConfig.load("configs/1b_deepseek_gsa.yaml")

# Method 2: Get preset
config = get_preset_config("1b-deepseek-gsa")

# Modify
config.attention.gsa_k_base = 256
config.max_position_embeddings = 8192

# Save/Load
config.save("config.json")
config = ModelConfig.load("config.json")
```

## Performance

### Model Size (~0.57B parameters)

| Component | Parameters |
|-----------|------------|
| Token Embedding | 262M |
| Attention (×16) | 134M |
| FFN (×16) | 201M |
| Other | ~3M |


## References

- [Gated Sparse Attention](https://arxiv.org/abs/2601.15305v1) - GSA paper
- [YaRN](https://arxiv.org/abs/2309.00071) - Context extension
- [Manifold Hyper-Connections](https://arxiv.org/abs/2512.24880) - mHC paper
- [DeepSeek V3](https://arxiv.org/abs/2412.19437) - MTP


