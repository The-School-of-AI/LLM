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
pip install torch>=2.0.0 transformers datasets

# Train with DeepSeek GSA on MPS (Apple Silicon)
python training/train_wikitext2_gpt2.py \
    --preset 1b-deepseek-gsa \
    --device mps \
    --seq-length 256 \
    --batch-size 1 \
    --max-steps 100

# Train with standard GQA on CUDA
python training/train_wikitext2_gpt2.py \
    --preset 1b-base \
    --device cuda \
    --seq-length 512 \
    --batch-size 4 \
    --max-steps 1000
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

### Available Presets

| Preset | Attention | Position | Description |
|--------|-----------|----------|-------------|
| `1b-base` | GQA | YaRN | Standard model with grouped query attention |
| `1b-deepseek-gsa` | DeepSeek GSA | YaRN | **Recommended** - Gated sparse attention with memory optimization |
| `1b-gsa` | GSA | YaRN | Original gated sparse attention |
| `1b-deepseek` | DeepSeek MLA | YaRN | Multi-head latent attention with KV compression |
| `1b-mhc` | GQA | YaRN | Manifold hyper-connections |
| `1b-yarn` | GQA | YaRN | Extended context (32K) |
| `1b-full` | GSA | YaRN | All features enabled |

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
    ├── train.py
    └── train_wikitext2_gpt2.py
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

### Basic Training

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
# WikiText-2 with GPT-2 tokenizer
python training/train_wikitext2_gpt2.py \
    --preset 1b-deepseek-gsa \
    --device auto \
    --seq-length 512 \
    --batch-size 2 \
    --max-steps 1000 \
    --learning-rate 3e-4

# Generic training
python training/train.py \
    --preset 1b-base \
    --max-steps 10000 \
    --batch-size 8 \
    --gradient-accumulation 4
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

# Get preset
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

### Memory Usage (seq_length=256, batch=1)

| Configuration | MPS (M4 Pro) | CUDA |
|---------------|--------------|------|
| 1b-base (GQA) | ~4GB | ~3GB |
| 1b-deepseek-gsa | ~4GB | ~3GB |

## References

- [Gated Sparse Attention](https://arxiv.org/abs/2601.15305v1) - GSA paper
- [YaRN](https://arxiv.org/abs/2309.00071) - Context extension
- [Manifold Hyper-Connections](https://arxiv.org/abs/2512.24880) - mHC paper
- [DeepSeek V3](https://arxiv.org/abs/2412.19437) - MLA and MTP

## License

MIT License
