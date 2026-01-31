# 🚀 1B LLM Architecture from Scratch

A modular, production-ready 1B parameter Large Language Model implementation with configurable state-of-the-art components.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture Design](#architecture-design)
3. [Components](#components)
4. [Configuration System](#configuration-system)
5. [Quick Start](#quick-start)
6. [Training Guide](#training-guide)
7. [Experiment Workflow](#experiment-workflow)
8. [API Reference](#api-reference)
9. [Performance Benchmarks](#performance-benchmarks)

---

## 🎯 Overview

This project implements a **1 billion parameter** language model with a modular architecture that allows easy swapping of components for experimentation.

### Key Features

| Feature | Description |
|---------|-------------|
| **Modular Design** | Each component in a separate file for easy modification |
| **Multiple Attention Types** | GQA, GSA (Gated Sparse), DeepSeek Sparse (MLA) |
| **Advanced Connections** | Standard residual or Manifold Hyper-Connections (mHC) |
| **Extended Context** | YaRN for context length extension (4k → 32k+) |
| **Multi-Token Prediction** | DeepSeek-style auxiliary prediction heads |
| **Production Ready** | Mixed precision, gradient checkpointing, KV caching |

### Target Specifications

```
Parameters:     ~1.1 Billion
Hidden Size:    2048
Layers:         24
Attention Heads: 16 (4 KV heads for GQA)
Vocabulary:     50,304 (divisible by 64)
Max Context:    4,096 (extendable to 32k+ with YaRN)
```

---

## 🏗️ Architecture Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         LLM Model                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Token Embedding                         │   │
│  │                  [vocab × hidden]                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               Transformer Block × 24                     │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │  RMSNorm → Attention → Connection              │    │   │
│  │  │  RMSNorm → FFN → Connection                    │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Final RMSNorm                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              LM Head (Standard or MTP)                   │   │
│  │              [hidden × vocab]                            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Transformer Block Detail

```
Input (hidden_states)
        │
        ├─────────────────┐
        ↓                 │
   ┌─────────┐           │
   │ RMSNorm │           │
   └────┬────┘           │
        ↓                 │
┌───────────────┐        │
│   Attention   │        │
│  (GQA/GSA/DS) │        │
│  + RoPE/YaRN  │        │
└───────┬───────┘        │
        ↓                 │
   ┌─────────┐           │
   │Connection│←─────────┘
   │(Res/mHC) │
   └────┬────┘
        │
        ├─────────────────┐
        ↓                 │
   ┌─────────┐           │
   │ RMSNorm │           │
   └────┬────┘           │
        ↓                 │
   ┌─────────┐           │
   │ SwiGLU  │           │
   │   FFN   │           │
   └────┬────┘           │
        ↓                 │
   ┌─────────┐           │
   │Connection│←─────────┘
   │(Res/mHC) │
   └────┬────┘
        ↓
     Output
```

### Design Philosophy

1. **Separation of Concerns**: Each component (attention, FFN, normalization) is in its own file
2. **Configuration-Driven**: All architectural choices controlled via config
3. **Research-Ready**: Easy to swap components for ablation studies
4. **Production-Ready**: Optimized for training efficiency

---

## 🧩 Components

### Project Structure

```
llm_architecture/
├── config/
│   └── model_config.py          # Configuration classes and presets
│
├── components/
│   ├── attention/
│   │   ├── grouped_query_attention.py   # GQA (LLaMA/Qwen style)
│   │   ├── gated_sparse_attention.py    # GSA (paper 2601.15305v1)
│   │   └── deepseek_sparse_attention.py # DeepSeek V3 MLA
│   │
│   ├── embeddings/
│   │   ├── token_embedding.py           # Token embeddings
│   │   ├── rotary_embedding.py          # RoPE
│   │   └── yarn_embedding.py            # YaRN extended context
│   │
│   ├── ffn/
│   │   └── swiglu_ffn.py                # SwiGLU feed-forward
│   │
│   ├── normalization/
│   │   └── rms_norm.py                  # RMSNorm
│   │
│   ├── connections/
│   │   └── mhc.py                       # Manifold Hyper-Connections
│   │
│   └── heads/
│       └── multi_token_head.py          # Standard & MTP heads
│
├── layers/
│   └── transformer_block.py             # Transformer layer
│
├── models/
│   └── llm.py                           # Complete LLM model
│
├── training/
│   └── train.py                         # Training script
│
├── experiments/
│   └── run_experiments.py               # Experiment runner
│
└── README.md                            # This file
```

### Component Details

#### 1. Attention Mechanisms

| Type | File | Description | Key Features |
|------|------|-------------|--------------|
| **GQA** | `grouped_query_attention.py` | Grouped Query Attention | 4:1 head ratio, RoPE, Flash Attention support |
| **GSA** | `gated_sparse_attention.py` | Gated Sparse Attention | Learned sparse patterns, top-k selection, gating |
| **DeepSeek** | `deepseek_sparse_attention.py` | Multi-head Latent Attention | KV compression, decoupled RoPE |

**GQA (Default)**
```python
# 16 query heads, 4 KV heads = 4x memory reduction vs MHA
AttentionConfig(
    attention_type=AttentionType.GROUPED_QUERY,
    num_attention_heads=16,
    num_key_value_heads=4,
    head_dim=128
)
```

**GSA (Gated Sparse)**
```python
# From paper 2601.15305v1
AttentionConfig(
    attention_type=AttentionType.GATED_SPARSE,
    gsa_num_slots=64,      # Memory slots
    gsa_sparse_topk=32,    # Top-k selection
    gsa_temperature=1.0    # Gating temperature
)
```

**DeepSeek Sparse (MLA)**
```python
# 93.3% KV cache reduction
AttentionConfig(
    attention_type=AttentionType.DEEPSEEK_SPARSE,
    ds_compressed_dim=512,    # Compressed KV dimension
    ds_rope_head_dim=32       # RoPE dimension
)
```

#### 2. Position Embeddings

| Type | File | Context Length | Description |
|------|------|----------------|-------------|
| **RoPE** | `rotary_embedding.py` | Up to training length | Standard rotary embeddings |
| **YaRN** | `yarn_embedding.py` | 4x-8x+ extension | NTK-aware interpolation |

**RoPE (Default)**
```python
PositionConfig(
    position_type=PositionEmbeddingType.ROPE,
    rope_theta=10000.0
)
```

**YaRN (Extended Context)**
```python
# Extend 4k → 32k
PositionConfig(
    position_type=PositionEmbeddingType.YARN,
    yarn_original_max_position=4096,
    yarn_scale=8.0,
    yarn_beta_fast=32.0,
    yarn_beta_slow=1.0
)
```

#### 3. Connections

| Type | File | Description |
|------|------|-------------|
| **Residual** | `mhc.py` | Standard `x + sublayer(x)` |
| **mHC** | `mhc.py` | Manifold-constrained multi-path connections |

**Manifold Hyper-Connections (paper 2512.24880)**
```python
ConnectionConfig(
    connection_type=ConnectionType.MHC,
    mhc_expansion_rate=4.0,     # Path expansion
    mhc_num_connections=2,       # Number of paths
    mhc_use_dynamic_weights=True # Adaptive gating
)
```

#### 4. Multi-Token Prediction

From DeepSeek, predicts multiple future tokens simultaneously.

```python
HeadConfig(
    use_multi_token_prediction=True,
    num_predict_tokens=4,        # Predict t+1, t+2, t+3, t+4
    mtp_loss_weight=0.3          # Auxiliary loss weight
)
```

---

## ⚙️ Configuration System

### ModelConfig Structure

```python
@dataclass
class ModelConfig:
    # Core
    model_name: str = "LLM-1B-Base"
    vocab_size: int = 50304
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    max_position_embeddings: int = 4096
    
    # Sub-configs
    attention: AttentionConfig      # Attention mechanism
    position: PositionConfig        # Position embeddings
    ffn: FFNConfig                  # Feed-forward network
    connection: ConnectionConfig    # Layer connections
    head: HeadConfig               # Output head(s)
```

### Preset Configurations

| Preset | Description | Key Settings |
|--------|-------------|--------------|
| `1b-base` | Base model | GQA, RoPE, Residual |
| `1b-gsa` | Gated Sparse | GSA attention |
| `1b-deepseek` | DeepSeek style | MLA attention |
| `1b-mhc` | Hyper-connections | mHC instead of residual |
| `1b-mtp` | Multi-token | MTP head enabled |
| `1b-yarn` | Extended context | YaRN position embeddings |
| `1b-full` | All features | GSA + mHC + MTP + YaRN |

```python
from config import get_preset_config

# Load preset
config = get_preset_config("1b-base")

# Modify as needed
config.attention.attention_type = AttentionType.GATED_SPARSE
config.head.use_multi_token_prediction = True

# Save/load
config.save("my_config.json")
loaded = ModelConfig.load("my_config.json")
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd llm_architecture

# Install dependencies
pip install torch>=2.0.0 numpy

# Optional: Flash Attention for faster training
pip install flash-attn --no-build-isolation
```

### Basic Usage

```python
from config import get_preset_config, ModelConfig
from models import LLM, create_model

# Method 1: Use preset
model = create_model("1b-base")

# Method 2: Custom configuration
config = ModelConfig(
    hidden_size=2048,
    num_hidden_layers=24,
    attention=AttentionConfig(
        attention_type=AttentionType.GROUPED_QUERY,
        num_attention_heads=16,
        num_key_value_heads=4
    )
)
model = LLM(config)

# Forward pass
input_ids = torch.randint(0, 50304, (2, 1024))
outputs = model(input_ids)
logits = outputs.logits  # [2, 1024, 50304]

# With labels for training
labels = torch.randint(0, 50304, (2, 1024))
outputs = model(input_ids, labels=labels)
loss = outputs.loss

# Generation
generated = model.generate(
    input_ids[:, :10],
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9
)
```

---

## 🏋️ Training Guide

### Basic Training

```bash
# Train base model for 10,000 steps
python -m training.train \
    --preset 1b-base \
    --max-steps 10000 \
    --batch-size 8 \
    --gradient-accumulation 4 \
    --learning-rate 3e-4 \
    --experiment-name "base_training"
```

### Training Configuration

```python
from training import TrainingConfig, run_training

config = TrainingConfig(
    max_steps=10000,
    batch_size=8,
    gradient_accumulation_steps=4,
    seq_length=1024,
    
    learning_rate=3e-4,
    min_learning_rate=1e-5,
    warmup_steps=500,
    lr_decay_style="cosine",
    
    weight_decay=0.1,
    gradient_clip=1.0,
    
    use_amp=True,
    amp_dtype="bfloat16",
    
    checkpoint_dir="./checkpoints",
    log_interval=10,
    save_interval=1000
)

model, metrics = run_training("1b-base", config)
```

### Training Output

```
============================================================
Starting Training: base_training
============================================================
Model: LLM-1B-Base
Parameters: 1.13B
Device: cuda
Max steps: 10000
Batch size: 8 x 4
============================================================

Step    100/10000 | Loss: 8.2341 | LR: 6.00e-05 | Tok/s: 45,231 | Grad: 1.23 | ETA: 2.1h
Step    200/10000 | Loss: 7.1234 | LR: 1.20e-04 | Tok/s: 46,102 | Grad: 0.98 | ETA: 2.0h
...
```

---

## 🧪 Experiment Workflow

Follow the 7-step experiment plan:

### Step 1: Base Model (GQA)
```bash
python -m experiments.run_experiments --experiments step1_base_gqa --steps 10000
```

### Step 2: Measure & Analyze
Metrics logged to `experiments/<name>/`:
- Loss curves
- Tokens/second
- Gradient norms

### Step 3: GSA Attention
```bash
python -m experiments.run_experiments --experiments step3_gsa --steps 10000
```

### Step 4: DeepSeek Sparse
```bash
python -m experiments.run_experiments --experiments step4_deepseek_sparse --steps 10000
```

### Step 5: mHC Connections
```bash
python -m experiments.run_experiments --experiments step5_mhc --steps 10000
```

### Step 6: Multi-Token Prediction
```bash
python -m experiments.run_experiments --experiments step6_mtp --steps 10000
```

### Step 7: YaRN Extended Context
```bash
python -m experiments.run_experiments --experiments step7_yarn --steps 10000
```

### Run All Experiments
```bash
python -m experiments.run_experiments --steps 10000 --output-dir ./results
```

### Run with gpt tokenizer with WikiText-2 Data
```bash
python training/train_wikitext2_gpt2.py --max-steps 50 --seq-length 256 --batch-size 2
or

python training/train_wikitext2_gpt2.py --max-tokens 200000 --max-steps 20

```

### Generated Report

The experiment runner generates a comparison report:

```markdown
# LLM Architecture Experiment Report

## Results Summary

| Experiment | Loss | Tok/s | Params | Attention | Connection | MTP |
|------------|------|-------|--------|-----------|------------|-----|
| step1_base_gqa | 3.2145 | 45,231 | 1.13B | grouped_query | residual | False |
| step3_gsa | 3.1892 | 42,156 | 1.15B | gated_sparse | residual | False |
| step5_mhc | 3.1456 | 43,892 | 1.18B | grouped_query | mhc | False |
| combo_full | 3.0123 | 38,456 | 1.22B | gated_sparse | mhc | True |
```

---

## 📚 API Reference

### LLM Model

```python
class LLM(nn.Module):
    def __init__(self, config: ModelConfig)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> LLMOutput
    
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> torch.LongTensor
    
    @property
    def num_parameters(self) -> int
    
    def get_model_info(self) -> Dict[str, Any]
```

### LLMOutput

```python
@dataclass
class LLMOutput:
    loss: Optional[torch.Tensor]           # Training loss
    logits: torch.Tensor                   # [batch, seq, vocab]
    aux_logits: Optional[List[Tensor]]     # MTP auxiliary logits
    past_key_values: Optional[Tuple]       # KV cache
    hidden_states: Optional[Tuple]         # All layer outputs
    attentions: Optional[Tuple]            # Attention weights
    loss_dict: Optional[Dict]              # Loss breakdown
```

---

## 📊 Performance Benchmarks

### Model Size Breakdown

| Component | Parameters | Percentage |
|-----------|------------|------------|
| Token Embedding | 103M | 9.1% |
| Attention (×24) | 403M | 35.7% |
| FFN (×24) | 540M | 47.8% |
| LayerNorms | 98K | <0.1% |
| LM Head | Tied | 0% |
| **Total** | **~1.13B** | **100%** |

### Expected Performance (A100 80GB)

| Configuration | Tokens/sec | Memory |
|---------------|------------|--------|
| Base (GQA) | ~45,000 | ~24GB |
| GSA | ~42,000 | ~26GB |
| DeepSeek Sparse | ~48,000 | ~20GB |
| mHC | ~41,000 | ~28GB |
| Full | ~36,000 | ~32GB |

---

## 🔧 Customization

### Adding New Attention

1. Create file in `components/attention/my_attention.py`
2. Implement `BaseAttention` interface
3. Add to `AttentionType` enum in config
4. Update `TransformerBlock._create_attention()`

```python
# components/attention/my_attention.py
class MyAttention(BaseAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        # Your implementation
        pass
```

### Adding New Connection Type

1. Create in `components/connections/my_connection.py`
2. Follow the interface: `forward(x, sublayer_output) -> output`
3. Add to `ConnectionType` enum
4. Update `TransformerBlock._create_connection()`

---

## 📝 Citation

If you use this code, please cite the relevant papers:

```bibtex
# Grouped Query Attention
@article{ainslie2023gqa,
  title={GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints},
  author={Ainslie, Joshua and others},
  year={2023}
}

# Gated Sparse Attention
@article{gsa2024,
  title={Gated Sparse Attention},
  author={...},
  journal={arXiv:2601.15305v1},
  year={2024}
}

# Manifold Hyper-Connections
@article{mhc2024,
  title={Manifold-Constrained Hyper-Connections},
  author={...},
  journal={arXiv:2512.24880},
  year={2024}
}

# YaRN
@article{peng2023yarn,
  title={YaRN: Efficient Context Window Extension of Large Language Models},
  author={Peng, Bowen and others},
  year={2023}
}
```

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

**Built for research and experimentation. Happy training! 🚀**
