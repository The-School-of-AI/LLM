# MoE Architecture - Team 8

## Mixture of Experts Architecture for LLM Development

A production-ready implementation of Mixture of Experts (MoE) transformer architecture, featuring GSA-style routing, loss-free load balancing, and comprehensive telemetry.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Growth Cadence](#growth-cadence)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [Components](#components)
7. [Team Integration](#team-integration)
8. [CUDA Kernels](#cuda-kernels)
9. [Training Guide](#training-guide)
10. [Expansion Guide](#expansion-guide)
11. [API Reference](#api-reference)
12. [Mathematical Foundations](#mathematical-foundations)

---

## 🌟 Overview

This package implements a state-of-the-art MoE transformer supporting the 4-stage growth cadence:

| Stage | Model | Parameters | Experts | Top-K | Key Transition |
|-------|-------|------------|---------|-------|----------------|
| 1 | 1B Dense | 1.0B | — | — | Foundation |
| 2 | 3B MoE-8 | 3.0B | 8 | 2 | Expert Explosion (1→8) |
| 3 | 8B MoE-8 | 8.0B | 8 | 2 | Dimension Scaling |
| 4 | 70B MoE-64 | 70B | 64 | 4 | Expert Expansion (8→64) |

### Key Features

- **GSA-Style Router**: Multi-head sigmoid routing with adaptive sparsity
- **Dual Gating (G1+G2)**: Collapse prevention from GSA paper
- **Null Experts**: Zero-compute pathway for junk token absorption
- **Loss-Free Load Balancing**: Bias-only adjustment (no auxiliary loss)
- **Team 7 Telemetry**: Comprehensive routing health monitoring
- **CUDA Kernels**: High-performance Triton implementations

---

## 🏗 Architecture

### Model Architecture

```
Input IDs
    ↓
┌─────────────────────────────────────┐
│         Token Embedding             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│     Transformer Layer × N           │
│  ┌─────────────────────────────┐   │
│  │   RMSNorm → GQA Attention   │   │
│  │   (with RoPE)               │   │
│  └─────────────────────────────┘   │
│              ↓ + residual          │
│  ┌─────────────────────────────┐   │
│  │   RMSNorm → MoE Block       │   │
│  │   (or Dense FFN)            │   │
│  └─────────────────────────────┘   │
│              ↓ + residual          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│    RMSNorm → LM Head → Logits      │
└─────────────────────────────────────┘
```

### MoE Block Architecture

```
                    Input
                      ↓
┌─────────────────────────────────────────────────────────────┐
│                  GSA Router                                  │
│   1. Query projection: h → q (low-dim, per head)            │
│   2. Head weights: h → w (query-dependent importance)        │
│   3. Expert affinity: Σⱼ wⱼ · σ(qⱼ · expert_key + bⱼ)       │
│   4. Bias adjustment: affinity + bias (load balancing)       │
│   5. Top-k selection: select k experts                       │
│   6. Gating weights: normalize(original_scores[selected])    │
└─────────────────────────────────────────────────────────────┘
          ↓                              ↓
┌──────────────────┐           ┌──────────────────────────────┐
│  Shared Experts  │           │      Routed Experts          │
│  (always active) │           │  (selected by router)        │
│                  │           │                              │
│  ┌────┐ ┌────┐  │           │  ┌────┐ ┌────┐ ... ┌────┐   │
│  │ S1 │ │ S2 │  │           │  │ E1 │ │ E2 │     │ ∅  │   │
│  └────┘ └────┘  │           │  └────┘ └────┘     └────┘   │
└──────────────────┘           │     (Routed)      (Null)    │
          ↓                    └──────────────────────────────┘
          │                              ↓
          │              weighted by gating_weights
          └──────────────┬───────────────┘
                         ↓
                      Output
```

---

## 📈 Growth Cadence

### Stage 1: 1B Dense (Foundation)

**Purpose**: Establish base capabilities and weight distributions.

```python
config = get_config('1b_dense')
# hidden_size: 2048
# num_layers: 24
# intermediate_size: 5504
# No MoE
```

### Stage 2: 3B MoE-8 (Learn Routing)

**Purpose**: Learn expert routing patterns with 8 experts.

**Transition**: Copy 1B FFN to all 8 experts (explosion).

```python
config = get_config('3b_moe')
# hidden_size: 2048 (same as 1B)
# num_layers: 24 (same as 1B)
# num_routed_experts: 8
# num_shared_experts: 2
# num_null_experts: 1
# top_k: 2
```

**Mathematical basis**:
- `N_experts = 8` (minimum for meaningful specialization)
- `K = √8 ≈ 2.8 → 2` (DeepSeek √N rule)
- `N_shared = 2` (25% of effective capacity)
- `N_null = ⌈0.20 × 2 / 0.6⌉ = 1`

### Stage 3: 8B MoE-8 (Scale Dimensions)

**Purpose**: Scale model capacity while preserving routing patterns.

**Transition**: Interpolate weights to 2× dimensions (same 8 experts).

```python
config = get_config('8b_moe')
# hidden_size: 4096 (2× larger)
# num_layers: 40 (2× more)
# intermediate_size: 2048 (2× larger)
# num_routed_experts: 8 (SAME!)
# top_k: 2 (SAME!)
```

### Stage 4: 70B MoE-64 (Expert Expansion)

**Purpose**: Fine-grained specialization with 64 experts.

**Transition**: Each of 8 experts → 8 children (8×8=64).

```python
config = get_config('70b_moe')
# hidden_size: 4096 (same as 8B)
# num_layers: 40
# num_routed_experts: 64 (8× expansion)
# num_shared_experts: 4
# num_null_experts: 2
# top_k: 4 (√64 × 0.5 = 4)
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository_url>
cd moe_architecture

# Install dependencies
pip install torch triton
```

### Basic Usage

```python
from moe_architecture import get_config, create_model

# Create 3B MoE model
config = get_config('3b_moe')
model = create_model(config)

# Forward pass
input_ids = torch.randint(0, config.vocab_size, (2, 128))
outputs = model(input_ids)
logits = outputs['logits']  # [batch, seq, vocab]
```

### Command Line

```bash
# Show configuration
python -m moe_architecture --config 3b_moe --info

# Run tests
python -m moe_architecture --config 3b_moe --test

# Run forward pass
python -m moe_architecture --config 3b_moe --run

# Export checkpoint
python -m moe_architecture --config 3b_moe --export model.pt
```

```bash
# How to run 3B parameter Config
python main.py --config 3b_moe --action summary
python main.py --config 3b_moe --action create --device cpu --output /tmp/3b_test.pt
```
---


## ⚙️ Configuration

### Configuration Files

| File | Description |
|------|-------------|
| `configs/config_1b_dense.py` | Stage 1: 1B Dense |
| `configs/config_3b_moe.py` | Stage 2: 3B MoE-8 |
| `configs/config_8b_moe.py` | Stage 3: 8B MoE-8 |
| `configs/config_70b_moe.py` | Stage 4: 70B MoE-64 |

### Custom Configuration

```python
from moe_architecture.config import MoEModelConfig, GSARouterConfig, ExpertConfig

config = MoEModelConfig(
    model_name="custom_moe",
    hidden_size=2048,
    num_layers=24,
    
    expert=ExpertConfig(
        num_routed_experts=16,
        num_shared_experts=2,
        num_null_experts=1,
        intermediate_size=2048,
        use_dual_gating=True,
    ),
    
    router=GSARouterConfig(
        num_router_heads=4,
        router_dim=64,
        top_k=3,
        use_adaptive_k=True,
    ),
)
```

---

## 🧩 Components

### GSA Router (`model/router.py`)

Multi-head router with sigmoid scoring inspired by the GSA paper.

```python
from moe_architecture.model import GSARouter

router = GSARouter(config)
indices, weights, aux_info = router(hidden_states)
# indices: [batch, seq, top_k] selected expert IDs
# weights: [batch, seq, top_k] gating weights
```

**Key innovations**:
1. **Multi-head routing**: Like attention indexer heads
2. **Sigmoid scoring**: Bounded scores, no forced competition
3. **Adaptive top-k**: Based on score variance
4. **Loss-free balancing**: Bias affects selection, not contribution

### Gated Expert (`model/expert.py`)

SwiGLU FFN with optional dual gating (G1+G2).

```python
from moe_architecture.model import GatedExpert

expert = GatedExpert(hidden_size=2048, intermediate_size=2048, config=expert_config)
output = expert(input)  # [batch, seq, hidden]
```

**Dual gating**:
- **G2 (Input Gate)**: Suppresses uninformative dimensions
- **G1 (Output Gate)**: Provides "do nothing" pathway

### Null Expert (`model/expert.py`)

Zero-compute pathway for junk token absorption.

```python
from moe_architecture.model import NullExpert

null_expert = NullExpert(hidden_size=2048)
output = null_expert(input)  # ≈ zeros
```

### Load Balancer (`model/load_balancer.py`)

Loss-free load balancing via bias adjustment.

```python
from moe_architecture.model import LoadBalancer

balancer = LoadBalancer(num_experts=8, config=balance_config)
balancer.accumulate(expert_indices)
metrics = balancer.update()
bias = balancer.get_bias()
```

---

## 🔗 Team Integration

### Team 6: Tokenizer Constraints

```python
from moe_architecture.config import Team6TokenizerConfig

tokenizer_config = Team6TokenizerConfig(
    vocab_size=32000,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    junk_token_ids=[0],  # Tokens to route to null
    special_token_range=(0, 100),
)
```

### Team 7: Telemetry Interface

```python
from moe_architecture.utils import MoETelemetrySystem, TelemetryConfig

telemetry = MoETelemetrySystem(
    num_experts=8,
    num_null_experts=1,
    junk_token_ids=[0],
)

# During training
telemetry.log_routing(expert_indices, gating_weights, token_ids)

# Check health
health = telemetry.check_health()
if not health['is_healthy']:
    print(health['alerts'])
```

**Null routing targets**:
- Junk tokens: 60-80% should route to null
- Signal tokens: <10% should route to null

---

## ⚡ CUDA Kernels

High-performance kernels using Triton (with PyTorch fallback).

```python
from moe_architecture.kernels import MoEKernels

kernels = MoEKernels(device='cuda')

# Sigmoid gating
gated = kernels.sigmoid_gating(scores, bias)

# Top-k selection
indices, weights = kernels.topk_gating(adjusted, original, k=2)

# Expert scatter/gather
expert_in, positions, counts = kernels.scatter(hidden, indices, num_experts)
output = kernels.gather(expert_out, positions, weights, indices, shape)

# Bias update
new_bias, metrics = kernels.update_bias(counts, bias, total_tokens)
```

---

## 📚 Training Guide

### Stage Transitions

#### 1B Dense → 3B MoE

```python
from moe_architecture.utils import expand_dense_to_moe

# Load dense checkpoint
dense_state = torch.load('1b_dense.pt')

# Expand to MoE
moe_state = expand_dense_to_moe(
    dense_state['model_state_dict'],
    num_experts=8,
    num_shared_experts=2,
    noise_std=1e-4,  # Symmetry breaking
)

# Create MoE model and load
moe_model = create_model(get_config('3b_moe'))
moe_model.load_state_dict(moe_state)
```

#### 8B MoE → 70B MoE (Expert Expansion)

```python
from moe_architecture.utils import expand_moe_experts

# Expand 8 experts to 64
expanded_state = expand_moe_experts(
    source_state,
    source_num_experts=8,
    target_num_experts=64,
    children_per_parent=8,
    noise_std=1e-3,  # Divergence noise
)
```

### Training Loop

```python
model = create_model(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    outputs = model(batch['input_ids'], labels=batch['labels'])
    loss = outputs['loss']
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Update load balancing biases
    metrics = model.post_training_step()
```

---

## 🔢 Mathematical Foundations

### Expert Count Formula

```
N_experts = 2^round(log2(param_ratio × 4))

Where param_ratio = Total_Params / Active_Params
```

### Top-K Selection (√N Rule)

```
K_optimal ≈ 0.5 × √N

N = 8  → K = 2
N = 64 → K = 4
```

### Shared Expert Formula

```
N_shared = α × K

Where α ≈ 0.5-1.0 for small MoE, 0.1-0.25 for large MoE
```

### Null Expert Formula

```
N_null = ⌈junk_rate × K / target_utilization⌉

With junk_rate ≈ 0.20, target_utilization ≈ 0.60
```

### GSA Router Score

```
score_i = Σⱼ σ(h W_j^weight) · σ(q_j · expert_key_i + b_j)

Bounded in (0, num_heads) for stability
```

---

## 📁 Project Structure

```
moe_architecture/
├── __init__.py           # Main package
├── __main__.py           # CLI entry point
├── config.py             # Configuration classes
├── configs/              # Pre-defined configurations
│   ├── config_1b_dense.py
│   ├── config_3b_moe.py
│   ├── config_8b_moe.py
│   └── config_70b_moe.py
├── model/                # Model components
│   ├── transformer.py    # Main model
│   ├── attention.py      # GQA attention
│   ├── router.py         # GSA router
│   ├── expert.py         # Expert modules
│   ├── moe_block.py      # MoE block
│   └── load_balancer.py  # Load balancing
├── kernels/              # CUDA kernels
│   └── moe_kernels.py    # Triton kernels
└── utils/                # Utilities
    ├── model_utils.py    # Expansion, checkpointing
    └── telemetry.py      # Team 7 integration
```

---

## 📖 References

1. **DeepSeek-V3**: MoE architecture and loss-free load balancing
2. **GSA Paper** (arXiv:2601.15305v1): Gated attention and adaptive sparsity
3. **Switch Transformer**: Sparse expert routing
4. **Mixtral**: Open-source MoE implementation
5. **Megablocks**: Efficient sparse MoE training

---

## 📜 License

MIT License - Team 8 MoE Architecture

---

## 👥 Team 8 Contributors

MoE Architecture Team - Expert Expansion & Routing
