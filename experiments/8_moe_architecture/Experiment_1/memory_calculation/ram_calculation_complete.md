# Complete RAM Requirements for MoE Training

## Team 8 - Memory Analysis

This document provides **accurate** RAM calculations including ALL memory components.

---

## Memory Components Overview

Training a model requires memory for **5 main components**:

```
Total_Memory = Model_Weights + Gradients + Optimizer_States + Activations + Temporary_Buffers

┌─────────────────────────────────────────────────────────────────┐
│                    GPU MEMORY BREAKDOWN                         │
├─────────────────────────────────────────────────────────────────┤
│  1. Model Weights      │  2 bytes/param (bf16)                  │
│  2. Gradients          │  2 bytes/param (bf16)                  │
│  3. Optimizer States   │  8 bytes/param (AdamW: m + v in fp32)  │
│  4. Activations        │  VARIABLE (biggest component!)         │
│  5. Temporary Buffers  │  ~10-20% overhead                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Static Memory (Fixed, Independent of Batch Size)

### Formula:
```
Static_Memory = N_params × (2 + 2 + 8) = N_params × 12 bytes

Where:
- Model weights: 2 bytes/param (bf16)
- Gradients: 2 bytes/param (bf16)  
- Optimizer (AdamW):
  - First moment (m): 4 bytes/param (fp32)
  - Second moment (v): 4 bytes/param (fp32)
```

### Static Memory by Model:

| Model | Total Params | Static Memory |
|-------|-------------|---------------|
| 1B Dense | 1.0B | 12 GB |
| 3B MoE-8 | 3.0B | 36 GB |
| 8B MoE-8 | 8.0B | 96 GB |
| 70B MoE-64 | 70B | 840 GB |

**⚠️ This is what I calculated before - but it's INCOMPLETE!**

---

## 2. Activation Memory (The Missing Piece!)

Activations are the intermediate tensors stored during forward pass, needed for backward pass gradient computation.

### What's Stored Per Layer:

```
┌─────────────────────────────────────────────────────────────────┐
│                 ACTIVATIONS PER TRANSFORMER LAYER               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ATTENTION BLOCK:                                               │
│  ├── Input to layer norm:     B × S × H                        │
│  ├── Query projection:        B × S × H                        │
│  ├── Key projection:          B × S × H_kv                     │
│  ├── Value projection:        B × S × H_kv                     │
│  ├── Attention scores:        B × num_heads × S × S  ← HUGE!   │
│  ├── Attention output:        B × S × H                        │
│  └── Output projection:       B × S × H                        │
│                                                                 │
│  FFN BLOCK (Dense):                                             │
│  ├── Input to layer norm:     B × S × H                        │
│  ├── Gate projection (W1):    B × S × intermediate             │
│  ├── Up projection (W3):      B × S × intermediate             │
│  ├── SwiGLU activation:       B × S × intermediate             │
│  └── Down projection (W2):    B × S × H                        │
│                                                                 │
│  FFN BLOCK (MoE):                                               │
│  ├── Router scores:           B × S × num_experts              │
│  ├── Expert indices:          B × S × top_k                    │
│  ├── Gating weights:          B × S × top_k                    │
│  ├── Shared expert output:    B × S × H                        │
│  └── Routed expert output:    B × S × H × (top_k activations)  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Where:
- B = batch_size
- S = sequence_length  
- H = hidden_size
- H_kv = hidden_size / GQA_ratio
```

### Activation Memory Formula:

**Per Layer (Dense Transformer):**
```python
def activation_per_layer_dense(B, S, H, intermediate, num_heads, num_kv_heads, dtype_bytes=2):
    """Calculate activation memory per dense layer."""
    
    # Attention
    attn_input = B * S * H                           # Input
    attn_qkv = B * S * (H + 2 * H // (num_heads // num_kv_heads))  # Q, K, V with GQA
    attn_scores = B * num_heads * S * S              # THE BIG ONE!
    attn_output = B * S * H                          # After attention
    
    attention_total = (attn_input + attn_qkv + attn_scores + attn_output)
    
    # FFN (SwiGLU has 3 projections)
    ffn_input = B * S * H
    ffn_intermediate = B * S * intermediate * 2       # W1 and W3 outputs
    ffn_activation = B * S * intermediate             # After SwiGLU
    ffn_output = B * S * H
    
    ffn_total = (ffn_input + ffn_intermediate + ffn_activation + ffn_output)
    
    # Layer norms (2 per layer)
    norms = 2 * B * S * H
    
    total_elements = attention_total + ffn_total + norms
    return total_elements * dtype_bytes
```

**Per Layer (MoE):**
```python
def activation_per_layer_moe(B, S, H, intermediate, num_heads, num_kv_heads,
                              num_routed, num_shared, top_k, dtype_bytes=2):
    """Calculate activation memory per MoE layer."""
    
    # Attention (same as dense)
    attn_input = B * S * H
    attn_qkv = B * S * (H + 2 * H // (num_heads // num_kv_heads))
    attn_scores = B * num_heads * S * S
    attn_output = B * S * H
    attention_total = attn_input + attn_qkv + attn_scores + attn_output
    
    # Router
    router_scores = B * S * (num_routed + 1)  # +1 for null
    router_indices = B * S * top_k
    router_weights = B * S * top_k
    router_total = router_scores + router_indices + router_weights
    
    # Shared experts (always computed)
    shared_intermediate = num_shared * B * S * intermediate * 2
    shared_activation = num_shared * B * S * intermediate
    shared_output = B * S * H
    shared_total = shared_intermediate + shared_activation + shared_output
    
    # Routed experts (only top_k computed)
    routed_intermediate = top_k * B * S * intermediate * 2
    routed_activation = top_k * B * S * intermediate
    routed_output = B * S * H
    routed_total = routed_intermediate + routed_activation + routed_output
    
    # Layer norms
    norms = 2 * B * S * H
    
    total_elements = attention_total + router_total + shared_total + routed_total + norms
    return total_elements * dtype_bytes
```

### The Killer: Attention Scores!

The attention score matrix `B × num_heads × S × S` is **quadratic in sequence length**:

```
Attention Scores Memory = B × H × S² × 2 bytes

Example (S=2048, B=8, H=16 heads):
= 8 × 16 × 2048 × 2048 × 2 bytes
= 8 × 16 × 4,194,304 × 2
= 1.07 GB per layer!

For 24 layers: 25.7 GB just for attention scores!
```

---

## 3. Complete Memory Calculation

### 1B Dense Model

```
Configuration:
- hidden_size = 2048
- num_layers = 24
- num_heads = 16, num_kv_heads = 4
- intermediate_size = 5504
- batch_size = 8
- seq_length = 2048

STATIC MEMORY:
- Model weights:     1.0B × 2 bytes = 2.0 GB
- Gradients:         1.0B × 2 bytes = 2.0 GB
- Optimizer states:  1.0B × 8 bytes = 8.0 GB
Static Total: 12.0 GB

ACTIVATION MEMORY (per layer):
- Attention input:        8 × 2048 × 2048 × 2 = 67 MB
- Attention Q:            8 × 2048 × 2048 × 2 = 67 MB
- Attention K (GQA):      8 × 2048 × 512 × 2 = 17 MB
- Attention V (GQA):      8 × 2048 × 512 × 2 = 17 MB
- Attention scores:       8 × 16 × 2048 × 2048 × 2 = 1,073 MB  ← HUGE!
- Attention output:       8 × 2048 × 2048 × 2 = 67 MB
- FFN intermediate (×2):  8 × 2048 × 5504 × 2 × 2 = 360 MB
- FFN activation:         8 × 2048 × 5504 × 2 = 180 MB
- FFN output:            8 × 2048 × 2048 × 2 = 67 MB
- Layer norms:           2 × 8 × 2048 × 2048 × 2 = 134 MB

Per layer total: ~2.0 GB
For 24 layers: 48 GB

TEMPORARY BUFFERS: ~10% = 6 GB

TOTAL WITHOUT CHECKPOINTING:
Static + Activations + Buffers = 12 + 48 + 6 = 66 GB

WITH ACTIVATION CHECKPOINTING (saves ~60-70%):
Static + Activations/3 + Buffers = 12 + 16 + 6 = 34 GB
```

### 3B MoE-8 Model

```
Configuration:
- hidden_size = 2048
- num_layers = 24
- num_heads = 16, num_kv_heads = 4
- intermediate_size = 5504
- num_routed_experts = 8, num_shared = 2, top_k = 2
- batch_size = 8
- seq_length = 2048

STATIC MEMORY:
- Model weights:     3.0B × 2 bytes = 6.0 GB
- Gradients:         3.0B × 2 bytes = 6.0 GB
- Optimizer states:  3.0B × 8 bytes = 24.0 GB
Static Total: 36.0 GB

ACTIVATION MEMORY (per MoE layer):
- Attention (same):  ~1.4 GB
- Router:            8 × 2048 × 9 × 2 = 0.3 MB (negligible)
- Shared experts:    2 × (8 × 2048 × 5504 × 2) × 2 = 720 MB
- Routed experts:    2 × (8 × 2048 × 5504 × 2) × 2 = 720 MB (only top_k=2)
- Other:             ~200 MB

Per MoE layer total: ~3.0 GB
For 24 MoE layers: 72 GB

TEMPORARY BUFFERS: ~10% = 11 GB

TOTAL WITHOUT CHECKPOINTING:
36 + 72 + 11 = 119 GB

WITH ACTIVATION CHECKPOINTING:
36 + 24 + 11 = 71 GB
```

### 8B MoE-8 Model

```
Configuration:
- hidden_size = 4096
- num_layers = 48
- num_heads = 32, num_kv_heads = 8
- intermediate_size = 11008
- num_routed_experts = 8, num_shared = 2, top_k = 2
- batch_size = 8
- seq_length = 2048

STATIC MEMORY:
- Model weights:     8.0B × 2 bytes = 16.0 GB
- Gradients:         8.0B × 2 bytes = 16.0 GB
- Optimizer states:  8.0B × 8 bytes = 64.0 GB
Static Total: 96.0 GB

ACTIVATION MEMORY (per layer):
- Attention scores:  8 × 32 × 2048 × 2048 × 2 = 2.1 GB  ← Doubled!
- Attention other:   ~0.5 GB
- MoE FFN:          ~2.5 GB (larger intermediate)
- Other:            ~0.4 GB

Per layer total: ~5.5 GB
For 48 layers: 264 GB

TEMPORARY BUFFERS: ~10% = 36 GB

TOTAL WITHOUT CHECKPOINTING:
96 + 264 + 36 = 396 GB

WITH ACTIVATION CHECKPOINTING:
96 + 88 + 36 = 220 GB
```

### 70B MoE-64 Model

```
Configuration:
- hidden_size = 4096
- num_layers = 80
- num_heads = 32, num_kv_heads = 8
- intermediate_size = 11008
- num_routed_experts = 64, num_shared = 4, top_k = 4
- batch_size = 8
- seq_length = 2048

STATIC MEMORY:
- Model weights:     70B × 2 bytes = 140 GB
- Gradients:         70B × 2 bytes = 140 GB
- Optimizer states:  70B × 8 bytes = 560 GB
Static Total: 840 GB

ACTIVATION MEMORY (per layer):
- Attention:         ~2.6 GB
- Router:            ~10 MB (64 experts)
- Shared experts:    4 × FFN_act = ~5 GB
- Routed experts:    4 × FFN_act = ~5 GB (top_k=4)
- Other:             ~0.4 GB

Per layer total: ~13 GB
For 80 layers: 1,040 GB

TEMPORARY BUFFERS: ~10% = 188 GB

TOTAL WITHOUT CHECKPOINTING:
840 + 1040 + 188 = 2,068 GB (2 TB!)

WITH ACTIVATION CHECKPOINTING:
840 + 347 + 188 = 1,375 GB (1.3 TB)
```

---

## 4. Summary Table: CORRECTED RAM Requirements

### Training Memory (Batch=8, Seq=2048)

| Model | Static | Activations | Buffers | **Total** | **With Checkpointing** |
|-------|--------|-------------|---------|-----------|------------------------|
| 1B Dense | 12 GB | 48 GB | 6 GB | **66 GB** | **~34 GB** |
| 3B MoE-8 | 36 GB | 72 GB | 11 GB | **119 GB** | **~71 GB** |
| 8B MoE-8 | 96 GB | 264 GB | 36 GB | **396 GB** | **~220 GB** |
| 70B MoE-64 | 840 GB | 1,040 GB | 188 GB | **2,068 GB** | **~1,375 GB** |

### Memory Breakdown Visualization

```
1B Dense (66 GB total):
├── Static (weights+optim): ████████████░░░░░░░░ 18%
├── Activations:            ████████████████████████████████████ 73%
└── Buffers:                ████░░░░░░░░░░░░░░░░ 9%

3B MoE (119 GB total):
├── Static (weights+optim): ██████████████████░░ 30%
├── Activations:            ████████████████████████████████████ 61%
└── Buffers:                ████░░░░░░░░░░░░░░░░ 9%

8B MoE (396 GB total):
├── Static (weights+optim): ████████████░░░░░░░░ 24%
├── Activations:            ████████████████████████████████████████ 67%
└── Buffers:                ████░░░░░░░░░░░░░░░░ 9%

70B MoE (2068 GB total):
├── Static (weights+optim): ██████████████████████ 41%
├── Activations:            ████████████████████████████████ 50%
└── Buffers:                ████░░░░░░░░░░░░░░░░ 9%
```

---

## 5. Memory Reduction Strategies

### Strategy 1: Activation Checkpointing (Gradient Checkpointing)
```
Trade compute for memory:
- Only store activations at checkpoint boundaries
- Recompute forward pass during backward
- Typically saves 60-70% activation memory
- Increases training time by ~30%

Implementation:
torch.utils.checkpoint.checkpoint(layer, input)
```

### Strategy 2: Reduce Batch Size
```
Activations scale linearly with batch size.

From batch=8 to batch=4:
- Activation memory halved
- But: 2× more steps needed, slower training
- Use gradient accumulation to maintain effective batch
```

### Strategy 3: Sequence Length Reduction
```
Attention scores scale QUADRATICALLY with sequence length!

From seq=2048 to seq=1024:
- Attention scores: 4× reduction
- Other activations: 2× reduction
- But: Limited context window
```

### Strategy 4: Flash Attention
```
Flash Attention avoids materializing full attention matrix:
- Computes attention in tiles
- O(S) memory instead of O(S²)
- 3-5× memory reduction for attention

For 8B model:
- Without Flash: 2.1 GB attention per layer
- With Flash: ~0.5 GB attention per layer
- Savings: 1.6 GB × 48 layers = 77 GB saved!
```

### Strategy 5: Mixed Precision Optimizer (8-bit Adam)
```
Use 8-bit quantized optimizer states:
- Standard Adam: 8 bytes/param
- 8-bit Adam: 2 bytes/param
- Savings: 6 bytes/param

For 70B model:
- Standard: 70B × 8 = 560 GB
- 8-bit: 70B × 2 = 140 GB
- Savings: 420 GB!
```

### Strategy 6: ZeRO Optimization (DeepSpeed)
```
ZeRO-1: Partition optimizer states across GPUs
ZeRO-2: + Partition gradients
ZeRO-3: + Partition parameters

For 70B on 8 GPUs with ZeRO-3:
- Each GPU holds: 70B/8 = 8.75B params
- Static per GPU: 8.75B × 12 = 105 GB
- vs Full replication: 840 GB per GPU
```

### Strategy 7: CPU Offloading
```
Offload inactive data to CPU RAM:
- Optimizer states → CPU (brought to GPU when needed)
- Inactive experts → CPU (for MoE)

70B MoE with expert offloading:
- Active experts (12B): 24 GB on GPU
- Inactive experts (58B): 116 GB on CPU
```

---

## 6. Practical GPU Requirements

### Minimum GPU Memory (with all optimizations)

| Model | Min GPU Memory | Recommended Setup |
|-------|---------------|-------------------|
| 1B Dense | **24 GB** | 1× A100-40GB |
| 3B MoE-8 | **40 GB** | 1× A100-80GB |
| 8B MoE-8 | **80 GB** | 2× A100-80GB (tensor parallel) |
| 70B MoE-64 | **320 GB** | 8× A100-80GB (ZeRO-3 + offload) |

### Recommended Setup for Training

| Model | GPUs | Memory Strategy |
|-------|------|-----------------|
| 1B Dense | 1-4× A100-40GB | Checkpointing + Flash Attention |
| 3B MoE-8 | 4-8× A100-40GB | Checkpointing + Flash + 8-bit Adam |
| 8B MoE-8 | 8× A100-80GB | ZeRO-2 + Checkpointing + Flash |
| 70B MoE-64 | 64× A100-80GB | ZeRO-3 + Expert Offload + Flash |

---

## 7. Quick Reference Formulas

```python
# Static memory (doesn't change with batch)
static_memory_gb = total_params * 12 / 1e9

# Activation memory per layer (approximate)
def activation_per_layer_gb(batch, seq, hidden, intermediate, num_heads):
    attention_scores = batch * num_heads * seq * seq * 2 / 1e9
    attention_other = batch * seq * hidden * 4 * 2 / 1e9
    ffn = batch * seq * intermediate * 4 * 2 / 1e9
    return attention_scores + attention_other + ffn

# Total with checkpointing
def total_memory_gb(static, activation_per_layer, num_layers, checkpoint=True):
    if checkpoint:
        activation_total = activation_per_layer * num_layers / 3
    else:
        activation_total = activation_per_layer * num_layers
    buffers = (static + activation_total) * 0.1
    return static + activation_total + buffers

# Flash Attention savings
flash_savings_gb = batch * num_heads * seq * seq * 2 * num_layers / 1e9 * 0.75
```

---

## 8. Why My Previous Calculation Was Wrong

| Component | My Previous | Correct | Error |
|-----------|------------|---------|-------|
| 1B Dense | 12 GB | 34-66 GB | **3-5×** underestimate |
| 3B MoE | 36 GB | 71-119 GB | **2-3×** underestimate |
| 8B MoE | 96 GB | 220-396 GB | **2-4×** underestimate |
| 70B MoE | 840 GB | 1375-2068 GB | **1.6-2.5×** underestimate |

**The key mistake was ignoring activation memory, especially the O(S²) attention scores!**

---

## Appendix: Memory Calculation Code

```python
def calculate_training_memory(
    total_params: float,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    intermediate_size: int,
    batch_size: int = 8,
    seq_length: int = 2048,
    use_checkpointing: bool = True,
    use_flash_attention: bool = True,
    dtype_bytes: int = 2,  # bf16
) -> dict:
    """Calculate complete training memory requirements."""
    
    # Static memory
    model_weights = total_params * dtype_bytes
    gradients = total_params * dtype_bytes
    optimizer_states = total_params * 8  # AdamW fp32
    static_total = model_weights + gradients + optimizer_states
    
    # Activation memory per layer
    B, S, H = batch_size, seq_length, hidden_size
    
    # Attention
    if use_flash_attention:
        attn_scores = B * S * H * dtype_bytes  # O(S) instead of O(S²)
    else:
        attn_scores = B * num_heads * S * S * dtype_bytes  # O(S²)
    
    attn_other = B * S * H * 4 * dtype_bytes  # Q, K, V, O
    
    # FFN
    ffn_memory = B * S * intermediate_size * 4 * dtype_bytes
    
    # Per layer total
    activation_per_layer = attn_scores + attn_other + ffn_memory
    
    # Total activations
    if use_checkpointing:
        activation_total = activation_per_layer * num_layers / 3
    else:
        activation_total = activation_per_layer * num_layers
    
    # Buffers
    buffers = (static_total + activation_total) * 0.1
    
    total = static_total + activation_total + buffers
    
    return {
        'static_gb': static_total / 1e9,
        'activations_gb': activation_total / 1e9,
        'buffers_gb': buffers / 1e9,
        'total_gb': total / 1e9,
    }
```
