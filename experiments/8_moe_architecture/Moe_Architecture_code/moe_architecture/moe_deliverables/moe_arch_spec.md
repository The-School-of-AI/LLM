# MoE Architecture Specification

## Team 8 - Expert Expansion & Routing

**Version:** 1.0.0  
**Date:** January 2026  
**Authors:** Team 8 - MoE Architecture  

---

## 1. Executive Summary

This document specifies the Mixture of Experts (MoE) architecture for the complete growth cadence:
- **Stage 1:** 1B Dense (Foundation)
- **Stage 2:** 3B MoE-8 (Learn Routing)
- **Stage 3:** 8B MoE-8 (Scale Dimensions)
- **Stage 4:** 70B MoE-64 (Expert Expansion)

Key innovations:
- GSA-inspired multi-head sigmoid router
- Dual gating (G1+G2) for collapse prevention
- Null expert for junk token absorption
- Loss-free load balancing via bias adjustment
- Hierarchical expert expansion (8→64)

---

## 2. MoE Block Definition

### 2.1 Block Placement

**MoE replaces FFN only** - Attention layers remain dense.

```
┌─────────────────────────────────────────────────────────┐
│                   TRANSFORMER LAYER                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   Input ─────► LayerNorm ─────► Attention ─────┐       │
│                                                 │       │
│                                    Residual ◄───┘       │
│                                        │                │
│                                        ▼                │
│                                   LayerNorm             │
│                                        │                │
│                    ┌───────────────────┴───────────────┐│
│                    │           MoE BLOCK               ││
│                    │  ┌─────────────────────────────┐  ││
│                    │  │      GSA Router             │  ││
│                    │  │  (Multi-head Sigmoid)       │  ││
│                    │  └──────────┬──────────────────┘  ││
│                    │             │                      ││
│                    │    ┌────────┼────────┐            ││
│                    │    ▼        ▼        ▼            ││
│                    │ Shared   Routed    Null           ││
│                    │ Experts  Experts   Expert         ││
│                    │    │        │        │            ││
│                    │    └────────┼────────┘            ││
│                    │             ▼                      ││
│                    │      Weighted Combine             ││
│                    └─────────────┬─────────────────────┘│
│                                  │                      │
│                       Residual ◄─┘                      │
│                           │                             │
│                           ▼                             │
│                        Output                           │
└─────────────────────────────────────────────────────────┘
```

### 2.2 MoE Layer Frequency

| Stage | Model | MoE Layers | Dense Layers | MoE Frequency |
|-------|-------|------------|--------------|---------------|
| 1 | 1B Dense | 0 | 24 | N/A (all dense) |
| 2 | 3B MoE-8 | 24 | 0 | Every layer |
| 3 | 8B MoE-8 | 48 | 0 | Every layer |
| 4 | 70B MoE-64 | 80 | 0 | Every layer |

**Rationale:** MoE on every layer maximizes parameter efficiency and specialization potential.

### 2.3 Expert FFN Architecture

Each expert uses **SwiGLU activation** with optional **dual gating**:

```python
class GatedExpert(nn.Module):
    """
    SwiGLU FFN with dual gating for collapse prevention.
    
    Architecture:
        G2 (input gate):  x' = x ⊙ σ(W_g_in @ x)
        SwiGLU:           h = (W1 @ x') ⊙ SiLU(W3 @ x')
        G1 (output gate): out = (W2 @ h) ⊙ σ(W_g_out @ x)
    
    Dual gating provides:
        - G2: Suppress uninformative input dimensions
        - G1: "Do nothing" pathway for misrouted tokens
    """
    
    def forward(self, x):
        # Input gate (G2)
        if self.use_dual_gating:
            gate_in = torch.sigmoid(self.gate_in(x))
            x = x * gate_in
        
        # SwiGLU FFN
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        h = gate * up
        out = self.w2(h)
        
        # Output gate (G1)
        if self.use_dual_gating:
            gate_out = torch.sigmoid(self.gate_out(x))
            out = out * gate_out
        
        return out
```

---

## 3. Expert Composition Per Stage

### 3.1 Stage 2: 3B MoE-8

```
┌─────────────────────────────────────────────────────────┐
│                    3B MoE-8 EXPERT POOL                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ROUTED EXPERTS (8)         SHARED EXPERTS (2)        │
│   ┌───┬───┬───┬───┐         ┌───┬───┐                  │
│   │ E0│ E1│ E2│ E3│         │ S0│ S1│ ◄── Always active│
│   ├───┼───┼───┼───┤         └───┴───┘                  │
│   │ E4│ E5│ E6│ E7│                                    │
│   └───┴───┴───┴───┘                                    │
│         ▲                                               │
│         │ Top-K=2 selection                            │
│                                                         │
│   NULL EXPERT (1)                                       │
│   ┌───┐                                                │
│   │ N0│ ◄── Zero-compute, absorbs junk                 │
│   └───┘                                                │
│                                                         │
│   ROUTING POOL: 8 routed + 1 null = 9 total            │
│   ACTIVE PER TOKEN: 2 shared + 2 routed = 4 experts    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**3B Configuration:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Routed Experts | 8 | Minimum for meaningful specialization |
| Shared Experts | 2 | α × K where α=1.0, K=2 |
| Null Experts | 1 | ⌈junk_rate × K / target_util⌉ = ⌈0.2×2/0.6⌉ |
| Top-K | 2 | √8 × 0.5 ≈ 1.4 → 2 |
| Total in Pool | 9 | 8 routed + 1 null |
| Active per Token | 4 | 2 shared + 2 routed |

### 3.2 Stage 4: 70B MoE-64

```
┌─────────────────────────────────────────────────────────┐
│                   70B MoE-64 EXPERT POOL                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ROUTED EXPERTS (64) - Hierarchical from 8 parents    │
│   ┌───┬───┬───┬───┬───┬───┬───┬───┐                    │
│   │P0 │P0 │P0 │P0 │P0 │P0 │P0 │P0 │ ◄── Parent 0's 8  │
│   │C0 │C1 │C2 │C3 │C4 │C5 │C6 │C7 │     children       │
│   ├───┼───┼───┼───┼───┼───┼───┼───┤                    │
│   │P1 │P1 │P1 │P1 │P1 │P1 │P1 │P1 │ ◄── Parent 1's 8  │
│   │C0 │C1 │C2 │C3 │C4 │C5 │C6 │C7 │     children       │
│   ├───┴───┴───┴───┴───┴───┴───┴───┤                    │
│   │         ... (8 parents × 8 children = 64)          │
│   └─────────────────────────────────────────────────────┤
│                                                         │
│   SHARED EXPERTS (4)         NULL EXPERTS (2)          │
│   ┌───┬───┬───┬───┐         ┌───┬───┐                  │
│   │ S0│ S1│ S2│ S3│         │ N0│ N1│                  │
│   └───┴───┴───┴───┘         └───┴───┘                  │
│                                                         │
│   ROUTING POOL: 64 routed + 2 null = 66 total          │
│   ACTIVE PER TOKEN: 4 shared + 4 routed = 8 experts    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**70B Configuration:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Routed Experts | 64 | 8 parents × 8 children |
| Shared Experts | 4 | 2× for larger pool (α=0.5, K=4) |
| Null Experts | 2 | ⌈0.2×4/0.6⌉ for more capacity |
| Top-K | 4 | √64 × 0.5 = 4 |
| Total in Pool | 66 | 64 routed + 2 null |
| Active per Token | 8 | 4 shared + 4 routed |

---

## 4. Top-K Selection and Null-in-Top-K Behavior

### 4.1 GSA Router Scoring

The router computes affinity scores using multi-head sigmoid:

```
Affinity_i = Σⱼ σ(h·W_j^weight) · σ(q_j · expert_key_i + bias_i)

Where:
- h = hidden state
- W_j^weight = head weight projection
- q_j = query for head j
- expert_key_i = learnable centroid for expert i
- bias_i = adjustable expert bias (for load balancing)
```

### 4.2 Top-K Selection Process

```python
def select_experts(scores, k, null_indices):
    """
    Select top-k experts with null handling.
    
    Key behaviors:
    1. Null experts participate in top-k selection
    2. If null selected, token gets zero-compute path
    3. Gating weights normalized over selected experts
    """
    # Get top-k indices and scores
    top_scores, top_indices = torch.topk(scores, k, dim=-1)
    
    # Normalize gating weights (sigmoid scores, not softmax)
    # Bounded in (0, k) for stability
    gating_weights = top_scores / top_scores.sum(dim=-1, keepdim=True)
    
    return top_indices, gating_weights
```

### 4.3 Null-in-Top-K Behavior

**When null expert is selected in top-k:**

```
┌─────────────────────────────────────────────────────────┐
│              NULL EXPERT SELECTION FLOW                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   Token x ──► Router ──► Top-K Selection                │
│                              │                          │
│              ┌───────────────┼───────────────┐          │
│              ▼               ▼               ▼          │
│         Expert E3       Expert E7       NULL N0         │
│         (weight=0.4)    (weight=0.35)   (weight=0.25)   │
│              │               │               │          │
│              ▼               ▼               ▼          │
│         FFN(x)          FFN(x)          x × 0.001       │
│              │               │               │          │
│              └───────────────┼───────────────┘          │
│                              ▼                          │
│                    Weighted Combination                 │
│                              │                          │
│           out = 0.4×E3(x) + 0.35×E7(x) + 0.25×(x×0.001)│
│                              │                          │
│                              ▼                          │
│                     Final Output                        │
│                                                         │
│   Result: Null contributes near-zero, but:             │
│   - Occupies a top-k slot (reduces active compute)     │
│   - Maintains gradient flow (tiny scale)               │
│   - Signals "this token is unimportant"                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Null expert targets:**
| Token Type | Target Null Rate | Acceptable Range |
|------------|------------------|------------------|
| Junk (PAD, special) | 60-80% | Must exceed 50% |
| Signal (content) | <10% | Alert if >15% |

### 4.4 Adaptive Top-K

Based on score variance, dynamically adjust k:

```python
def adaptive_top_k(scores, base_k=2, min_k=1, max_k=4):
    """
    Adjust k based on router confidence.
    
    High variance = confident routing = fewer experts
    Low variance = uncertain = more experts
    """
    variance = scores.var(dim=-1)
    normalized_var = variance / (variance.mean() + 1e-6)
    
    # Scale k inversely with confidence
    k = base_k / torch.clamp(normalized_var, 0.5, 2.0)
    k = torch.clamp(k.round(), min_k, max_k).int()
    
    return k
```

---

## 5. Expert Explosion Plan for 70B

### 5.1 Growth Cadence Overview

```
Stage 1          Stage 2          Stage 3          Stage 4
1B Dense    ──►  3B MoE-8    ──►  8B MoE-8    ──►  70B MoE-64
                                                        
   │               │               │               │
   ▼               ▼               ▼               ▼
Foundation    Expert          Dimension       Expert
Training      Explosion       Scaling         Expansion
              (1→8)          (same 8)         (8→64)
```

### 5.2 Stage 1→2: Expert Explosion (1→8)

**Process:** Copy dense FFN to all 8 routed experts with noise.

```python
def explode_dense_to_moe(dense_model, moe_model, noise_std=1e-4):
    """
    Initialize MoE from dense model.
    
    All experts start identical (lossless at initialization).
    Noise enables symmetry breaking during training.
    """
    dense_ffn = dense_model.ffn
    
    for layer_idx in range(num_layers):
        # Copy to all routed experts
        for expert_idx in range(8):
            expert = moe_model.layers[layer_idx].experts.routed[expert_idx]
            
            expert.w1.weight.data = dense_ffn.w1.weight.data.clone()
            expert.w2.weight.data = dense_ffn.w2.weight.data.clone()
            expert.w3.weight.data = dense_ffn.w3.weight.data.clone()
            
            # Add small noise for symmetry breaking
            expert.w1.weight.data += torch.randn_like(expert.w1.weight) * noise_std
            expert.w2.weight.data += torch.randn_like(expert.w2.weight) * noise_std
            expert.w3.weight.data += torch.randn_like(expert.w3.weight) * noise_std
        
        # Copy to shared experts (no noise - they should stay general)
        for shared_idx in range(2):
            shared = moe_model.layers[layer_idx].experts.shared[shared_idx]
            shared.w1.weight.data = dense_ffn.w1.weight.data.clone()
            shared.w2.weight.data = dense_ffn.w2.weight.data.clone()
            shared.w3.weight.data = dense_ffn.w3.weight.data.clone()
        
        # Initialize fresh router
        moe_model.layers[layer_idx].router.reset_parameters()
```

### 5.3 Stage 2→3: Dimension Scaling (Same 8 Experts)

**Process:** Interpolate weights for larger dimensions, keep routing intact.

```python
def scale_dimensions(source_model, target_model):
    """
    Scale model dimensions while preserving routing.
    
    Changes:
    - hidden: 2048 → 4096
    - intermediate: 5504 → 11008
    - layers: 24 → 48
    - Experts: STILL 8 (routing preserved!)
    """
    # Weight interpolation for larger dimensions
    for layer_idx in range(target_model.num_layers):
        src_layer = layer_idx * source_model.num_layers // target_model.num_layers
        
        for expert_idx in range(8):
            src_expert = source_model.layers[src_layer].experts.routed[expert_idx]
            tgt_expert = target_model.layers[layer_idx].experts.routed[expert_idx]
            
            # Interpolate weights using bilinear interpolation
            tgt_expert.w1.weight.data = interpolate_weight(
                src_expert.w1.weight.data,
                tgt_expert.w1.weight.shape
            )
            # ... similar for w2, w3
        
        # Scale router input projection
        target_model.layers[layer_idx].router.scale_input_dim(
            source_dim=2048, target_dim=4096
        )
```

### 5.4 Stage 3→4: Expert Expansion (8→64)

**Process:** Each of 8 parent experts spawns 8 children.

```python
def expand_experts(source_model, target_model, children_per_parent=8, noise_std=1e-3):
    """
    Hierarchical expert expansion: 8 parents → 64 children.
    
    Each parent expert is copied to 8 children with noise.
    Router keys initialized hierarchically:
    - Children of same parent start with similar keys
    - Enables gradual specialization
    """
    for layer_idx in range(target_model.num_layers):
        src_layer_idx = layer_idx * source_model.num_layers // target_model.num_layers
        
        for parent_idx in range(8):
            parent_expert = source_model.layers[src_layer_idx].experts.routed[parent_idx]
            parent_router_key = source_model.layers[src_layer_idx].router.expert_keys[parent_idx]
            
            for child_idx in range(children_per_parent):
                global_child_idx = parent_idx * children_per_parent + child_idx
                child_expert = target_model.layers[layer_idx].experts.routed[global_child_idx]
                
                # Copy parent weights with noise
                child_expert.w1.weight.data = parent_expert.w1.weight.data.clone()
                child_expert.w2.weight.data = parent_expert.w2.weight.data.clone()
                child_expert.w3.weight.data = parent_expert.w3.weight.data.clone()
                
                # Add noise for divergence (slightly larger than explosion)
                child_expert.w1.weight.data += torch.randn_like(child_expert.w1.weight) * noise_std
                child_expert.w2.weight.data += torch.randn_like(child_expert.w2.weight) * noise_std
                child_expert.w3.weight.data += torch.randn_like(child_expert.w3.weight) * noise_std
                
                # Hierarchical router key initialization
                # Children of same parent have correlated keys
                base_key = parent_router_key.clone()
                perturbation = torch.randn_like(base_key) * 0.1
                target_model.layers[layer_idx].router.expert_keys.data[global_child_idx] = base_key + perturbation
        
        # Expand shared experts: 2 → 4
        for i in range(2):
            src_shared = source_model.layers[src_layer_idx].experts.shared[i]
            for j in range(2):
                tgt_idx = i * 2 + j
                tgt_shared = target_model.layers[layer_idx].experts.shared[tgt_idx]
                tgt_shared.load_state_dict(src_shared.state_dict())
        
        # Expand null experts: 1 → 2
        # Just initialize fresh (null experts have no learned weights)
```

### 5.5 Expansion Validation Checkpoints

| Checkpoint | Validation | Pass Criteria |
|------------|------------|---------------|
| Post-Explosion | Loss matches dense baseline | Δloss < 0.1% |
| Post-Explosion | Router entropy > 0.7 | Entropy normalized |
| Post-Scaling | Routing patterns preserved | Correlation > 0.9 |
| Post-Expansion | No dead experts | All experts > 1% util |
| Post-Expansion | Children diverge from parents | Cosine sim < 0.95 after 1K steps |

---

## 6. Architecture Diagrams

### 6.1 Complete MoE Block

```
                              Input: [B, S, H]
                                     │
                                     ▼
                            ┌─────────────────┐
                            │   Layer Norm    │
                            └────────┬────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
        │                   ┌────────▼────────┐                   │
        │                   │   GSA Router    │                   │
        │                   │                 │                   │
        │                   │ Multi-head σ    │                   │
        │                   │ Query-dep wts   │                   │
        │                   │ Adaptive top-k  │                   │
        │                   └────────┬────────┘                   │
        │                            │                            │
        │              ┌─────────────┼─────────────┐              │
        │              │             │             │              │
        │              ▼             ▼             ▼              │
        │     ┌──────────────┐ ┌──────────┐ ┌──────────┐         │
        │     │   Shared     │ │  Routed  │ │   Null   │         │
        │     │   Experts    │ │  Experts │ │  Expert  │         │
        │     │  (always)    │ │ (top-k)  │ │ (if sel) │         │
        │     │              │ │          │ │          │         │
        │     │  SwiGLU+G1G2 │ │ SwiGLU   │ │ x×0.001  │         │
        │     └──────┬───────┘ └────┬─────┘ └────┬─────┘         │
        │            │              │            │                │
        │            └──────────────┼────────────┘                │
        │                           ▼                             │
        │                  ┌─────────────────┐                    │
        │                  │ Weighted Combine │                    │
        │                  │  Σ wᵢ × expertᵢ  │                    │
        │                  └────────┬────────┘                    │
        │                           │                             │
        └───────────────► (+) ◄─────┘                             
                          │ Residual
                          ▼
                   Output: [B, S, H]
```

### 6.2 Growth Cadence Visualization

```
          STAGE 1              STAGE 2              STAGE 3              STAGE 4
        ┌─────────┐          ┌─────────┐          ┌─────────┐          ┌─────────┐
        │         │          │ ■ ■ ■ ■ │          │ ■ ■ ■ ■ │          │■■■■■■■■│
        │    ■    │   ──►    │ ■ ■ ■ ■ │   ──►    │ ■ ■ ■ ■ │   ──►    │■■■■■■■■│
        │         │          │   + ○ ● │          │   + ○ ● │          │■■■■■■■■│
        └─────────┘          └─────────┘          └─────────┘          │■■■■■■■■│
         1B Dense             3B MoE-8             8B MoE-8            │■■■■■■■■│
                                                                       │■■■■■■■■│
        ■ = FFN/Expert       Expert Pool:          Same 8 experts      │■■■■■■■■│
        ○ = Shared           8R + 2S + 1N          Bigger dims         │■■■■■■■■│
        ● = Null                                                       │ +○○○○●●│
                                                                       └─────────┘
                                                                        70B MoE-64
                                                                       64R + 4S + 2N
```

---

## 7. Integration Points

### 7.1 Team 6 (Tokenizer) Integration

```python
# Token classification for null routing
tokenizer_config = {
    'junk_token_ids': [0, 1, 2, 3],  # PAD, BOS, EOS, UNK
    'special_token_range': (0, 10),
    'punctuation_range': (10, 100),
    
    # Classification method
    'is_junk': lambda token_id: token_id in junk_token_ids or 
                                token_id in range(*special_token_range)
}
```

### 7.2 Team 7 (Telemetry) Integration

```python
# Telemetry hooks
telemetry_config = {
    'log_interval': 100,
    'metrics': [
        'expert_utilization',      # Per-expert token counts
        'null_routing_rate',       # % tokens → null
        'junk_null_rate',          # % junk → null (target: 60-80%)
        'signal_null_rate',        # % signal → null (target: <10%)
        'router_entropy',          # Selection diversity
        'gini_coefficient',        # Load balance measure
        'dead_expert_count',       # Experts with <1% utilization
    ],
    'alerts': {
        'junk_null_low': 'junk_null_rate < 0.5',
        'signal_null_high': 'signal_null_rate > 0.15',
        'router_collapse': 'router_entropy < 0.5',
        'dead_experts': 'dead_expert_count > 0',
    }
}
```

---

## 8. Appendix: Quick Reference

### 8.1 Configuration Summary

| Parameter | 3B MoE-8 | 70B MoE-64 |
|-----------|----------|------------|
| Hidden Size | 2048 | 4096 |
| Num Layers | 24 | 80 |
| Intermediate | 5504 | 11008 |
| Routed Experts | 8 | 64 |
| Shared Experts | 2 | 4 |
| Null Experts | 1 | 2 |
| Top-K | 2 | 4 |
| Active Experts | 4 | 8 |
| Total Params | ~3B | ~70B |
| Active Params | ~1.2B | ~12B |
| Active Ratio | 40% | 17% |

### 8.2 Key Formulas

```
Top-K:           K = 0.5 × √N_routed
Shared Experts:  N_shared = α × K (α ∈ [0.5, 1.0])
Null Experts:    N_null = ⌈junk_rate × K / target_util⌉
Active Ratio:    (N_shared + K) / (N_routed + N_shared)
Router Score:    Σⱼ σ(wⱼ) · σ(qⱼ · key + bias)
```
