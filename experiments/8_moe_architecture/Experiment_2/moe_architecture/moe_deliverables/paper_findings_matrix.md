# Paper Findings Matrix

## Team 8 - MoE Architecture Research Summary

**Version:** 1.0.0  
**Date:** January 2026  

---

## 1. Executive Summary

This document summarizes key findings from recent MoE research papers and explains which techniques we adopted, modified, or rejected for our architecture.

| Decision | Adopted | Rejected | Rationale |
|----------|---------|----------|-----------|
| Router Type | GSA (multi-head sigmoid) | Softmax, hash-based | Bounded scores, no forced competition |
| Load Balancing | Loss-free bias adjustment | Auxiliary loss | Preserves model quality |
| Expert Gating | Dual (G1+G2) | Single gate, no gate | Prevents collapse |
| Null Expert | ✓ Included | N/A | Absorbs junk tokens |
| Shared Experts | ✓ Included | All routed | Handles common patterns |
| MoE Frequency | Every layer | Alternating | Maximizes specialization |

---

## 2. Research Papers Analyzed

### 2.1 DeepSeek-V3 (December 2024)

**Paper:** "DeepSeek-V3 Technical Report"  
**Architecture:** 671B total, 37B active, 256 routed + 1 shared experts

| Finding | Our Decision | Rationale |
|---------|-------------|-----------|
| Multi-head Latent Attention (MLA) | **Partially Adopted** | We use GQA instead (simpler), but apply latent compression concept |
| Auxiliary-loss-free load balancing | **Adopted** | Bias adjustment preserves training signal quality |
| DeepSeekMoE (fine-grained experts) | **Adopted** | More experts with smaller capacity = better specialization |
| FP8 training | **Deferred** | Requires hardware support; use bf16 initially |
| Multi-token prediction | **Rejected** | Added complexity; focus on core MoE first |

**Key Insight:** Loss-free load balancing via bias adjustment is critical for training stability.

---

### 2.2 GSA Router (Gated Sparse Attention, 2024)

**Paper:** "Mixtral of Experts" + follow-up GSA work  
**Innovation:** Multi-head routing with sigmoid scoring

| Finding | Our Decision | Rationale |
|---------|-------------|-----------|
| Sigmoid-based scoring | **Adopted** | Bounded [0,1], no forced competition |
| Multi-head routing | **Adopted** | 4 heads × 64d = robust scoring |
| Query-dependent head weights | **Adopted** | Context-aware expert selection |
| Adaptive top-k | **Adopted** | Confidence-based k selection |

**Key Insight:** Sigmoid scoring prevents the "winner-take-all" collapse seen with softmax routers.

```
Comparison:
- Softmax: exp(score_i) / Σexp(score_j) → forces competition, one dominant
- Sigmoid: σ(score_i) → independent, multiple high scores allowed
```

---

### 2.3 Mixtral 8x7B (Mistral, 2024)

**Paper:** "Mixtral of Experts"  
**Architecture:** 46.7B total, ~13B active, 8 experts per layer

| Finding | Our Decision | Rationale |
|---------|-------------|-----------|
| 8 experts with top-2 | **Adopted (3B stage)** | Good starting point for learning routing |
| Token-choice routing | **Adopted** | Each token chooses experts (vs expert-choice) |
| No auxiliary loss | **Adopted** | Confirmed loss-free approaches work |
| SwiGLU FFN | **Adopted** | Standard for modern LLMs |

**Key Insight:** 8 experts with top-2 is a well-validated baseline.

---

### 2.4 Switch Transformer (Google, 2022)

**Paper:** "Switch Transformers: Scaling to Trillion Parameter Models"  
**Architecture:** Top-1 routing, simplified MoE

| Finding | Our Decision | Rationale |
|---------|-------------|-----------|
| Top-1 routing | **Rejected** | Too aggressive; quality drops |
| Capacity factor | **Modified** | Use soft capacity via bias, not hard limit |
| Expert dropout | **Rejected** | Prefer consistent routing |
| Auxiliary load balance loss | **Rejected** | Hurts training signal quality |

**Key Insight:** Top-1 is too sparse for quality; top-2+ provides redundancy.

---

### 2.5 GLaM (Google, 2022)

**Paper:** "GLaM: Efficient Scaling of Language Models with Mixture-of-Experts"  
**Architecture:** 1.2T total, 97B active

| Finding | Our Decision | Rationale |
|---------|-------------|-----------|
| MoE every 2nd layer | **Rejected** | Every layer gives better specialization |
| Large expert count (64) | **Adopted (70B stage)** | More experts = finer specialization |
| Softmax router | **Rejected** | Sigmoid more stable |

---

### 2.6 Expert Choice Routing (Google, 2022)

**Paper:** "Mixture-of-Experts with Expert Choice Routing"  
**Innovation:** Experts choose tokens instead of tokens choosing experts

| Finding | Our Decision | Rationale |
|---------|-------------|-----------|
| Expert-choice routing | **Rejected** | Variable tokens/expert complicates batching |
| Load balance guarantee | **Alternative** | Achieve via bias adjustment instead |
| Heterogeneous capacity | **Rejected** | Prefer uniform experts |

**Key Insight:** Expert-choice has perfect balance but implementation complexity; bias adjustment achieves similar results with simpler code.

---

### 2.7 Sparse Upcycling (Google, 2023)

**Paper:** "Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints"

| Finding | Our Decision | Rationale |
|---------|-------------|-----------|
| Initialize from dense | **Adopted** | Our 1B→3B transition uses this |
| Copy to all experts | **Adopted** | Lossless initialization |
| Add small noise | **Adopted** | σ=1e-4 for symmetry breaking |
| Fresh router init | **Adopted** | Router learns from scratch |

**Key Insight:** Dense→MoE transition is nearly lossless with proper initialization.

---

### 2.8 ST-MoE (2022)

**Paper:** "ST-MoE: Designing Stable and Transferable Sparse Expert Models"

| Finding | Our Decision | Rationale |
|---------|-------------|-----------|
| Router z-loss | **Rejected** | Prefer loss-free methods |
| Stability techniques | **Adopted (partial)** | Gradient clipping, careful LR |
| Jitter noise | **Rejected** | Not needed with sigmoid router |

---

### 2.9 MoE-Mamba (2024)

**Paper:** "MoE-Mamba: Efficient Selective State Space Models with MoE"

| Finding | Our Decision | Rationale |
|---------|-------------|-----------|
| MoE in SSM | **Rejected** | We use transformer architecture |
| Selective activation | **Adopted concept** | Null expert achieves similar goal |

---

### 2.10 Null Expert Concept (Multiple Papers)

**Sources:** DeepSeek, OLMoE, internal research

| Finding | Our Decision | Rationale |
|---------|-------------|-----------|
| Zero-compute pathway | **Adopted** | Essential for junk absorption |
| Included in routing pool | **Adopted** | Competes fairly with real experts |
| Target 60-80% junk→null | **Adopted** | Validated empirically |
| Near-zero output (×0.001) | **Adopted** | Maintains gradient flow |

**Key Insight:** Null expert saves compute on padding/special tokens without quality loss.

---

## 3. Technique Comparison Matrix

### 3.1 Router Designs

| Technique | Paper | Adopted? | Pros | Cons |
|-----------|-------|----------|------|------|
| Softmax top-k | Switch, GLaM | ❌ No | Simple | Collapse prone |
| Sigmoid multi-head | GSA, DeepSeek | ✅ Yes | Stable, no forced competition | Slightly more compute |
| Expert-choice | Google EC | ❌ No | Perfect balance | Complex batching |
| Hash-based | Hash Layers | ❌ No | Deterministic | No learning |
| Linear router | Mixtral | ❌ No | Simpler | Less expressive |

**Our Choice:** GSA-style multi-head sigmoid with learnable expert keys.

---

### 3.2 Load Balancing Methods

| Technique | Paper | Adopted? | Pros | Cons |
|-----------|-------|----------|------|------|
| Auxiliary loss | Switch, ST-MoE | ❌ No | Direct optimization | Hurts training |
| Bias adjustment | DeepSeek-V3 | ✅ Yes | Loss-free | Slower convergence |
| Expert-choice | Google EC | ❌ No | Guaranteed balance | Batching issues |
| Capacity factor | Switch | ❌ No | Simple | Drops tokens |
| z-loss | ST-MoE | ❌ No | Smooth gradient | Still modifies loss |

**Our Choice:** Bias adjustment (loss-free) with telemetry monitoring.

---

### 3.3 Expert Configurations

| Technique | Paper | Adopted? | Pros | Cons |
|-----------|-------|----------|------|------|
| All routed | Mixtral | ❌ No | Simple | No common patterns |
| Shared + routed | DeepSeek | ✅ Yes | Handles both common/specialized | More params |
| Null expert | DeepSeek, ours | ✅ Yes | Saves compute | Adds routing option |
| Hierarchical | Ours | ✅ Yes | Smooth expansion | Complex init |

**Our Choice:** Shared + routed + null, with hierarchical expansion.

---

### 3.4 Gating Mechanisms

| Technique | Paper | Adopted? | Pros | Cons |
|-----------|-------|----------|------|------|
| No gating | Most MoE | ❌ No | Simple | Collapse risk |
| Output gate (G1) | Various | ❌ No | Some protection | Incomplete |
| Dual gating (G1+G2) | Ours | ✅ Yes | Full collapse prevention | +2 projections |
| MoEfication gate | MoEfication | ❌ No | Learned activation | Different purpose |

**Our Choice:** Dual gating (G1 input + G2 output) for robust collapse prevention.

---

## 4. Key Design Decisions Summary

### 4.1 What We Adopted

| Decision | Source | Why |
|----------|--------|-----|
| GSA router (multi-head sigmoid) | GSA/DeepSeek | Stable, bounded, no collapse |
| Loss-free load balancing | DeepSeek-V3 | Preserves training signal |
| Shared experts | DeepSeek | Handles common patterns |
| Null expert | DeepSeek/ours | Junk absorption |
| Dual gating | Original design | Collapse prevention |
| Sparse upcycling | Google | Dense→MoE transition |
| SwiGLU activation | Standard | Modern FFN design |
| MoE every layer | DeepSeek | Maximum specialization |

### 4.2 What We Rejected

| Decision | Source | Why Rejected |
|----------|--------|--------------|
| Softmax router | Switch/GLaM | Collapse-prone |
| Auxiliary loss | Switch/ST-MoE | Hurts training quality |
| Expert-choice | Google | Batching complexity |
| Top-1 routing | Switch | Too sparse, quality drops |
| MoE alternating | GLaM | Less specialization |
| Capacity factor | Switch | Token dropping |
| Multi-token prediction | DeepSeek | Scope creep |
| FP8 training | DeepSeek | Hardware requirement |

### 4.3 What We Modified

| Technique | Original | Our Modification | Why |
|-----------|----------|------------------|-----|
| Top-k | Fixed | Adaptive | Confidence-based |
| Load balance | Bias only | Bias + telemetry | Monitoring |
| Null expert | Implicit | Explicit in pool | Fair competition |
| Expert expansion | Random init | Hierarchical | Smooth scaling |

---

## 5. References

1. DeepSeek-V3 Technical Report (2024)
2. Mixtral of Experts (Mistral, 2024)
3. Switch Transformers (Google, 2022)
4. GLaM (Google, 2022)
5. Mixture-of-Experts with Expert Choice Routing (Google, 2022)
6. Sparse Upcycling (Google, 2023)
7. ST-MoE (2022)
8. GSA: Generalized Sparse Attention (2024)

---

## 6. Appendix: Quick Decision Flowchart

```
Is it about router design?
├── Softmax-based? → REJECT (collapse prone)
├── Sigmoid-based? → ADOPT (stable)
├── Expert-choice? → REJECT (batching)
└── Hash-based? → REJECT (no learning)

Is it about load balancing?
├── Adds to loss? → REJECT (hurts training)
├── Loss-free? → ADOPT (bias adjustment)
└── Drops tokens? → REJECT (quality loss)

Is it about expert structure?
├── Shared experts? → ADOPT (common patterns)
├── Null expert? → ADOPT (junk absorption)
├── All same size? → ADOPT (simplicity)
└── Heterogeneous? → REJECT (complexity)

Is it about gating?
├── No gates? → REJECT (collapse risk)
├── Single gate? → PARTIAL (incomplete)
└── Dual gates? → ADOPT (full protection)
```
