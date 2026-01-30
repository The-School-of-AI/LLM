# Paper Findings Matrix — MoE + Null Experts

**Purpose**

This document consolidates our findings from the DeepSeekMoE / DeepSeek-V3 literature, Mixtral of experts, Meta Scalable MoE and the Null Experts (data-sparsity) paper, and translates them into final, paper-faithful MoE configuration proposals for our training cadence. This file is intended to be submitted to close the issue and act as the single-source-of-truth for architecture decisions.

---

## Executive summary

- We keep **DeepSeek invariants**: fixed routing segments (m = 4), fixed per-segment activation pattern (K per segment), and *constant active experts per token* (routed ≈ 8, shared ≈ 1–2 → total ≈ 9–10). Scaling is performed by increasing **real expert pool (N)**, not by increasing active K.
- We integrate **Null Experts** (zero-compute routing copies) from the *null experts* paper to compose weight and data sparsity while **preserving DeepSeek invariants**. We follow the expected top-ℓ formulation: `E[K_real] = k_max * rho` and null copy count `M = N * (1 - rho) / rho`.
- To *preserve DeepSeek final active experts* we **raise k_max** when using null experts. We standardize on `rho = 0.5` (stable eval region in the paper) and therefore set `k_max = 16`, which yields `E[K_real] = 8` (same as DeepSeek).

---

## Core principles (ground truth)

1. **DeepSeek invariants:**
   - Segments: `m = 4`
   - Routed experts activated per token (real): `E[K_real] ≈ 8`
   - Shared experts per layer: `1–2`, decays with scale
   - Expert FFN dimension: **fixed** (fine-grained experts)

2. **Null-expert relations (from null-experts paper):**
   - Expected real experts: `E[K_real] = k_max * rho`.
   - Null copies: `M = N * (1 - rho) / rho` (where N is number of real routed experts)
   - Stable eval range: `rho ≈ 0.5–0.67`; we adopt `rho = 0.5` as default.

3. **Design rule that must hold:**
   - Keep `E[K_real]` equal to DeepSeek value (8) to preserve FLOPs/token ceiling and downstream behavior.
   - Therefore, with `rho = 0.5`, `k_max = 8 / 0.5 = 16`.

---

## Short definitions / notation

- `m` — number of routing segments (DeepSeek uses 4)
- `N` — total real routed experts (across all segments)
- `N_seg` — experts per segment (so `N = m * N_seg`)
- `Ks` — shared experts per MoE layer (always active)
- `k_max` — top-K the router selects before null-thresholding (total slots)
- `rho` — data sparsity (fraction of selected slots that are *real* experts in expectation)
- `M` — number of null copies inserted into the routing pool
- `E[K_real]` — expected number of real experts activated per token = `k_max * rho`

---

## DeepSeek-faithful MoE baseline (no nulls)

These are the invariants observed in DeepSeek / DeepSeek-V3 experiments and used as our baseline.

| Property | Value |
|---|---:|
| Segments (m) | 4 |
| Routed experts activated (real) | 8 |
| Shared experts (small→large) | 2 → 1 |
| k_max (no nulls) | 8 |
| Expert FFN dim | fixed (fine-grained, e.g., 2048 in V3 example) |

---

## Null-expert mechanism (key formulas)

- Expected real experts per token: `E[K_real] = k_max * rho`.
- Null copies to add to routing pool: `M = N * (1 - rho) / rho`.
- With `rho = 0.5`, `E[K_real] = k_max * 0.5` so `k_max = 2 * E[K_real]`.

We adopt `rho = 0.5` as the safe default (paper shows `rho = 0.5` is stable and yields eval gains). See the Null Experts paper for dynamics and failure modes at more aggressive sparsity (rho &lt; 0.5). Note: Null experts are zero-compute and only enlarge the router’s choice set.

---

## Stage-wise proposals (DeepSeek + Null Experts integrated)

Below are the final, paper-faithful stage proposals we should use. These preserve DeepSeek's per-token active experts by adjusting `k_max` when null experts are inserted.

### Stage: 3B (MoE-small, routing-learning)

| Component | Value |
|---|---:|
| Segments (m) | 4 |
| Experts per segment (N_seg) | 8 |
| **Real routed experts (N)** | **32** |
| Shared experts (Ks) | 2 |
| `rho` (data sparsity) | 0.5 |
| `k_max` | **16** |
| Null experts (M) | `N * (1-rho)/rho` = **32** |
| Expected real routed experts `E[K_real]` | `k_max * rho` = **8** |
| Total active per token (incl. shared) | **≈10** (8 routed + 2 shared) |

**Notes:** matches DeepSeek compute budget with data-adaptive allocation; use dense warmup before enabling MoE routing as recommended by the null-expert paper.

---

### Stage: 8B (MoE-medium)

| Component | Value |
|---|---:|
| Segments (m) | 4 |
| Experts per segment (N_seg) | 8 |
| **Real routed experts (N)** | **32** |
| Shared experts (Ks) | 2 |
| `rho` (data sparsity) | 0.5 |
| `k_max` | **16** |
| Null experts (M) | `N * (1-rho)/rho` = **32** |
| Expected real routed experts `E[K_real]` | `k_max * rho` = **8** |
| Total active per token (incl. shared) | **≈10** (8 routed + 2 shared) |

**Notes:** Here as mentioned in project expectation, we are not changing Experts from 3B to 8B transition. Growth team will identify optimal way to transition from 3B to 8B.

---

### Stage: 70B (MoE-large / expert explosion)

| Component | Value |
|---|---:|
| Segments (m) | 4 |
| Experts per segment (N_seg) | 64–128 |
| **Real routed experts (N)** | **256–512** |
| Shared experts (Ks) | 1 |
| `rho` | 0.5 |
| `k_max` | **16** |
| Null experts (M) | **256–512** |
| Expected real routed experts `E[K_real]` | **8** |
| Total active per token (incl. shared) | **≈9** (8 routed + 1 shared) |

**Notes:** this preserves DeepSeek-V3 compute while massively increasing capacity. Keep expert FFN shapes fixed across scale.

---

## Why this design is safe and paper-faithful

1. **Preserves DeepSeek active-expert invariants** — we do not change the per-token compute ceiling; we only gate allocation with nulls.
2. **Scales capacity by increasing N** — exactly as DeepSeek demonstrates (increase total experts, keep active per-token fixed).
3. **Uses the null-expert stable regime** (`rho = 0.5`) to get evaluation gains shown in the Null Experts paper while avoiding the high-sparsity collapse described in Appendix A of that paper.
4. **Retains shared experts** for stabilization at smaller sizes and decays them at large scale to allow routed specialization.
5. **Implements null experts cheaply** — no extra compute; router-only modifications and duplication of the null logit (M copies) are sufficient (per the paper’s implementation notes).

---

## Failure modes & mitigations (from Null Experts paper)

1. **Router resolution collapse at very low rho**: if `rho << 0.5` gradients discriminating between real experts can be reduced. *Mitigation:* avoid rho &lt; 0.5 in production runs; if experimenting with lower rho, track router entropy and identity metrics.

2. **Thresholding instability / polarization**: at aggressive sparsity the model may polarize (most tokens get 0 real experts). *Mitigation:* prefer `rho >= 0.5`, warm up with dense steps, use z-loss and load-balancing weights recommended in the paper.

3. **Auxiliary-objective interference**: balancing loss spreads mass across (N + M) slots; at high M the auxiliary objective might incentivize using nulls rather than improving real expert specialization. *Mitigation:* tune balancing weight and monitor per-expert specialization metrics; consider asynchronous balancing as recommended in the paper’s infra notes.

---

## Implementation checklist (practical)

- [ ] Dense warmup (20k steps or similar) before enabling MoE and null routing.
- [ ] Router expansion to N + M logits (duplicate null logit M times). Follow Algorithm 1 style modifications.
- [ ] Keep `k_max = 16` when `rho = 0.5` to preserve `E[K_real] = 8`.
- [ ] Use bias-only (aux-loss-free) balancing where DeepSeek uses it for large-scale runs; with null experts, apply standard global load balancing over N + M slots but tune weight carefully (paper used `L_bal` weight ~2e-2 and z-loss ~1e-3 as a starting point).
- [ ] Monitor: zero-compute token ratio, per-expert token counts, router entropy, and downstream eval metrics to detect collapse or polarization.

---

## Consolidated final table (all stages)

| Stage | Real Experts (N) | Null Experts (M) | Shared (Ks) | `k_max` | `rho` | `E[K_real]` | Total Active (incl. shared) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3B (MoE-small) | 32 | 32 | 2 | **16** | 0.5 | **8** | **≈10** |
| 8B (MoE-medium) | 64 | 64 | 2 | **16** | 0.5 | **8** | **≈10** |
| 70B (MoE-large) | 256–512 | 256–512 | 1 | **16** | 0.5 | **8** | **≈9** |

---
