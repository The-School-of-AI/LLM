# Purpose

This document records **binding architectural risk decisions** taken by the MoE Architecture Team to de-risk training and scaling of the 70B model.
Each decision is binary (GO / NO-GO), evidence-backed, and aligned with routing health gates, compute ceilings, and downstream implementation constraints.

The goal is to **prevent subtle, expensive failures** in routing stability, gradient flow, and expert specialization that typically emerge only at scale.

---

## Decision 1 — Null Experts (Inside Top-k)

### Status: **GO (with hard constraints)**

### Decision Summary

We approve the use of **zero-compute null experts** that **compete inside top-k routing** as a first-class architectural component of all MoE stages.

This decision is based on direct empirical and theoretical evidence that null experts are the *only known mechanism* that:

* recovers data sparsity **without violating autoregressive causality**
* preserves the dense solution space (loss-free fallback)
* allows junk tokens to be dropped predictably under bias control

### Rationale

Analysis of [Improving MoE Compute Efficiency by Composing Weight and Data Sparsity](https://arxiv.org/abs/2601.15370) demonstrates that:

* zero-output null experts preserve output magnitude via renormalization
* routing remains reversible and bias-controllable
* junk token groups naturally polarize toward null (≥60%) without supervision
* high-signal tokens retain access to real experts under moderate sparsity

Crucially, the paper shows that **copy / identity / residual-pass nulls break the solution space** and lead to irreversible routing collapse. These are therefore explicitly disallowed.

### Hard Constraints

* Null experts **must**:

  * produce exact zero output
  * compete natively inside top-k
  * renormalize routing weights over real experts only
* Operating regime:

  * effective sparsity ρ ∈ [0.5, 0.67]
  * junk token groups: 60–80% null routing
  * signal groups: monitored and gated
* Control surface:

  * bias-only routing control
  * no auxiliary router losses in base spec

### Explicitly Disallowed

* Copy / identity / constant experts
* External null paths outside top-k
* Aggressive sparsity without telemetry gates

---

## Decision 2 — Multi-Token Prediction (MTP)

### Status: **NO-GO for MoE routing stages**

#### Stance:

We sacrifice MTP’s potential inference speed for training stability and routing signal purity.

### Decision Summary

Multi-Token Prediction (MTP) is **explicitly disallowed** in:

* 3B MoE-small (routing-learning stage)
* 70B MoE-large (base spec)

MTP may be used only in **dense-only experiments**, **post-routing ablations**, or **inference-only speculative decoding research**.

### Rationale

The routing-learning phase depends on **early, sharp representational separation** so that:

* expert identities form quickly
* router entropy collapses cleanly
* null competes correctly against low-information tokens

Evidence from the Meta MTP paper (*Better & Faster LLMs via Multi-Token Prediction*) shows that:

* MTP smooths representations early in training
* small and mid-scale models regress before benefits appear
* gains disappear once capacity is sufficient

DeepSeek-V3 uses MTP successfully at extreme scale, but **does not attribute routing stability or expert specialization to MTP**, nor provide evidence that MTP improves routing dynamics.

Mechanistically, MTP introduces **multi-future gradient interference** into the shared trunk, which:

* blurs token-level distinctions
* delays expert specialization
* increases the risk of null stealing borderline signal tokens

These risks directly conflict with the purpose of the 3B MoE-small stage.

### Override Condition

MTP may only be reconsidered if small-run evidence demonstrates:

* faster router entropy collapse
* earlier expert specialization
* stable null behavior under Team 7 metrics

Absent such evidence, MTP remains **out of spec**.

---

## Decision 3 — Manifold-Constrained Hyper-Connections (mHC)

### Status: **GO (depth-gated | 70B ONLY)**

### Decision Summary

We approve **mHC residual connections** for the **70B MoE-large stage only**.
mHC is **not used** in 1B dense, 3B MoE-small, or 8B dense-deep stages.

### Rationale

Standard residuals are stable but limit expressivity at extreme depth.
Unconstrained Hyper-Connections (HC) are provably unstable due to loss of identity mapping, leading to gradient explosion and training collapse.

*mHC: Manifold-Constrained Hyper-Connections* demonstrates that:

* constraining residual mixing matrices to the Birkhoff polytope restores identity mapping
* gradient norms remain bounded across depth
* stability is preserved under MoE sparsity
* training remains stable in 27B MoE models with improved loss

mHC provides **mathematical guarantees**, not heuristic stabilization, making it suitable for deep (70B-class) models.

### Constraints

* Enabled **only** at 70B MoE-large
* Expansion rate fixed (n = 4)
* No interaction with routing logic
* No auxiliary losses
* Requires infra sign-off for ~6–7% overhead

### Rollback Triggers

* Gradient norm instability
* Throughput regression beyond budget ceiling
* Unexpected interaction with routing telemetry

---

## Consolidated Risk Posture

| Component            | Decision  |
| -------------------- | --------- |
| Null experts         | **GO**    |
| MTP (routing stages) | **NO-GO** |
| mHC (70B only)       | **GO**    |

Together, these decisions:

* preserve routing signal purity
* prevent early expert identity collapse
* maintain gradient stability at depth
* keep all control surfaces loss-free and reversible

This document is **binding** unless overturned by new empirical evidence reviewed and approved by the MoE Architecture Team.