## Purpose

This document formally evaluates and validates the proposed **model growth cadence**:

**1B Dense → 3B MoE-small → 8B Dense-deep (same MoE expert configuration) → 70B MoE-large**

The goal is to determine whether this cadence is:

* **causally helpful** for routing stability and expert specialization
* **aligned with empirical evidence** from successful MoE systems
* **safe under hard compute and stability constraints**

The outcome of this document is a **binary recommendation (YES / NO)** with explicit conditions and rollback triggers.

---

## Proposed Cadence (Restated)

| Stage | Model Type | Purpose | Key Constraint |
| :--- | :--- | :--- | :--- |
| 1B | Dense | Learn clean token representations | Baseline for SLM loss matching |
| 3B | MoE-small | Learn routing & expert identity | Routing Health Gate: Entropy < 1.5 bits |
| 8B | Deep-MoE | Consolidate representations | Topology Frozen: No count changes |
| 70B | MoE-large | Exploit scale via expert explosion | Expert Count scaling only |

Key properties:

* MoE is **FFN-only**
* MoE is present **in every layer** (no sparse placement schedule)
* **Expert configuration is frozen** after 3B and reused
* Only expert **count scales**, not dimensions, at 70B

---

## Core Questions

1. Does this cadence **reduce routing instability**, or merely delay it?
2. Is the cadence **causally helpful**, not just operationally convenient?
3. Is the cadence **consistent with how successful MoE systems actually scaled**?

---

## Question 1

### Does Dense → MoE → Dense-deep → MoE-large reduce routing instability?

### Finding

**Yes. Routing instability is reduced, not delayed.**

### Evidence-Based Reasoning

Across MoE literature, routing failures:

* emerge **early in training**
* are **structural**, not stochastic
* persist and amplify when scaling if not corrected early

Recent large-scale MoE systems (notably DeepSeek) show that:

* expert identity forms early
* later scaling amplifies existing specialization
* scaling does *not* repair poor routing decisions

By introducing MoE at **3B**, routing failures:

* surface quickly
* are observable under telemetry
* can be corrected before scale makes them irreversible

In contrast, systems that introduced MoE directly at extreme scale (e.g. GShard, early Switch) required:

* auxiliary routing losses
* token dropping
* overprovisioned capacity
  to remain stable.

These mechanisms mask instability rather than resolve it.

### Conclusion (Q1)

The proposed cadence **materially reduces routing instability** by forcing routing correctness to be learned and validated at small scale, before amplification.

---

## Question 2

### Is this cadence causally helpful, or merely convenient?

### Finding

**The cadence is causally helpful.**

### Causal Decomposition of Learning Problems

The cadence deliberately separates four learning problems that otherwise interfere:

---

### Phase 1 — 1B Dense: Representation Formation

* No routing noise
* Clean token manifolds
* Strong inductive bias for semantic separation

Dense models are empirically strongest at learning base representations without sparsity-induced noise.

---

### Phase 2 — 3B MoE-small: Routing & Expert Identity Learning

This is the **only phase** optimized for routing learning.

At this scale:

* capacity is sufficient for specialization
* routing errors are visible
* expert collapse cannot hide behind scale

This mirrors how DeepSeekMoE was validated (~2B scale) before scaling further.

---

### Phase 3 — 8B Dense-deep (Same MoE Config): Consolidation

Key property: **expert configuration is frozen**.

Increasing depth and width:

* strengthens representations flowing *through* experts
* improves robustness to routing noise
* does not redefine expert boundaries

This phase consolidates expert semantics instead of perturbing them.

---

### Phase 4 — 70B MoE-large: Expert Explosion

Only after routing and expert identity are stable:

* expert count increases
* compute efficiency is exploited
* scale amplifies known-good behavior

This avoids the common failure mode where scaling magnifies routing defects.

---

### Conclusion (Q2)

The cadence is **causally helpful** because it isolates:

* representation learning
* routing learning
* consolidation
* capacity expansion

into non-interfering stages.

---

## Question 3

### Is this cadence consistent with successful MoE scaling history?

### Cross-System Evidence

| System      | When MoE Introduced     | How Experts Scaled    | Outcome                        |
| ----------- | ----------------------- | --------------------- | ------------------------------ |
| GShard      | Very early, very large  | Immediate explosion   | Feasible but brittle           |
| Switch      | Early                   | Immediate, simplified | Stable but weak specialization |
| DeepSeekMoE | ~2B                     | After validation      | Stable specialization          |
| DeepSeek-V2 | After MoE proven        | Large-scale reuse     | Stable                         |
| DeepSeek-V3 | Same MoE, massive scale | Late explosion        | Highly stable                  |
| Mixtral     | Medium scale            | Small expert count    | Strong but constrained         |

The proposed cadence **most closely matches DeepSeek**, the only lineage that has demonstrated:

* stable MoE at >500B scale
* aux-loss-free routing
* reproducible training without rollbacks

---

## Final Decision

### Cadence Validation — Decision Record

**Status: APPROVED (YES)**

The growth path:

**Dense → MoE-small → Dense-deep → MoE-large**

is:

* empirically justified
* causally sound
* aligned with the only MoE systems that have scaled reliably to extreme sizes

---

## Conditions for Validity (Binding)

This cadence remains valid **only if**:

1. **3B MoE-small passes routing health gates**

   * no expert starvation
   * no collapse or polarization
   * null behavior within target bands

2. **Expert configuration is frozen after 3B**

   * no expert redefinition during 8B consolidation

3. **8B Dense-deep improves stability**

   * no regression in routing metrics
   * loss matches dense SLM expectations

4. **70B introduces only expert count scaling**

   * no new routing mechanisms
   * no new auxiliary losses

Failure of any condition requires cadence re-evaluation.

---

## Rollback Triggers

Immediate review is required if:

* routing instability emerges post-8B
* null begins consuming signal groups
* expert imbalance grows with scale
* training stability degrades at 70B

---

## Conclusion

This cadence converts MoE from a **high-risk scaling gamble** into a **controlled, evidence-based expansion strategy**.

It does not rely on scale to fix routing.
It ensures routing is correct *before* scale makes it expensive.

---
