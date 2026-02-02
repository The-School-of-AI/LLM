# Team 2 — Data Bucketing Implementation Brief

(Path A: A1 & A2)

## 1. Purpose of This System

This system assigns difficulty bands (B0–B5) to training samples so that downstream curriculum scheduling can deliberately control what kind of signal the model sees at different growth stages.

This is not a data quality classifier.
This is not a tokenizer-dependent system.
This is not a learned model.

It is a conservative, rule-based, explainable bucketing system whose only goal is to prevent premature exposure to complexity and reasoning.

---

## 2. Non-Goals (Explicit)

The implementation team should not attempt to:
*   build a machine-learning classifier
*   perfectly rank difficulty
*   infer curriculum proportions
*   rewrite or summarize content
*   depend on tokenizer internals
*   enforce sampling ratios (handled later)

This system exists solely to label samples with a difficulty band and rationale.

---

## 3. Conceptual Model

**Core Principle**

Difficulty is defined by absorbable structural complexity, not topic or prestige.

A “hard” topic can appear in a low band if it is presented simply.
A “simple” topic can appear in a high band if it requires deep reasoning.

---

## 4. Unit of Classification
*   Atomic unit: training sample
*   typically a document or document chunk
*   already produced by upstream data processing
*   The system does not operate on:
    *   raw crawls
    *   sentence fragments
    *   entire datasets as monoliths

Each input sample must be classified independently.

---

## 5. Inputs Expected

Each sample provided to the system should include:
*   text (string)
*   language (e.g. en, hi)
*   domain_tags (list of strings)
*   source_id (dataset identifier)
*   approximate length indicators (chars / words)

Tokenizer-derived token counts are optional and not required.

---

## 6. Outputs Required

For each input sample, the system must output:
*   Assigned band: B0–B5
*   Decision rationale:
    *   which signals were observed
    *   which bands were ruled out
*   Flags (optional):
    *   suspected boilerplate
    *   suspected verbosity without depth
    *   suspected reasoning leakage

The output must be auditable by humans.

---

## 7. Difficulty Band Definitions (Authoritative)

**B0 — Nursery**

Purpose: language fundamentals
*   Simple declarative text
*   Local sentence-level dependencies only
*   No explanations, no reasoning, no steps

Excludes:
Any “why/how” explanations, procedures, or logic chains

---

**B1 — Primary**

Purpose: fluent everyday language
*   Clean narrative or exposition
*   Common knowledge
*   Shallow structure

Allows: definitions without derivation
Excludes: multi-step logic or algorithms

---

**B2 — High School**

Purpose: structured knowledge without explicit reasoning
*   Concepts build across paragraphs
*   Reasoning is implicit, not spelled out
*   Educational tone

Excludes: explicit chains of reasoning, proofs, formal steps

---

**B3 — Undergraduate**

Purpose: controlled reasoning emergence
*   Multi-step explanations
*   Clear procedural structure
*   Non-trivial code or tutorials

Allows: short, curated reasoning
Excludes: verbose “thinking aloud”

---

**B4 — Graduate**

Purpose: explicit reasoning and abstraction
*   Formal reasoning or derivation
*   Long dependency chains
*   High abstraction density

Requires: reasoning to be necessary, not decorative

---

**B5 — PhD**

Purpose: push ceiling without destabilization
*   Novel synthesis or planning
*   Cross-domain abstraction
*   Very low redundancy

Strictly rare and conservative

---

## 8. Bucketing Signals (Heuristic Guidance)

These signals are indicators, not a scoring system.

**Structural Depth**
*   Paragraph count
*   Section / heading structure
*   Nested explanations

**Reasoning Markers**
*   Causal language (“therefore”, “because”)
*   Enumerated steps
*   Proof-style phrasing (“assume”, “let X be”)

**Dependency Horizon**
*   Sentence-level → lower bands
*   Section / document-level → higher bands

**Code Presence**
*   Inline examples → B1–B2
*   Functions / APIs → B3
*   Systems / architecture → B4–B5

**Cleanliness Override**

Verbose, repetitive, or SEO-style content must be downgraded or rejected, regardless of apparent complexity.

---

## 9. Decision Rules (Critical)
*   Bands must be assigned conservatively
*   If uncertain, assign the lower band
*   Decisions must be explainable post-hoc
*   Dataset reputation must not influence band assignment

This system is designed to prevent over-classification, not under-classification.

---

## 10. What This System Does Not Decide

The bucketing system does not decide:
*   how much of each band is used
*   when bands appear during training
*   model architecture
*   MoE routing
*   curriculum schedules

Those are handled by regime composition later.

---

## 11. Tokenizer Independence

This system must:
*   operate without tokenizer features
*   use character / word / structure proxies
*   remain valid if the tokenizer changes

Tokenizer-derived signals may be added later as optional refinements, not dependencies.

---

## 12. Expected Quality Bar

This system is considered successful if:
*   B0–B2 rarely contain explicit reasoning
*   B4–B5 are rare and high-signal
*   Human audits find errors to be bounded and explainable
*   No model instability is caused by premature difficulty exposure

Perfect classification is not required.

---

## 13. Philosophy (Read This Twice)

This system exists to avoid silent failure, not to optimize labels.

Conservatism, simplicity, and auditability are features, not shortcomings.
