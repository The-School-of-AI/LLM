# Curriculum Architecture — Rationale & Design Notes

> Owner: Team 2 – Curriculum Architects

⸻

### Why curriculum is defined

The curriculum is the constitution/the policy, not implementation.

It defines non-negotiable training data guarantees.

Encoding it in YAML ensures:
  - it is auditable
  - it is diffable
  - it is reviewable across teams
  - it can be validated independently of training code

⸻

### Why we explicitly define guarantees (even if they seem obvious)

The guarantees section exists to prevent silent violations, not to restate best practices.

In large training runs:
  - nondeterminism doesn’t crash
  - data drift doesn’t alert
  - mistakes surface months later, when compute is already spent

By codifying guarantees:
  - Teams can audit against explicit expectations
  - Training failures can be traced to which guarantee broke
  - “We assumed X” is no longer an excuse

⸻

### Language policy: why English + Hindi only

This is not a philosophical choice; it is a capacity allocation decision.

Reasons:
  - Token budget is finite
  - Multilingual coverage scales sublinearly with token count
  - Supporting many languages poorly is worse than supporting a few well

Hindi is included because:
  - it is structurally different from English
  - it stress-tests tokenizer and morphology
  - it aligns with project goals

Other languages are excluded because:
  - they dilute early representation learning
  - they distort difficulty bands via tokenization
  - they provide low marginal benefit at current scale

⸻

### Why context length is fixed at 4K from day one

Short-context warmups are an outdated optimization.

Problems with short-context pretraining:
  - models overfit to local patterns
  - long-range attention emerges late and unreliably
  - later extension causes representation mismatch

By fixing context early:
  - positional statistics stabilize early
  - long-context reasoning emerges naturally
  - no architectural “context shock” occurs later

We control effective length via data, not architecture.

⸻

### Why tokenizer proxy signals exist at all

We cannot manually label trillions of tokens by difficulty.

Tokenizer statistics give us:
  - a cheap, model-agnostic proxy for conceptual rarity
  - a way to detect hidden complexity
  - a cross-dataset normalization mechanism

Rare tokens ≠ intelligence
But high-density rare tokens reliably correlate with abstraction, math, code, and planning.

Tokenizer proxy is:
  - not perfect
  - not semantic
  - but scalable, deterministic, and auditable

Which is exactly what curriculum needs.

⸻

### Why difficulty bands are discrete (B0–B5)

Curriculum needs hard boundaries, not vibes.

Discrete bands allow:
  - deterministic sampling
  - clear staging
  - enforceable caps and floors
  - reproducible experiments

Each band corresponds to a qualitative shift in cognitive demand:
  - B0–B1: language modeling
  - B2: structured knowledge
  - B3: reasoning emergence
  - B4: explicit abstraction
  - B5: planning and systems thinking

Continuous difficulty sounds elegant, but cannot be enforced at scale.

⸻

### Why Chain-of-Thought is gated and capped

Chain-of-thought is dangerous when overused.

Failure modes:
  - models learn to imitate reasoning instead of doing it
  - verbosity becomes correlated with correctness
  - generalization degrades

Therefore:
  - CoT is introduced only after reasoning emerges
  - it is explicitly gated
  - it is never allowed to dominate tokens

Distilled views remain primary.
CoT is scaffolding, not the house.

⸻

### Why synthetic data is capped (5–10%)

Synthetic data is an accelerant, not a foundation.

If it dominates:
  - model style collapses
  - teacher artifacts leak
  - diversity decreases

If it is absent:
  - known weaknesses persist too long
  - retraining becomes necessary

The 5–10% window:
  - is large enough to move capability
  - small enough to preserve natural data statistics
  - aligns with findings from recent frontier models

⸻

### Why stage ratios are hard numbers (not “tunable”)

Curriculum drift is catastrophic and silent.

Hard ratios ensure:
  - comparability across runs
  - valid ablations
  - post-hoc analysis is meaningful

If ratios were adjustable mid-training:
  - no run would be defensible
  - regressions could not be diagnosed
  - benchmarks would be uninterpretable

Adjustments belong before freeze, not during training.

⸻

### Why perplexity is bounded per band (not globally)

Raw perplexity is misleading.

High perplexity can mean:
  - deep math
  - clean code
  - or garbage OCR

Low perplexity can mean:
  - clean prose
  - or templated spam

Therefore:
  - perplexity is band-conditioned
  - each difficulty band has a plausible perplexity range
  - samples outside that range are statistically abnormal for that band

This removes:
  - junk masquerading as “hard”
  - boilerplate masquerading as “clean”

⸻

### Why rolling-window constraints exist

Even correct global ratios can fail locally.

Example failure:
  - reasoning data spikes in a short window
  - optimizer destabilizes
  - loss explodes
  - training “recovers” but representations degrade

Rolling-window limits:
  - prevent shock exposure
  - enforce smooth transitions
  - protect optimizer dynamics

This is curriculum in time, not just in aggregate.

⸻

### Why violations halt or reject (instead of down-weighting)

Silent fixes are worse than loud failures.

Down-weighting:
  - hides upstream errors
  - creates non-obvious behavior
  - makes reproduction impossible

Hard failure:
  - forces correction at source
  - preserves trust in the pipeline
  - prevents gradual corruption

If curriculum is violated, training should not proceed.

⸻

### Final design philosophy

Curriculum errors do not crash training.
They quietly waste trillions of tokens.

This file exists so that:
  - no team relies on intuition
  - no behavior is “implied”
  - no assumption survives unchallenged

If something is important, it is explicit.
If it is not explicit, it is not guaranteed.
