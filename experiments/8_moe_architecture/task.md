# Team 3: Coreset Engineering

## Category

**Data**

---

## Objective (Brief)

Reduce the **2 trillion token raw corpus** into **stage-specific coresets totaling ~400B tokens** (20B / 40B / 100B / 240B) while preserving **representational coverage, curriculum integrity, and learning dynamics**.

The goal is not maximal compression, but **efficient signal concentration**: remove redundancy and low-value repetition while retaining the breadth, tail phenomena, and difficulty progression required for stable training and benchmark emergence.

**NOTE**: Additional 5-10% data may be added to each stage by **Synthetic Data & Self Distillation** Team. 

DO NOT FORGET SFT, ALIGNMENT, and POST-TRAINING DATASETS

---

## Purpose (Charter)

To make large-scale training **computationally viable without collapsing the data distribution**.

A correct coreset:

* accelerates early learning
* improves signal-to-noise
* preserves rare but capability-critical content

An incorrect coreset silently amputates the model’s future abilities.

This team exists to prevent that failure mode.

---

## What You Must Do (Brief)

* Design and implement a **deterministic coreset selection pipeline** that:

  * operates at **chunk level**
  * produces **stage-specific, non-overlapping coresets**
  * adheres strictly to curriculum ratios (B0–B5, domain groups)
* Enforce **gradual curriculum transitions** across stages:

  * no abrupt ratio jumps
  * smooth interpolation between milestone profiles
* Remove redundancy using **cheap, scalable methods**:

  * exact and near-deduplication
  * token-signature diversity
  * stratified sampling within buckets
* Validate coresets via **cheap proxy training runs** and diagnostic metrics.

---

## Responsibilities (Charter)

### Distribution Preservation

* Preserve coverage across:

  * curriculum difficulty bands (B0–B5)
  * domain groups (code, math, reasoning, agentic, Indic, clean web)
* Ensure **protected slices** (especially B4/B5 reasoning, code, agentic traces) are never dropped due to density-based selection.

### Stage Integrity

* Produce **distinct coresets per stage**:

  * no data point appears in more than one stage
  * stages share **structural buckets**, not examples
* Respect target budgets exactly:

  * 1B stage: 20B tokens
  * 3B stage: 40B tokens
  * 8B stage: 100B tokens
  * MoE stage (16B active): 240B tokens

### Curriculum Smoothness

* Treat stage profiles as **milestones along a continuous trajectory**, not isolated regimes.
* Enforce **rolling-window anti-spike constraints** so that:

  * no band or domain exceeds allowed share within any rolling window
  * transitions between stages are gradual and monotonic
* Prevent shock effects that destabilize training or distort learning signals.

### Reproducibility

* Ensure the pipeline is:

  * deterministic
  * seed-controlled
  * versioned
* Emit manifests that allow downstream teams to audit and replay selections exactly.

---

## Inputs

* Clean, approved raw datasets and metadata from **Team 1**
* Curriculum definitions, ratios, and guardrails from **Team 2**
* Early benchmark targets and proxy evaluation criteria from benchmarking teams (as available)

---

## Tools

* Chunking and hashing utilities
* Exact and near-duplicate detection (hashing, MinHash / SimHash)
* Token-signature analysis (token histograms, compression proxies)
* Optional small embedding models for **limited protected slices only**
* FAISS / clustering (restricted to small subsets)
* Python (NumPy, PyTorch)
* Visualization notebooks for coverage and ablations

---

## Required Submissions (Brief)

* `coreset_builder.py`
  Deterministic, configurable pipeline for coreset generation.
* Stage-wise index manifests:

  * selected indices
  * token counts
  * band/domain composition
  * seeds and config hashes
* Ablation and validation report:

  * methods evaluated
  * achieved reduction ratios
  * coverage diagnostics
  * proxy training comparisons (coreset vs full)

---

## Outputs (Charter)

* Four **stage-specific coresets** totaling ~400B tokens
* Reproducible index files and manifests
* Clear justification for:

  * selection strategy
  * protection rules
  * curriculum adherence
* Evidence that the coreset improves efficiency without degrading learning.

---

## Success Criteria

* Coresets are approved for training use
* Early training converges faster or equally fast versus full data
* Early benchmark deltas (MMLU, code, math, agentic, Indic) are not degraded
* No curriculum violations or domain spikes are observed
* Downstream teams can consume outputs without rework

---

## Failure Conditions

* Curriculum ratios are violated
* Sudden domain or difficulty spikes occur
* B4/B5 signal is diluted or lost
* Proxy runs show slower learning or degraded benchmarks
* Results are non-deterministic or irreproducible

---

## Dependencies (Blocking Prerequisites)

* **Team 1** — finalized raw dataset pool and metadata
* **Team 2** — frozen curriculum structure, ratios, and guardrails

---

## Downstream Teams Blocked If Delayed

* **Team 10** (training and scaling teams)
* Any team dependent on finalized training mixtures

---

## Lifecycle

### Setup / Preparation

**Jan 29 – Jan 30**

* Finalize selection unit and dedup strategy
* Lock curriculum constraints and rolling-window rules
* Implement baseline pipeline skeleton

### Peak Execution

**Jan 30 – Feb 5**

* Run coreset selection per stage
* Validate curriculum adherence and smooth transitions
* Execute proxy training comparisons
* Produce ablation and coverage reports

### Monitor / Support

**Feb 2 – Feb 4**

* Final validation
* Stress-test anti-spike constraints
* Package and audit outputs

### Done / Frozen

**Feb 9**

* Coresets locked
* Any changes require escalation due to training distribution impact

---

