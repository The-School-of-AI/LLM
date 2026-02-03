```markdown
# Synthetic Data Generation Plan

**Team 4: Synthetic Data & Self-Distillation**  
**70B LLM — Pre-planned buffers, gated by diagnostics, solver-verified, curriculum-aligned**

---

## 1. Purpose & Principles

- **Pre-planned, not reactive.** All synthetic data is prepared in advance and can be injected if diagnostic gates show weakness. Data may be partially or entirely unused. **Unprepared buffers are failure.**
- **Targeted.** Synthetic data targets **skills/failure modes**, not benchmark items.
- **Bands.** Synthetic data is **B3–B5 only** (B0–B2 remain curriculum/coreset).
- **Cap.** Synthetic share is capped at **5–10%** per stage (and never dominates any rolling window).
- **Two-view policy.** Every accepted item has:
  - **Distilled view (primary):** final answer + short justification/outline; higher sampling weight.
  - **CoT view (secondary, gated):** `<think>…</think>`, bounded length, never dominant.
- **Compute discipline.** Prefer programmatic generation + deterministic verification; use teacher models mainly for surface-form diversity (paraphrase), not correctness.

---

## 2. Diagnostics Preconditions 

Synthetic shards are considered **“ready-to-inject” only if the diagnostic suite is stable and locked**.

### 2.1 Required diagnostic patches (minimum)

These must be applied **before** generating any shard that claims it is “gated by TEST-*”.

#### TEST-009 (Loop trace): initialize `total` (invalid prompt fix)

**Before**
python
for i in range(1, 6):
    total += i
print(total)
`

**After**

python
total = 0
for i in range(1, 6):
    total += i
print(total)


#### TEST-035 (JSON completion): replace non-deterministic expectation

Replace “any city name” with a deterministic closure or schema-checked options.

Example replacement:

* Prompt: `{"name":"Alice","age":25,"city":"Paris"`
* Expected: `}`

#### TEST-036 (Table reading): remove instruction-like QA formatting

Convert to continuation with fixed label.

Example replacement:

* Prompt:

  
  Name Age
  Alice 25
  Bob 30
  Bob_age =
  
* Expected: `30`

#### Ambiguity removal (math wording)

Avoid “3 times older”; use “three times as old”.

---

### 2.2 Suite versioning and lock

* Add `suite_version` (e.g., `diag_suite_v1.1`) to `diagnostic_tests.json`.
* Every synthetic sample must reference the locked suite version via metadata:

  * `diag_suite_version: "diag_suite_v1.1"`
  * `linked_tests: ["TEST-001", "TEST-003", ...]`

---

### 2.3 Scoring primitive (execution requirement)

Avoid brittle absolute `P(token) > threshold` gating where tokenization can change outcomes.

**Allowed scoring modes**

* **Preferred**: generate short completion → **deterministic verifier pass/fail**
* **Alternative**: log-likelihood ratio vs distractors
  `LL(correct) - max_i LL(distractor_i) > margin`

---

## 3. Mapping: Diagnostic Buckets → Skill Buckets → Pretraining-Compatible Targets

### 3.1 Diagnostic scope (base-model testable)

Diagnostics cover **8 testable areas** (TEST-001…036) and map to the broader taxonomy in `skill_buckets.md`.

### 3.2 Updated mapping table (clarified PRE-safe structured outputs)

| Diag. bucket                            |   Tests | Primary taxonomy bucket IDs                   | Priority     | Notes                                                                                  |
| --------------------------------------- | ------: | --------------------------------------------- | ------------ | -------------------------------------------------------------------------------------- |
| 1. Multilingual LM (English)            | 029–031 | `FND-LEX-EN`                                  | MEDIUM       | Hindi deferred for base diagnostics; Hindi buffers optional (not gated by this suite). |
| 2. Long-Context Retention               | 032–034 | `FND-LCX`                                     | LOW          | Base suite is short-context; prepare longer-needle buffers for later.                  |
| 3. General Knowledge                    | 025–028 | `FND-FACT`                                    | MEDIUM       | Synthetic should be **closed-world micro-KB** to reduce contamination risk.            |
| 4. Logical Reasoning                    | 019–024 | `RSN-LOG` (and light `RSN-CAUS`)              | HIGH         | Prefer solver-verified logical forms + paraphrase diversity.                           |
| 5. Mathematical Reasoning               | 001–008 | `RSN-ARITH`, `RSN-ALG`, `RSN-WPT`             | **CRITICAL** | Deterministic verification mandatory (arith + SymPy).                                  |
| 6. Code Understanding & Error Detection | 015–018 | `CODE-COMP` (and “DBG-like probes”)           | HIGH         | Keep fix-generation separate; diagnostics are detection/classification.                |
| 7. Code Generation & Synthesis          | 009–014 | `CODE-SYN`, `CODE-GEN-T1`                     | **CRITICAL** | Verification via code execution + unit tests / golden outputs.                         |
| 8. Structured Output (PRE-safe)         | 035–036 | `PRE-STRUCT` (subset aligned to `ALN-STRUCT`) | LOW          | Base-model formatting probes only; instruction-following alignment kept for SFT.       |

---

## 4. Stage & Curriculum Alignment

Synthetic data is stage-tagged and respects Team 2’s frozen curriculum and caps.

| Stage         |  Params | Tokens | Bands in scope    | Planned injection | Prepared buffers                                              |
| ------------- | ------: | -----: | ----------------- | ----------------- | ------------------------------------------------------------- |
| Stage 1       |      1B |    20B | B0–B2             | **0%**            | None required                                                 |
| Stage 2       |      3B |    40B | mostly B0–B2      | **0% (default)**  | **YES: S2-B3 micro-buffer** (emergency / early B3 on-ramp)    |
| Stage 3       |      8B |   100B | B3 early, B4 late | 5–10% (gated)     | Full pretraining buffers for buckets 1–8                      |
| Stage 4 (MoE) | 16B/70B |   240B | B3–B5             | 5–10% (gated)     | Full pretraining buffers + separate SFT/RLHF buffers prepared |

**Anti-spike**: synthetic integrates into coresets; never dominates rolling windows.

---

## 5. Ready-to-Inject Shard Catalog 

> Sizes below are **buffer targets**, not mandatory injection volumes.
> Injection remains capped at 5–10% per stage and controlled by diagnostics.

### 5.1 Stage 2 — S2-B3 Micro-Buffer (Prepared, 0% planned injection)

| Shard               | Band | Buckets     | Linked tests | Verifier     | Target size |
| ------------------- | ---- | ----------- | ------------ | ------------ | ----------: |
| S2-MATH-CORE-MICRO  | B3   | `RSN-ARITH` | 001,005,006  | arithmetic   |    2–3M tok |
| S2-CODE-CORE-MICRO  | B3   | `CODE-SYN`  | 010,011,014  | exec+golden  |    2–3M tok |
| S2-LOGIC-CORE-MICRO | B3   | `RSN-LOG`   | 019,021,024  | logic_solver |    1–2M tok |

### 5.2 Stage 3 — Main Pretraining Buffers

| Shard            | Band  | Buckets                  | Linked tests    | Verifier               | CoT ratio | Target size |
| ---------------- | ----- | ------------------------ | --------------- | ---------------------- | --------: | ----------: |
| S3-MATH-ARITH    | B3    | `RSN-ARITH`              | 001,005,006,008 | arithmetic             |       ≤5% |  12–18M tok |
| S3-MATH-WPT      | B3–B4 | `RSN-WPT`                | 003,007,008     | solver+unit            |      ≤10% |  15–22M tok |
| S3-MATH-ALG-LIN  | B4    | `RSN-ALG`                | 002             | SymPy                  |      ≤15% |   8–12M tok |
| S3-CODE-COMPLETE | B3    | `CODE-SYN`,`CODE-GEN-T1` | 010,011,014     | exec                   |       ≤5% |  10–15M tok |
| S3-CODE-TRACE    | B3    | `CODE-COMP`              | 009,012,013     | exec+golden            |       ≤5% |   6–10M tok |
| S3-LOGIC-CORE    | B3–B4 | `RSN-LOG`                | 019–024         | logic_solver           |      ≤10% |   8–12M tok |
| S3-CTX-TRACK     | B3–B4 | `FND-LCX`                | 032–034         | retrieval              |      ≤10% |    4–8M tok |
| S3-PRE-STRUCT    | B4    | `PRE-STRUCT`             | 035–036         | json_parse/table_parse |      ≤15% |    2–4M tok |
| S3-EN-LEX        | B3    | `FND-LEX-EN`             | 029–031         | closed_set             |       ≤5% |    2–4M tok |
| S3-FACT-MICROKB  | B3    | `FND-FACT`               | patched 025–028 | micro_kb               |       ≤5% |    1–2M tok |

### 5.3 Stage 4 — Expanded Pretraining Buffers (still buckets 1–8)

| Shard                | Band  | Buckets                         | Linked tests    | Verifier         | CoT ratio | Target size |
| -------------------- | ----- | ------------------------------- | --------------- | ---------------- | --------: | ----------: |
| S4-MATH-HARD         | B4–B5 | `RSN-ARITH`,`RSN-ALG`,`RSN-WPT` | 001–008         | solver           |      ≤20% |  30–60M tok |
| S4-CODE-GEN-HARD     | B4–B5 | `CODE-GEN-T1`                   | 009–014         | exec+tests       |      ≤20% |  25–50M tok |
| S4-CODE-COMP-EDGE    | B4–B5 | `CODE-COMP`                     | 015–018         | exec+classify    |      ≤20% |  15–30M tok |
| S4-LOGIC-ROBUST      | B4–B5 | `RSN-LOG`                       | 019–024         | logic_solver     |      ≤20% |  12–25M tok |
| S4-LCX-NEEDLE-LONG   | B4–B5 | `FND-LCX`                       | extends 032–034 | retrieval        |      ≤20% |  10–20M tok |
| S4-PRE-STRUCT-NESTED | B4–B5 | `PRE-STRUCT`                    | 035–036         | json+schema_lite |      ≤20% |   5–10M tok |

---

## 6. Generation Process (Execution details)

### 6.1 Source hierarchy (cost/quality)

* **Tier A (preferred):** programmatic generation + deterministic verification
  (math, logic, LCX needles, structured JSON/table, code templates + tests)
* **Tier B:** teacher-assisted paraphrase / surface randomization
  (varied word problems, varied story contexts, varied code comments)
* **Tier C:** teacher-generated candidates only when verifiers exist and acceptance is strict.

### 6.2 Required sample metadata (JSONL)

Every record must include (minimum):

* `id`, `shard`, `stage`, `band`
* `skill_buckets` (taxonomy IDs), `linked_tests`
* `diag_suite_version`
* `view: "distilled" | "cot"`, `cot_triggered: bool`
* `prompt`, `completion`
* `verifier: {type, status, details_hash}`
* `dedup: {method, score, nearest_neighbor_id?}`
* `source: {generator, teacher_model?, seed}`

### 6.3 Two-view creation rules 

**Distilled view** is always created. **CoT view** is optional and capped.

* **B3**: CoT present in **≤ 5%** of items; `cot_len_cap ≤ 120 tokens`; sampling weight **distilled:cot ≥ 8:1**
* **B4**: CoT present in **≤ 15%**; `cot_len_cap ≤ 220`; sampling weight **≥ 6:1**
* **B5**: CoT present in **≤ 20%**; `cot_len_cap ≤ 300 tokens`; sampling weight **≥ 5:1**

CoT view must be gated with:

* `<think> ... </think>`
* never the dominant training signal

### 6.4 Hard negatives and correction trajectories (CHANGED — safe formats only)

Hard negatives are required but must be **non-poisoning**.

**Allowed formats**

1. **Binary classification**


Problem: ...
Candidate: 49.14
Label: CORRECT


and


Problem: ...
Candidate: 48.14
Label: INCORRECT


2. **Correction trajectory (distilled target ends correct)**


Attempt: ...
Error: ...
Correct: ...
Outline: ...


**Disallowed**

* Free-form wrong-answer samples without explicit “INCORRECT” labeling or correction framing.

### 6.5 Style randomization and anti-fingerprint

* Multiple prompt templates per shard (≥ 20 templates for critical shards)
* Randomize names/contexts/units while keeping solver-verifiable structure
* Ban teacher boilerplate phrases via style filter

### 6.6 General knowledge (CHANGED — closed-world micro-KB)

Replace open-world trivia augmentation with micro-KB tasks.

Example prompt:


Facts:
- Water freezes at 0°C.
- The capital of France is Paris.
Question: The capital of France is
Answer:


Verifier: answer must match a fact in the provided block.

### 6.7 Long-context (execution)

* Provide keyed facts distributed across the context:

  * positions 10% / 50% / 90%
* Require completion of `key = value` line at end.

Verifier: exact match.

### 6.8 Structured output (PRE-STRUCT)

Pretraining-compatible formatting tasks only:

* JSON closure / strict parse
* schema-lite with a fixed small schema (optional)
* table cell extraction via fixed label completion

Verifier: parse success / table parser.

---

## 7. Verification & Filtering 

### 7.1 Multi-stage acceptance gates (must all pass)

1. **Deterministic verifier pass (mandatory where available)**

   * Math: arithmetic + SymPy
   * Logic: truth-table / SAT evaluator
   * Code: execution + unit tests / golden outputs
   * JSON: strict parse (+ schema-lite when used)

2. **Student re-solve agreement (secondary; stage-appropriate)**

   * Use Team 10 checkpoint where available
   * Reject if systematic disagreement (or downgrade band / mark as stretch with tiny weight)

3. **Style fingerprint filter**

   * Remove repeated teacher artifacts
   * Enforce variability quotas

4. **Dedup + benchmark adjacency**

   * Dedup within synthetic set (minhash/near-neighbor)
   * Dedup against Team 1 pool where hashes/embeddings exist
   * Apply benchmark-adjacency filters (template similarity, n-gram overlap, known phrasing)

### 7.2 Rejection policy (explicit)

Reject:

* fluent-but-incorrect
* unverifiable answers
* ambiguous prompts
* benchmark look-alike phrasing
* near-duplicates above threshold

---

## 8. Gating, Injection, and Anti-Spike Policy (NEW — enforceable)

### 8.1 Gate criterion (per shard)

A shard is eligible for injection if linked test performance indicates weakness, e.g.:

* critical buckets (math/code): pass rate < target floor (stage-dependent)
* OR consistent failure mode signature (clustered errors)

### 8.2 Weight update rule (anti-spike control law)

For each bucket/shard maintain an injection weight `w` updated at evaluation checkpoints:

$
* `w_raw = clamp(alpha * (target_pass - current_pass), 0, cap_stage)`
* `w = (1 - beta) * w_prev + beta * w_raw`
$

Defaults:

* `cap_stage = 0.05` (Stage 2), `0.10` (Stage 3/4)
* `beta = 0.05` (slow changes prevent spikes)
* `alpha` tuned so typical gaps map to 1–10% range

### 8.3 Rolling-window constraint (hard)

Synthetic must not exceed the curriculum rolling-window cap:

* `synthetic_share(window) <= window_cap` (from Team 2)

### 8.4 Regression guardrail

After any injection decision:

* re-run full diagnostic suite
* block or reduce synthetic if any non-target bucket regresses by > **5% absolute**

---

## 9. SFT & Post-Training Buffers 

These datasets are **prepared in advance** but stored separately and used only in SFT/RLHF phases.

### 9.1 SFT buffers (examples)

* `RSN-ADVMATH` (B4–B5): calculus/probability/proofs (solver-verified where possible)
* `RSN-MH` (B4–B5): multi-hop reasoning with evidence tracking
* `CODE-GEN-T2/T3`: typed languages + SQL/Bash (compile/execute verification)
* `CODE-DBG/CODE-OPT/CODE-TEST`: fix/minimize/refactor/test-gen with execution harness
* `LANG-*` (Hindi): comprehension/generation/translation/hinglish (if in curriculum scope)

### 9.2 Alignment buffers (SFT/RLHF)

* `ALN-INST`, `ALN-STRUCT`, `ALN-HALL`, `ALN-SAFE`, `ALN-HELP`

**Hard rule:** Do not mix alignment-style instruction-response samples into base pretraining synthetic by default.

---

## 10. Deliverables Checklist

### 10.1 Must-submit artifacts

* [ ] Locked diagnostic suite + version (`diag_suite_v1.1`)
* [ ] Shard catalog (this section) + sizes + linked tests + verifiers
* [ ] Stage-tagged synthetic shards (distilled + optional CoT)
* [ ] Verifier logs + acceptance stats per shard
* [ ] Dedup + adjacency reports
* [ ] Injection policy implementation notes (anti-spike + regression checks)
* [ ] Separate SFT/RLHF buffers packaged (not injected into pretraining)

### 10.2 Recommended directory structure

```
synthetic/
  shard_catalog.yaml
  generators/
  verifiers/
  filters/
  shards/
    S2/
    S3/
    S4/
  reports/
    baseline_diag.json
    shard_acceptance.json
    dedup_report.json
    injection_recommendations.json
```

---

## 11. Dependencies

* **Team 1:** dataset pool + metadata for dedup/mix constraints
* **Team 2:** frozen curriculum + rolling-window caps
* **Team 10:** early checkpoints for re-solve verification and baseline diagnostics

---

## 12. Failure Conditions (Reiterated)

* Preparing synthetic reactively during training
* No stable diagnostic suite defined in advance
* No deterministic verification (accepting fluent-but-wrong)
* Benchmark look-alike improvements only
* Curriculum violations / domain spikes / rolling-window dominance
* Synthetic rejected for leakage/noise due to weak filtering
```


