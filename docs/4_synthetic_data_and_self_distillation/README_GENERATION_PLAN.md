# Synthetic Data Generation Plan

**Team 4: Synthetic Data & Self-Distillation**  
**70B LLM — Gated by diagnostic tests, aligned to skill buckets**

---

## 1. Purpose & Principles

- **Pre-planned, not reactive.** All synthetic data is prepared in advance, gated by the 36 diagnostic tests defined in `diagnostic_tests.md`. Data may be partially or entirely unused; unprepared data is failure.
- **Targeted.** Each dataset targets specific skill buckets and failure modes identified by tests and by `skill_buckets.md`, not benchmark items.
- **Bands.** Synthetic data is **B3–B5 only**; B0–B2 use curriculum/coreset only.
- **Cap.** 5–10% of effective training mix per stage; adjustable only when measured weakness justifies it. Synthetic data must never dominate any rolling window.
- **Two-view policy.** Every sample has:
  - **Distilled view (primary):** final answer + short justification/outline; higher sampling weight.
  - **CoT view (secondary, gated):** explicit reasoning under `<think>` token, bounded length; never dominant.
- **CoT by band:** B3 → rare, capped, short; B4/B5 → allowed, capped.

---

## 2. Mapping: Diagnostic Tests → Skill Buckets → Data Gen

The 36 diagnostic tests (see `diagnostic_tests.md`) map to **8 testable skill areas** on base (non-chat) models. These map further to the full **38 skill buckets** in `skill_buckets.md` for generation scope.

| Diag. bucket | Tests       | Skill bucket(s) (from #116)        | Priority | Primary benchmarks |
|--------------|------------|------------------------------------|----------|--------------------|
| 1. Multilingual LM | TEST-029–031 (3) | FND-LEX-EN                         | MEDIUM   | HellaSwag          |
| 2. Long-Context    | TEST-032–034 (3) | FND-LCX                            | LOW      | LongBench          |
| 3. General Knowledge | TEST-025–028 (4) | FND-FACT                          | MEDIUM   | MMLU-style         |
| 4. Logical Reasoning | TEST-019–024 (6) | RSN-LOG, RSN-CAUS (light)        | HIGH     | BigBench-Logic     |
| 5. Mathematical    | TEST-001–008 (8) | RSN-ARITH, RSN-ALG, RSN-WPT       | **CRITICAL** | GSM8K           |
| 6. Code Understanding | TEST-015–018 (4) | CODE-COMP                        | HIGH     | MBPP subset        |
| 7. Code Generation  | TEST-009–014 (6) | CODE-SYN, CODE-GEN-T1             | **CRITICAL** | HumanEval      |
| 8. Structured Output | TEST-035–036 (2) | ALN-STRUCT (pre-train probe)     | LOW      | JSON/format        |

**Heavy emphasis:** Buckets 5 (Math) and 7 (Code Gen) — GSM8K and HumanEval are primary benchmark targets. Synthetic data for these must be ready first and verified against TEST-001–014.

---

## 3. Stage & Curriculum Alignment

Synthetic data is **stage-tagged** and respects Team 2’s frozen curriculum and caps.

| Stage | Params | Tokens | Bands in scope | Synthetic focus |
|-------|--------|--------|----------------|-----------------|
| Stage 1 | 1B   | 20B   | —              | No synthetic (B0–B2 only) |
| Stage 2 | 3B   | 40B   | —              | No synthetic (B0–B2 only) |
| Stage 3 | 8B   | 100B  | B3 (early), B4 (late) | Math (RSN-ARITH, RSN-WPT), Code (CODE-SYN, CODE-COMP, CODE-GEN-T1), Logic (RSN-LOG) |
| Stage 4 (MoE) | 16B/70B | 240B | B3–B5 | All 8 testable buckets + SFT/alignment buckets as below |

**Anti-spike:** Respect curriculum caps and rolling-window rules; synthetic data integrates with coresets, not as a separate spike.

---

## 4. Pre-Training Synthetic Data (B3–B5, 8 testable buckets)

### 4.1 Bucket 5 — Mathematical & Quantitative (CRITICAL)

**Driven by:** TEST-001–008 (multi-step arithmetic, linear equation, word problem, fraction, rate, percentage, age, two-hop).

**Skill buckets:** RSN-ARITH, RSN-ALG, RSN-WPT.

**Generate:**

- **Correct solutions:** Multi-step arithmetic, word problems, linear equations, fractions/percentages/rates; style randomization; no benchmark-adjacent phrasing.
- **Hard negatives:** Wrong operator (e.g. + vs −), wrong question (total vs difference), unit/rate confusion; labeled as incorrect with short justification.
- **Error-correction trajectories:** Wrong step → corrected step → final answer (distilled view); optional short CoT in `<think>` for B4/B5.
- **Per skill_buckets priorities:** Multi-digit drills, order-of-operations, decimal/fraction arithmetic, word problems with decomposition, distractor-heavy problems, step-by-step equation solving.

**Verification:** Re-solve with early SLM checkpoint (Team 10); accept only if teacher correct and student agreement (stage-appropriate). Reject fluent-but-wrong and benchmark look-alikes.

**Tags:** `stage`, `band` (B3/B4/B5), `skill_bucket` (RSN-ARITH, RSN-ALG, RSN-WPT). CoT only for B4/B5; B3 CoT rare and short.

---

### 4.2 Bucket 7 — Code Generation (CRITICAL)

**Driven by:** TEST-009–014 (loop trace, function completion, conditional, list/string ops, list indexing).

**Skill buckets:** CODE-SYN, CODE-GEN-T1 (Python/JS focus for pre-train).

**Generate:**

- **Correct solutions:** Loop traces, function completions, conditionals, list/string operations; continuation-style prompts (no instruction format).
- **Hard negatives:** Off-by-one, wrong branch (if/else), wrong method (e.g. `upper` vs `lower`); labeled incorrect + short rationale.
- **Error-correction:** Buggy snippet → minimal fix → correct output; distilled view primary.
- **Per skill_buckets:** Syntax/structure examples, function-from-spec, algorithm implementation, execution tracing.

**Verification:** Run code or use execution harness; teacher correctness + student re-solve. Reject benchmark-adjacent phrasing and style collapse.

**Tags:** `stage`, `band`, `skill_bucket` (CODE-SYN, CODE-GEN-T1). CoT gated by band.

---

### 4.3 Bucket 6 — Code Understanding (HIGH)

**Driven by:** TEST-015–018 (edge case, type error, logic inversion, off-by-one).

**Skill bucket:** CODE-COMP.

**Generate:**

- **Correct:** Edge-case identification, type-error detection, logic-inversion detection, range-boundary comments/completions.
- **Hard negatives:** Misidentifying error type, missing edge case; with short justification.
- **Per skill_buckets:** Code explanation pairs, execution tracing, bug localization hints.

**Tags:** `stage`, `band`, CODE-COMP. CoT only where band allows.

---

### 4.4 Bucket 4 — Logical & Deductive Reasoning (HIGH)

**Driven by:** TEST-019–024 (syllogism, contradiction, transitive, negation, temporal, spatial).

**Skill buckets:** RSN-LOG, light RSN-CAUS.

**Generate:**

- **Correct:** Syllogisms, contradiction detection, transitive chains, negation, temporal/spatial inference; continuation format.
- **Hard negatives:** Affirming consequent, denying antecedent, false entailment; labeled with rationale.
- **Per skill_buckets:** Explicit logical form, fallacy identification (negative examples), multi-step deduction, contradiction pairs.

**Tags:** `stage`, `band`, RSN-LOG (and RSN-CAUS where applicable).

---

### 4.5 Bucket 3 — General Knowledge (MEDIUM)

**Driven by:** TEST-025–028 (science, geography, history, biology).

**Skill bucket:** FND-FACT.

**Generate:**

- **Correct:** Factual completions (science, geography, history, biology); no instruction format.
- **Hard negatives:** Entity confusion, temporal anachronism; with correction.
- **Per skill_buckets:** Entity disambiguation, temporal ordering, fact verification (true/false).

**Tags:** `stage`, `band`, FND-FACT.

---

### 4.6 Bucket 1 — Multilingual LM / English (MEDIUM)

**Driven by:** TEST-029–031 (vocabulary, synonym, idiom).

**Skill bucket:** FND-LEX-EN.

**Generate:**

- **Correct:** Antonym/synonym/idiom completions; grammar-agreement completions.
- **Per skill_buckets:** Grammar correction pairs, sentence completion with agreement constraints.

**Tags:** `stage`, `band`, FND-LEX-EN.

---

### 4.7 Bucket 2 — Long-Context (LOW)

**Driven by:** TEST-032–034 (retention, multi-fact tracking, context consistency).

**Skill bucket:** FND-LCX.

**Generate:**

- **Correct:** Short passage with multiple facts → completion requiring retention; position-varied fact retrieval.
- **Per skill_buckets:** Position-varied retrieval, multi-document synthesis, long-range dependency resolution.

**Tags:** `stage`, `band`, FND-LCX.

---

### 4.8 Bucket 8 — Structured Output (LOW)

**Driven by:** TEST-035–036 (JSON completion, table reading).

**Skill bucket:** ALN-STRUCT (probe-relevant subset for base model).

**Generate:**

- **Correct:** JSON continuation, table-cell completion; valid structure only.
- **Per skill_buckets:** JSON with schemas, nested structures (continuation style).

**Tags:** `stage`, `band`, ALN-STRUCT.

---

## 5. SFT & Post-Training (Alignment) Synthetic Data

Team objective: **Do not forget SFT, alignment, and post-training datasets.**

These are not probed by the 36 base-model diagnostic tests (buckets 9–14 in diagnostic scope are instruction/MoE/alignment), but they are part of the same skill-bucket taxonomy and must be **prepared in advance** for downstream stages.

### 5.1 SFT-relevant buckets (from skill_buckets)

- **RSN-ADVMATH (B4–B5):** Calculus, probability, combinatorics, proofs — problem/solution pairs; CoT allowed, capped.
- **RSN-MH (B4–B5):** Multi-hop QA, evidence aggregation — chain demonstrations; CoT capped.
- **CODE-GEN-T2 / CODE-GEN-T3:** Java, C++, TS, Go; Rust, SQL, Bash — implementation pairs; verification via execution.
- **CODE-DBG, CODE-OPT, CODE-TEST:** Bug localization, minimal fix, optimization, test generation — per skill_buckets synthetic priorities.
- **LANG-* (Hindi):** Hindi comprehension/generation/translation/Hinglish/Hindi logic — only if SFT curriculum includes them; tag with LANG-* bucket IDs.
- **RSN-MATH-HI:** Hindi math (Devanagari numerals, terminology) — GSM8K-Hi alignment; prepare if Hindi math is in scope.

### 5.2 Alignment buckets (SFT/RLHF)

- **ALN-INST:** Multi-constraint instruction pairs, negative instruction examples.
- **ALN-STRUCT:** JSON/schema, complex nested structures (instruction + response format).
- **ALN-HALL:** Uncertainty expression, abstention on unknowns.
- **ALN-SAFE:** Harmful-request refusals, jailbreak resistance (no leakage into pre-train probes).
- **ALN-HELP:** High-quality response examples, length calibration.

**Rules:** Tag with phase `[SFT]` or `[RLHF]`, band, and skill_bucket. Keep separate from pre-train continuation data; no contamination of base-model diagnostic tests.

---

## 6. Generation Process

### 6.1 Sources

- **Local LMs** (preferred) for generation and verification.
- **Larger teacher models** (as permitted) for distillation and hard-negative mining.
- **Prompt templates** with style randomization to avoid teacher fingerprints and benchmark-adjacent phrasing.

### 6.2 Two-view creation

For each sample:

1. **Distilled view:** Final answer + short justification/outline (1–2 sentences). This is the primary training view; higher sampling weight.
2. **CoT view (optional, gated):** Same content with a `<think>`…`</think>` block containing bounded explicit reasoning; secondary weight, never dominant. Omit CoT for B3 except rare/short; allow CoT for B4/B5 within caps.

### 6.3 Verification & filtering

- **Teacher correctness:** All correct-solution and error-correction samples verified (automated or human) before inclusion.
- **Student re-solve:** Where possible, check early SLM (Team 10) agreement on a subset; reject if systematic disagreement.
- **Reject:** Fluent-but-incorrect reasoning, teacher stylistic fingerprints, benchmark-adjacent phrasing.
- **Deduplicate:** Against training pool (Team 1) and eval sets (approximate if exact unavailable).
- **Similarity check:** Avoid near-duplicates within synthetic set and to known benchmarks.

### 6.4 Volume & caps

- **Per-stage synthetic share:** 5–10% of effective mix; start at lower end, increase only if diagnostic tests show weakness.
- **Per-bucket:** Allocate more to Math and Code Gen; ensure no single bucket dominates the synthetic portion.
- **Rolling window:** Synthetic data must not dominate any rolling window of the curriculum.

---

## 7. Gating & Readiness

### 7.1 Before training

- **36 diagnostic tests** (see `diagnostic_tests.md`) defined and harness ready.
- **Baseline run** on early B3 checkpoint (Team 10): record pass rates per bucket, identify weak buckets (< 50% or below threshold).
- **Synthetic datasets** pre-generated for all 8 testable buckets (and SFT/alignment buckets as above), tagged and filtered.
- **Buffers:** Ready-to-inject shards per stage; 5–10% cap documented.

### 7.2 Injection decision

- Use diagnostic results and curriculum plan to decide **whether** and **how much** synthetic data to inject per stage/bucket.
- Optional: use early checkpoint pass rates to prioritize which buckets get synthetic data first (e.g. Math and Code if TEST-001–014 are weak).

### 7.3 After injection (per stage / checkpoint)

- **Re-run** the same 36 tests on post-injection checkpoint.
- **Measure:** Delta per bucket, regression in non-targeted buckets.
- **Success (typical):** +15% pass rate in targeted buckets; < −5% regression elsewhere; correlation with GSM8K/HumanEval gains where applicable.
- **Impact report:** weakness → synthetic data used → before/after test deltas; document absence of regressions.

---

## 8. Deliverables Checklist

- [ ] **Skill buckets** — Defined (from `skill_buckets.md`).
- [ ] **25–50 diagnostic tests** — Defined and documented (36 in `diagnostic_tests.md`).
- [ ] **Synthetic dataset shards** — Stage-tagged, band-tagged, skill-bucket-tagged; B3–B5 only; two-view where CoT applies.
- [ ] **Filtering and verification logic** — Scripts/criteria for correctness, re-solve, dedup, similarity.
- [ ] **Impact reports** — Before/after test deltas, no regressions; mapping weakness → data → improvement.
- [ ] **SFT/alignment buffers** — Prepared for ALN-*, RSN-ADVMATH, RSN-MH, CODE-DBG/OPT/TEST, LANG-*, etc., as needed by curriculum.

---

## 9. Dependencies

- **Team 1:** Dataset pool and metadata (for dedup and mix caps).
- **Team 2:** Frozen curriculum structure and caps (for stage/band alignment and anti-spike).
- **Team 10:** Early checkpoints (for baseline diagnostic run and student re-solve verification).

---

## 10. Failure conditions to avoid

- Preparing synthetic data **reactively** during training.
- **No tests** defined in advance.
- Improvements only on **benchmark look-alikes** (no skill-bucket coverage).
- **General loss** or **non-target capability** regressions.
- Synthetic data **rejected** for noise, leakage, or benchmark contamination.

---

