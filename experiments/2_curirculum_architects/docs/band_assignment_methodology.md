# T2 Final Band Assignment Methodology

**Team:** 2 — Curriculum Architects  
**Date:** 2026-02-21  
**Status:** Production — all three scripts running on EMR Serverless  
**Scope:** How every document in the corpus gets assigned a curriculum band (B0–B5)

---

## 1. Overview

Three independent EMR Serverless jobs cover the entire corpus. Each produces the **same output schema** so downstream T3 jobs see a unified interface.

| Job | Script | Covers | Band Range |
|-----|--------|--------|------------|
| **Fast EMR Serverless** | `t2_fast_emr_serverless_no_stats.py` | All large-scale web/book/code sources (RedPajama, FineWeb, Dolma, Sangraha, arXiv, etc.) | B0–B5 (full range) |
| **Curated Datasets** | `t2_curated_datasets_curriculum.py` | HuggingFace curated instruction/preference/math/code datasets | Source-clamped (per dataset) |
| **ERAv4 Student Data** | `t2_erav4_data_curriculum.py` | Student-generated Q&A drills + Samvaad conversation | B0–B2 (tight range) |

All three share the same **probabilistic banding framework**. They differ only in signal extraction (what features are measured) and post-banding constraints (source-aware clamping).

---

## 2. Band Definitions

| Band | Name | Centroid | Intent | Example Content |
|------|------|----------|--------|-----------------|
| **B0** | Nursery | 0.05 | Surface language acquisition | Simple web text, basic sentences, spelling drills |
| **B1** | Primary | 0.20 | Fluent everyday language | Clean prose, news articles, simple Q&A |
| **B2** | High School | 0.35 | Structured knowledge | Wikipedia-style, intro tutorials, structured conversations |
| **B3** | Undergraduate | 0.55 | Reasoning emergence | Technical docs, meaningful code, multi-step problems |
| **B4** | Graduate | 0.75 | Explicit abstraction | Math proofs, algorithms, research papers |
| **B5** | PhD | 0.90 | Agentic / system-level reasoning | Tool-use traces, advanced math, planning workflows |

---

## 3. Pipeline Architecture (All Three Jobs)

```
Input (Parquet from T1)
  │
  ├─ Adaptive Sampling ──── text_sample = first N chars (saves compute)
  │
  ├─ Basic Stats ────────── byte_length, word_count, line_count, token_count_estimate
  │
  ├─ Noise Metrics ──────── whitespace_ratio, url_count, boilerplate_count
  │
  ├─ Character Stats ────── punct_ratio, digit_ratio, special_ratio, upper_ratio
  │     (on text_sample)      via F.translate() — no regex
  │
  ├─ Word Stats ─────────── unique_token_ratio, avg_word_length, compression_ratio
  │
  ├─ Keyword Scores ─────── code/math/reasoning/agentic/cot keyword hit counts
  │     (on text_sample)      via F.contains() — no regex
  │
  ├─ Composite Scores ───── code_score, math_score, reasoning_score, agentic_score, cot_score
  │
  ├─ Difficulty Score ───── Single float in [0, 1]
  │
  ├─ Quality Filters ────── Stage 1 (physical) + Stage 2 (noise) rejection
  │
  ├─ Probabilistic Banding ── Gaussian weights → nudges → normalize → pick band
  │
  ├─ Source Clamping ─────── (curated + erav4 only) clamp to [floor, ceiling]
  │
  └─ Output ─────────────── bands/ (partitioned by band) + rejections/
```

---

## 4. Signal Extraction (Fast EMR Serverless — Main Job)

### 4.1 Character-Level Signals (via `F.translate()`)

`translate(text, chars_to_remove, "")` does a single O(n) pass — no regex engine, no backtracking.

| Signal | Characters Removed | What It Measures |
|--------|--------------------|------------------|
| `punct_ratio` | `.,;:()[]{}!?` | Structural marking density |
| `digit_ratio` | `0123456789` | Numeric content (code/math indicator) |
| `special_ratio` | `{}=&\|<>` | Code syntax characters |
| `upper_ratio` | `[A-Z]` (sole regex on content) | Acronyms, constants, class names |

### 4.2 Word-Level Signals (via `F.split()` + `F.array_distinct()`)

| Signal | Formula | What It Measures |
|--------|---------|------------------|
| `unique_token_ratio` | unique_words / word_count | Vocabulary diversity (low = repetitive) |
| `avg_word_length` | total_chars_in_words / word_count | Lexical complexity |
| `compression_ratio` | byte_length / char_length | Information density proxy |

### 4.3 Keyword Signals (via `F.lower(col).contains(keyword)`)

Each keyword list produces a **hit count** (how many of the keywords appear at least once in the text_sample).

| Signal | # Keywords | Band Affinity | Example Keywords |
|--------|------------|---------------|------------------|
| `code_keyword_count` | 20 | B1–B4 | `def `, `function `, `class `, `import `, `return `, `if (`, `while (`, `malloc`, `iostream` |
| `math_keyword_count` | 11 | B3–B5 | `theorem`, `lemma`, `proof`, `corollary`, `equation`, `integral`, `derivative`, `qed` |
| `reasoning_keyword_count` | 10 | B2–B4 | `therefore`, `thus`, `hence`, `consequently`, `because`, `implies`, `we conclude` |
| `agentic_keyword_count` | 12 | B5 | `execute`, `invoke`, `orchestrate`, `delegate`, `workflow`, `pipeline`, `tool use`, `agent` |
| `cot_keyword_count` | 8 | B3–B5 | `let's think`, `step by step`, `first`, `finally`, `breaking down`, `analyzing` |

---

## 5. Composite Scores

Keyword counts and character ratios are combined into **5 integer scores** that represent modality presence and intensity:

```
code_score      = special_ratio × 30  +  digit_ratio × 20  +  code_keyword_count × 5
                  + (5 if avg_word_length > 8)

math_score      = math_keyword_count × 6  +  digit_ratio × 15  +  special_ratio × 8

reasoning_score = reasoning_keyword_count × 6  +  upper_ratio × 8

agentic_score   = agentic_keyword_count × 6

cot_score       = cot_keyword_count × 5  +  reasoning_keyword_count × 2
```

**Boolean flags** (used for output, not banding):

| Flag | Threshold |
|------|-----------|
| `has_code` | code_score ≥ 10 |
| `has_math` | math_score ≥ 8 |
| `has_reasoning` | reasoning_score ≥ 6 |
| `has_agentic` | agentic_score ≥ 8 |
| `has_cot` | cot_score ≥ 10 |

---

## 6. Difficulty Score

A single continuous value in [0, 1] that maps to band centroids.

### 6.1 Fast EMR Serverless (main job)

| Component | Formula | Weight | Intuition |
|-----------|---------|--------|-----------|
| **Vocabulary** | min(unique_token_ratio × 2.5, 1.0) | 25% | High diversity = harder |
| **Length** | min((avg_word_length − 4) / 6, 1.0) | 25% | Longer words = harder |
| **Structure** | min(punct_ratio × 3, 1.0) | 20% | More punctuation = more structured |
| **Specialty** | min((code_score + math_score + reasoning_score) / 60, 1.0) | 30% | Technical content = harder |

```
difficulty_score = 0.25 × Vocabulary + 0.25 × Length + 0.20 × Structure + 0.30 × Specialty
```

### 6.2 Curated Datasets (same structure, adjusted weights)

The curated job reweights because these datasets are denser with signals:

| Component | Weight | Change |
|-----------|--------|--------|
| Vocabulary | 20% | ↓ from 25% |
| Length | 20% | ↓ from 25% |
| Structure | 20% | Same |
| **Specialty** | **40%** | **↑ from 30%** |

Specialty also includes `latex_score` and `science_score` (divides by 80 instead of 60).

### 6.3 ERAv4 Student Data (completely different components)

Student-generated drills are short, repetitive Q&A — the main job's signals wouldn't differentiate them. Custom components:

| Component | Weight | Formula | Intuition |
|-----------|--------|---------|-----------|
| **Vocabulary** | 30% | unique_token_ratio (capped at 1.0) | Repetitive drills → low diversity → low difficulty |
| **Length** | 20% | char_length / 1000 (capped at 1.0) | Longer = more complex problems |
| **Q&A density** | 20% | (1 − min(qa_density, 1.0)) × 0.5 | Pure drills have high Q? density → low difficulty |
| **Language** | 10% | Indic = 0.15, English = 0.05 | Indic script adds processing complexity |
| **Conversation** | 20% | conv_marker_count / 5 (capped at 0.25) | Samvaad free conversation → harder than drills |

---

## 7. Probabilistic Banding Algorithm

This is the core band assignment mechanism, **identical across all three jobs**.

### Step 1: Compute Triangular Base Weights

For each band b ∈ {B0, B1, B2, B3, B4, B5}, compute a weight that peaks at the band's centroid and falls linearly to zero at distance WIDTH = 0.20:

```
w_b = max(0, 1 − |difficulty_score − center_b| / 0.20)
```

| Band | Center | Weight = 1.0 when score = | Weight = 0 when score deviates by ≥ |
|------|--------|---------------------------|--------------------------------------|
| B0 | 0.05 | 0.05 | 0.20 |
| B1 | 0.20 | 0.20 | 0.20 |
| B2 | 0.35 | 0.35 | 0.20 |
| B3 | 0.55 | 0.55 | 0.20 |
| B4 | 0.75 | 0.75 | 0.20 |
| B5 | 0.90 | 0.90 | 0.20 |

Example: A document with difficulty_score = 0.40 gets:
- w_B0 = max(0, 1 − |0.40 − 0.05| / 0.20) = max(0, 1 − 1.75) = **0.0**
- w_B1 = max(0, 1 − |0.40 − 0.20| / 0.20) = max(0, 1 − 1.00) = **0.0**
- w_B2 = max(0, 1 − |0.40 − 0.35| / 0.20) = max(0, 1 − 0.25) = **0.75**
- w_B3 = max(0, 1 − |0.40 − 0.55| / 0.20) = max(0, 1 − 0.75) = **0.25**
- w_B4 = 0.0, w_B5 = 0.0

### Step 2: Apply Content-Based Nudges

Composite scores **add weight** to specific bands, allowing content type to shift a document upward even if its raw difficulty score is moderate:

| Condition | Band Nudged | Amount | Rationale |
|-----------|-------------|--------|-----------|
| code_score 1–14 | B1 | +0.05 | Trivial code → at least Primary |
| code_score 15–24 | B2 | +0.08 | Intro technical → High School |
| reasoning_score ≥ 5 | B2 | +0.05 | Implicit reasoning → structured |
| code_score ≥ 25 | B3 | +0.10 | Meaningful code → Undergraduate |
| reasoning_score ≥ 8 | B3 | +0.08 | Multi-step reasoning |
| cot_score ≥ 10 | B3 | +0.05 | Chain-of-thought present |
| math_score ≥ 12 | B4 | +0.15 | Formal math → Graduate |
| code_score ≥ 40 | B4 | +0.12 | Hard code → Graduate |
| reasoning_score ≥ 12 | B4 | +0.10 | Strong reasoning |
| **agentic_score ≥ 8** | **B5** | **+0.20** | **Agentic traces → PhD** |
| math_score ≥ 20 | B5 | +0.12 | Advanced math |
| reasoning ≥ 15 AND code ≥ 30 | B5 | +0.10 | Combined technical depth |

**Curated datasets add extra nudges** for their additional signals:

| Condition | Band Nudged | Amount | Signal |
|-----------|-------------|--------|--------|
| conv_score ≥ 4 | B2 | +0.06 | Instruction format detected |
| conv_score ≥ 8 | B3 | +0.06 | Rich multi-turn conversation |
| step_score ≥ 5 | B3 | +0.08 | Structured step-by-step |
| step_score ≥ 12 | B4 | +0.08 | Deep structured reasoning |
| latex_score ≥ 6 | B4 | +0.12 | Formal math notation |
| latex_score ≥ 14 | B5 | +0.10 | Heavy LaTeX |
| science_score ≥ 8 | B3 | +0.06 | Academic vocabulary |

**ERAv4 student data applies NO nudges** — code/math/reasoning/agentic scores are hardcoded to 0 because this data contains none of those modalities.

### Step 3: Normalize to Probabilities

```
total = w_B0 + w_B1 + w_B2 + w_B3 + w_B4 + w_B5
band_p_Bi = w_Bi / total    (for each band)
```

This produces a full probability distribution over bands. These probabilities are **written to the output** (`band_p_B0` through `band_p_B5`) for downstream analysis.

### Step 4: Assign Final Band (Lowest Credible)

The assigned band is the **lowest band whose probability exceeds ε = 0.15**:

```
IF   band_p_B0 ≥ 0.15 → assigned_band = B0
ELIF band_p_B1 ≥ 0.15 → assigned_band = B1
ELIF band_p_B2 ≥ 0.15 → assigned_band = B2
ELIF band_p_B3 ≥ 0.15 → assigned_band = B3
ELIF band_p_B4 ≥ 0.15 → assigned_band = B4
ELSE                   → assigned_band = B5
```

**Why lowest?** This implements the curriculum design principle of **"downgrade on uncertainty"** — when a document has non-trivial probability in adjacent bands, it's placed in the lower band. This prevents the model from training on content beyond its current capacity at each stage.

---

## 8. Source-Aware Band Clamping (Curated + ERAv4 Only)

After probabilistic banding produces a raw `band`, the curated and ERAv4 jobs apply a **hard clamp** based on the dataset source:

```
assigned_band = BANDS[max(floor, min(ceil, raw_band_index))]
```

The raw `band` column is preserved unchanged so the full probability distribution remains interpretable.

### Curated Dataset Clamp Ranges

| Source | Floor | Ceiling | Rationale |
|--------|-------|---------|-----------|
| samvaad_hi | B0 | B2 | Everyday Hindi conversation |
| smoltalk | B1 | B3 | General SFT chit-chat |
| perfectblend | B2 | B4 | Mixed quality SFT |
| orpo_dpo | B2 | B4 | Preference/alignment mix |
| ultrafeedback | B2 | B4 | Binarised preferences |
| infinity_prefer | B2 | B4 | Early preference data |
| lmarena / arena | B2 | B4 | Arena human preference responses |
| helpsteer | B3 | B5 | Multi-attribute instruction following |
| nemotron_post | B3 | B5 | Diverse post-training |
| megascience | B3 | B5 | Multi-domain science text |
| ling_coder | B3 | B5 | Coding SFT instruction |
| gsm8k | B3 | B4 | Grade school math (capped — not PhD) |
| nemotron_math | B4 | B5 | Expert math |
| ultradata | B4 | B5 | Advanced math (L3 spec) |
| skywork | B4 | B5 | Scientific/math reward |
| hardgen | B4 | B5 | Hard generation tasks |
| teichai / high_reasoning | B4 | B5 | High-reasoning traces |

### ERAv4 Student Data Clamp Ranges

| Source | Floor | Ceiling | Domain Override |
|--------|-------|---------|-----------------|
| erav4_lang | B0 | B1 | language_literacy |
| erav4_math | B0 | B2 | education |
| erav4_pattern | B0 | B2 | education |
| samvaad | B0 | B2 | conversation |

**Why clamp?** These datasets have known difficulty ranges. A short GSM8K problem might look "easy" to the generic difficulty formula and score B1, but GSM8K problems require multi-step arithmetic reasoning — they should never be below B3. Clamping preserves the **relative ordering within a dataset** (easy vs. hard GSM8K problems still separate into B3 vs. B4) while preventing obviously wrong assignments.

---

## 9. Quality Rejection (Before Banding)

Documents are rejected **before** band assignment. Rejected documents are written to a separate `rejections/` output.

### Stage 1: Physical Corruption

| Rule | Condition | Rationale |
|------|-----------|-----------|
| Too short (bytes) | byte_length < 50 | Corrupted/empty |
| Too short (chars) | char_length < 20 | Corrupted/empty |
| Too few tokens | token_count_estimate < 10 | Not enough content to learn from |

### Stage 2: Noise & Spam

| Rule | Condition | Rationale |
|------|-----------|-----------|
| Repetitive template | unique_token_ratio < 0.01 AND word_count > 200 | Same word repeated 200+ times |
| Excessive whitespace | whitespace_ratio > 0.95 | Mostly blank |
| Link spam | url_ratio > 0.7 AND url_count > 50 | Link farm |
| Boilerplate spam | boilerplate_ratio > 0.50 | Cookie policy / terms of service |
| Thread fragment | thread_marker_count > 5 AND token_count < 200 | Orphaned forum reply |

**ERAv4 student data has relaxed filters** — only rejects char_length < 5 or word_count < 1, since the data is pre-curated.

**Target pass-through rate:** 95–98% of documents pass both stages.

---

## 10. Output Schema

All three jobs write the same columns:

| Column Group | Columns |
|-------------|---------|
| **Identity** | `uuid`, `id`, `file_path`, `source`, `domain`, `hash`, `language`, `metadata` |
| **Band** | `assigned_band`, `band`, `difficulty_score`, `band_p_B0`–`band_p_B5` |
| **Modality flags** | `has_code`, `has_cot`, `has_reasoning`, `has_agentic` |
| **Scores** | `agentic_score`, `cot_score`, `reasoning_score`, `code_score`, `math_score` |
| **Size** | `byte_length`, `word_count`, `unique_token_ratio`, `compression_ratio`, `token_count_estimate`, `fertility_estimate` |
| **Rejection** (rejected only) | `is_rejected`, `rejection_reason`, `rejection_level` |

Output is partitioned by `band` as Parquet with zstd compression.

---

## 11. Worked Example

**Document:** A 2,000-word Python tutorial explaining list comprehensions with 3 code blocks.

**Step 1 — Signals extracted:**
- `code_keyword_count` = 9 (hits: `def`, `function`, `class`, `import`, `return`, `for (`, `if (`, `let`, `var`)
- `math_keyword_count` = 0
- `reasoning_keyword_count` = 2 (hits: `because`, `therefore`)
- `agentic_keyword_count` = 0
- `special_ratio` = 0.03, `digit_ratio` = 0.01, `avg_word_length` = 5.2

**Step 2 — Composite scores:**
- `code_score` = 0.03×30 + 0.01×20 + 9×5 + 0 = 0.9 + 0.2 + 45 = **46**
- `reasoning_score` = 2×6 + upper_ratio×8 ≈ **13**
- `math_score`, `agentic_score`, `cot_score` ≈ 0

**Step 3 — Difficulty score:**
- vocab = min(0.65×2.5, 1) = 1.0 → × 0.25 = 0.25
- length = min((5.2−4)/6, 1) = 0.20 → × 0.25 = 0.05
- structure = min(0.04×3, 1) = 0.12 → × 0.20 = 0.024
- specialty = min((46+0+13)/60, 1) = 0.98 → × 0.30 = 0.294
- **difficulty_score = 0.618**

**Step 4 — Banding:**
- Base weights: w_B3 = max(0, 1−|0.618−0.55|/0.20) = 0.66, w_B4 = max(0, 1−|0.618−0.75|/0.20) = 0.34
- Nudges: code_score ≥ 25 → B3 += 0.10 → w_B3 = 0.76; code_score ≥ 40 → B4 += 0.12 → w_B4 = 0.46; reasoning ≥ 12 → B4 += 0.10 → w_B4 = 0.56
- Normalize: total = 0.76 + 0.56 = 1.32 → p_B3 = 0.576, p_B4 = 0.424
- Lowest credible: p_B3 = 0.576 ≥ 0.15 → **assigned_band = B3**

This is correct — a Python tutorial with code examples and some reasoning is Undergraduate-level content (B3).

---

## 12. Summary

The band assignment methodology is:

1. **Extract cheap signals** — keyword `contains()` and `translate()` character counts on a capped text sample
2. **Combine into composite scores** — weighted sums that capture modality (code, math, reasoning, agentic, CoT)
3. **Compute difficulty** — a 4-component weighted average mapping to [0, 1]
4. **Probabilistic banding** — Gaussian weights at band centroids, nudged by composite scores, normalized to probabilities
5. **Conservative assignment** — pick the lowest band with ≥ 15% probability
6. **Source clamping** (curated/student data only) — enforce known dataset difficulty bounds

The system processes 4TB for ~$150 in 8–16 hours, producing deterministic, reproducible band assignments for downstream curriculum-driven pretraining.
