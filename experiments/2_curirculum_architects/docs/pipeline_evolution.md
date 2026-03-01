# T2 Curriculum Calculator: Progression from High-Compute to Low-Compute

**Team:** 2 — Curriculum Architects  
**Date:** 2026-02-21  
**Scope:** Evolution of the signal extraction and band assignment pipeline from V5 (regex-heavy, $15K) to v7.1 Fast EMR Serverless (regex-free, <$1k)

---

## 1. Problem Statement

Team 2's job is to assign every document in the training corpus a **curriculum band** (B0–B5) representing its difficulty level. These bands drive the progressive training schedule: early model checkpoints (1B params) see mostly B0–B1 content; later stages (70B) shift toward B3–B5.

The challenge: doing this at **4TB scale** on Spark with acceptable cost and runtime.

---

## 2. V5 Metrics Calculator — The Regex-Heavy Baseline

**File:** `glue_jobs/claude_reviewed/v1_t2_metrics_calculator_v5.py`  
**Runtime:** AWS Glue, G.2X workers × 20, FLEX execution  
**Cost:** ~$15,000 per 4TB run  
**Time:** 48–72 hours

### 2.1 Signal Extraction (20+ Regex Patterns on Full Text)

V5 ran `regexp_count()` and `regexp_extract_all()` against the **entire text** of every document. Patterns were grouped into 8 families:

| # | Pattern Family | Example Pattern | What It Detected |
|---|---------------|----------------|------------------|
| 1 | **Agentic Structural** | `(?:Step\s+\d+\|Task\s+\d+):\s*(?:Call\|Execute...)` | Tool-use traces, plan/action format |
| 2 | **Agentic Vocabulary** | `\b(?:execute\|invoke\|orchestrate\|delegate...)\b` | Agentic vocabulary density (ratio-based) |
| 3 | **CoT Explicit** | `Let's\s+think\s+(?:step-by-step\|carefully...)` | Explicit chain-of-thought prompts |
| 4 | **CoT Connectives** | `\b(?:therefore\|thus\|hence\|consequently...)\b` | Reasoning connective density |
| 5 | **Formal Reasoning** | `(?:Proof:\|Theorem:\|Q\.E\.D\.\|∎)` | Mathematical proof structure |
| 6 | **Code (multi-lang)** | `PYTHON_SYNTAX`, `JAVASCRIPT_SYNTAX`, `JAVA_CPP_SYNTAX`, indent structure, fences | Language-specific code syntax (5 sub-patterns) |
| 7 | **Math Content** | `EQUATION_PATTERN`, `LATEX_COMMANDS`, `MATH_TERMINOLOGY`, `MATH_SYMBOLS_PATTERN` | Equations, LaTeX commands, math Unicode (4 sub-patterns) |
| 8 | **Q&A / Tables** | `QA_PAIR_PATTERN`, `TABLE_ROW_PATTERN` | Q&A format detection, markdown tables |

Each pattern family produced **hit counts and density ratios**, which were combined into 5 modality scores (`agentic_score`, `cot_score`, `reasoning_score`, `code_score`, `math_score`). The scoring used **threshold-gated point accumulation** — e.g., for code:

```
code_score =
    (10 if code_tokenish_ratio >= 0.10 else 7 if >= 0.05 else 5 if kw_hits >= 6 else 0)
  + (3 if code_fence_hits >= 2 else 0)
  + (4 if python_hits >= 1 or js_hits >= 1 or java_hits >= 1 else 0)
  + (2 if syntax_chars >= 15 and indent_ratio >= 0.05 else 0)
```

### 2.2 Difficulty Score (7 Components)

V5's difficulty score blended 7 weighted components:

| Component | Weight | Source |
|-----------|--------|--------|
| Normalized length (tokens / 10K) | 15% | Basic stat |
| Structural density (code blocks + headings + tables / lines) | 20% | Regex counts |
| Reasoning difficulty (CoT + Reasoning + Agentic boolean flags) | 25% | Regex modality scores |
| Symbol density (Math/Code boolean flags) | 15% | Regex modality scores |
| Rarity proxy (matches against 130 high-value vocabulary words) | 15% | Broadcast keyword regex |
| Flesch reading ease (inverted, clamped to [0,1]) | 10% | Derived stat |
| **Metadata override** (difficulty/grade from JSON or text) | **70% blend** | JSON extraction + regex |

When metadata (difficulty labels like "hard", "expert", or grade levels) was present, it dominated the final score at 70% weight.

### 2.3 What Went Wrong

1. **Cost**: 20+ regex patterns × 4TB of text = enormous compute. Each `regexp_count()` call scans the entire text column per row.
2. **Runtime instability**: Complex regex patterns (especially with `.*?` and alternation) caused catastrophic backtracking on certain documents.
3. **Glue regex escaping**: Spark SQL's regex engine required double-escaping (`\\\\`) for patterns passed through `F.expr()`, leading to subtle bugs that silently produced zero-match columns.
4. **Removed metrics with no signal**: V5.0 already removed 12 metrics from earlier versions (V2–V4) that provided <2% correlation with quality. These included `html_tag_density`, `non_printable_ratio` (broke Indic scripts), `sentence_boundary_coherence` (flagged all code/poetry), `dependency_depth_estimate`, `punctuation_density`, and others.

---

## 3. v7.1 Fast EMR Serverless — The Regex-Free Production Version

**File:** `new_datasets/t2_fast_emr_serverless_no_stats.py`  
**Runtime:** EMR Serverless (Spark standalone)  
**Cost:** ~$100–200 per 4TB run  
**Time:** 8–16 hours  
**Cost reduction:** ~75–100×

### 3.1 Core Insight: Keywords Are Sufficient

The V5 regex patterns were doing two things:
1. **Detecting presence** of code/math/reasoning/agentic content
2. **Measuring density** (hits per word or per line)

Observation: simple `string.contains(keyword)` across a curated keyword list achieves nearly the same discriminative power for band assignment, because:
- Documents with `"def "`, `"import "`, `"class "` in them are code — no regex for Python syntax needed.
- Documents with `"theorem"`, `"proof"`, `"lemma"` are math — no LaTeX parsing needed.
- The band assignment uses **composite scores** (sums of many weak signals), so individual false positives wash out.

### 3.2 Signal Extraction (Keyword Contains + Translate)

All character-level metrics use `F.translate()` (a single-pass character deletion operation — O(n) with no backtracking):

| Metric | V5 Method | v7.1 Method |
|--------|-----------|-------------|
| Punctuation ratio | `regexp_replace` | `translate(text, ".,;:()[]{}!?", "")` |
| Digit ratio | `regexp_replace` | `translate(text, "0123456789", "")` |
| Special char ratio | `regexp_count('[;{}()\\[\\]]')` | `translate(text, "{}=&\|<>", "")` |
| Whitespace ratio | `regexp_replace('\\S', '')` | `translate(text, " \\t\\n\\r", "")` |
| Uppercase ratio | regex | `regexp_replace(text, '[A-Z]', '')` (sole surviving regex on content) |

All modality detection uses `F.lower(col).contains(keyword)`:

| Score | Keywords (count) | Examples |
|-------|------------------|---------|
| `code_keyword_count` | 20 | `"def "`, `"function "`, `"class "`, `"import "`, `"return "`, `"if ("`, `"while ("` |
| `math_keyword_count` | 11 | `"theorem"`, `"lemma"`, `"proof"`, `"integral"`, `"derivative"`, `"qed"` |
| `reasoning_keyword_count` | 10 | `"therefore"`, `"thus"`, `"hence"`, `"consequently"`, `"implies"` |
| `agentic_keyword_count` | 12 | `"execute"`, `"orchestrate"`, `"workflow"`, `"pipeline"`, `"tool use"` |
| `cot_keyword_count` | 8 | `"let's think"`, `"step by step"`, `"breaking down"`, `"analyzing"` |

### 3.3 Adaptive Sampling (Major Cost Saver)

v7.1 does NOT analyze the full text. It creates a `text_sample` capped by document size:

| Document Size | Sample Size |
|---------------|-------------|
| < 1K chars | Full text |
| 1K–10K chars | First 5,000 chars |
| 10K–50K chars | First 15,000 chars |
| > 50K chars | First 25,000 chars |

All keyword and character statistics run on `text_sample`. This is safe because keyword density in the first N characters is a reliable proxy for the full document — the document type doesn't change midway through (a code file stays code, a math paper stays math).

### 3.4 Progressive Column Dropping

To control memory on Spark executors, v7.1 drops intermediate columns after each pipeline stage:

```
create_adaptive_sample → keeps text + text_sample
compute_basic_stats    → keeps text (for noise detection)
compute_noise_metrics  → DROPS text
compute_character_stats → uses text_sample
compute_word_stats     → uses text_sample
compute_keyword_scores → DROPS text_sample
compute_composite_scores → drops raw keyword counts
compute_difficulty_score → drops component columns
```

---

## 4. Side-by-Side: What Changed, What Stayed

### Stayed the Same
- **Band definitions** (B0–B5 with centroids 0.05–0.90)
- **Probabilistic banding algorithm** (Gaussian weights → content nudges → normalize → lowest-credible pick)
- **Two-stage quality rejection** (Stage 1: physical corruption; Stage 2: noise/spam)
- **Output schema** (downstream T3 jobs unchanged)
- **Conservative "downgrade on uncertainty"** principle (ε = 0.15 threshold, pick lowest credible band)

### Changed

| Aspect | V5 | v7.1 Fast |
|--------|-----|-----------|
| **Signal extraction** | 20+ regex on full text | 61 keyword `contains()` on text_sample |
| **Character metrics** | `regexp_replace` | `translate()` |
| **Difficulty components** | 7 (with Flesch, metadata, rarity) | 4 (vocab, length, structure, specialty) |
| **Metadata override** | 70% weight when present | Removed |
| **Flesch readability** | Computed and used | Removed |
| **High-value vocabulary** | 130 broadcast keywords + regex | Removed (absorbed into keyword lists) |
| **Structural counts** | Code fences, headings, tables via regex | Removed (punctuation ratio as proxy) |
| **Modality score method** | Ratio-gated point accumulation (e.g., `code_tokenish_ratio >= 0.10 → 10 pts`) | Weighted sum of keyword count + char ratios (e.g., `special_ratio*30 + code_kw*5`) |
| **ArXiv cleaning** | Applied to arxiv sources (regex) | Present but disabled (text not used by T3) |
| **Infrastructure** | AWS Glue (managed, expensive) | EMR Serverless (Spark standalone, cheap) |
| **Cost / 4TB** | ~$15,000 | ~$150 |
| **Runtime / 4TB** | 48–72 hours | 8–16 hours |

---

## 5. Removed Metrics (with Rationale from V5.0 Changelog)

These metrics were removed during the V2→V5 progression because they had low signal-to-noise ratio or caused false positives:

| Metric | Reason Removed |
|--------|---------------|
| `non_printable_ratio` | Flagged valid Indic Unicode (Hindi, Tamil, etc.) as "non-printable" |
| `html_tag_density` | Regex counted everything between `<html>` and `</html>` — flagged 40% of code tutorials |
| `risky_tld_count` | Flagged security research papers discussing `.tk`/`.ml` domains |
| `sentence_boundary_coherence` | Assumed prose structure — rejected all code, poetry, and lists (50–70% false positive) |
| `punctuation_density` | 0.02 correlation with quality (random) |
| `dependency_depth_estimate` | Code legitimately has 100+ brackets — no spam signal |
| `num_numeric_tokens` | Inverse correlation with spam (code/science have more numbers) |
| `citation_count` | Never used in any rejection rule |
| `step_indicator_count` | Redundant with CoT detection |
| `ellipsis_count` | No signal |
| `dialogue_turn_count` | Redundant with conversation markers |
| `Stage 3 rejections` (all) | Too subjective, domain-variable — caused rejection of valid training data |

---

## 6. Validation

The v7.1 approach was validated by:
1. Running both V5 and v7.1 on the same source subsets and comparing band distributions
2. Verifying that the same quality rejection rules (Stage 1 & 2) produce equivalent pass-through rates (~95–98%)
3. Confirming that downstream T3 jobs required zero changes (identical output schema)

The key finding: **the probabilistic banding framework is robust to signal precision** — because the Gaussian weights already spread probability mass across adjacent bands, and the conservative lowest-credible assignment means a document landing B3 in V5 might land B2 or B3 in v7.1, but never B0 or B5. The relative ordering within bands is preserved.
