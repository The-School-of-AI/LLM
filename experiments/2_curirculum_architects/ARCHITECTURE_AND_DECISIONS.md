# Team 2: Curriculum Architects — Architecture and Design Decisions

**Team:** 2 — Curriculum Architects
**Scope:** Curriculum policy design and difficulty-band assignment for the LLM pretraining corpus
**Status:** Production (EMR Serverless, 3 jobs)
**Last Updated:** 2026-02-21

---

## 1. Mandate and Problem Statement

### What Team 2 Owns

From Team's Charter:

> Design and lock a curriculum-driven data ordering and mixing strategy that maximizes learning efficiency, stability, and capability emergence across the 1B → 3B → 8B → 70B growth schedule.

Team 2 does not discover datasets (that is Team 1) and does not train the model. The mandate is to answer: **what should the model see, when, and in what proportions**.

The core thesis, also from the charter, is blunt:

> Curriculum errors do not fail loudly. They silently waste trillions of tokens. This team exists to prevent that.

The practical consequence is that every document in the ~4TB training corpus needs a **difficulty band label** (B0–B5). These band labels drive the sampling schedule used at each training stage. Getting the labels wrong does not crash training — it just makes the model worse, which is discovered weeks later.

### Two Deliverables

Team 2 was responsible for two concrete outputs:

1. **`curriculum.yaml`** — the canonical policy document that defines what the model's diet looks like at each growth stage. This is the contract between Team 2 and the training team. Structure is frozen; only weights change across stages.

2. **Band assignment at scale** — every document in the corpus gets a `assigned_band` (B0–B5). This required building a distributed Spark pipeline that runs on ~4TB of Parquet data within a constrained AWS budget.

---

## 2. The Band System (B0–B5)

Six bands model the model's growing cognitive capacity from 1B to 70B parameters. The bands are fixed — they don't change across training stages. What changes is their **sampling proportion**.

| Band | Name | Difficulty Centroid | What It Contains | CoT Policy | Agentic |
|------|------|---------------------|-----------------|------------|---------|
| **B0** | Nursery | 0.05 | Simple web text, basic sentences, language drills | Forbidden | Forbidden |
| **B1** | Primary | 0.20 | Clean prose, news, simple Q&A, everyday conversation | Forbidden | Forbidden |
| **B2** | High School | 0.35 | Wikipedia-style, intro tutorials, structured dialogue | Forbidden | Forbidden |
| **B3** | Undergraduate | 0.55 | Technical docs, meaningful code, multi-step problems | Allowed (max 64 tokens) | Forbidden |
| **B4** | Graduate | 0.75 | Math proofs, algorithms, research papers | Allowed (max 128 tokens) | Toy only |
| **B5** | PhD | 0.90 | Tool-use traces, advanced math, planning workflows | Allowed (max 256 tokens) | Allowed (max 2% share) |

**Why these thresholds?** The centroids (0.05, 0.20, 0.35, 0.55, 0.75, 0.90) were chosen to give roughly equal spacing in difficulty space with a 0.20 triangle width so adjacent bands overlap. This prevents hard edges in the training distribution. The CoT caps come from the intuition that early-stage models cannot absorb reasoning traces of arbitrary length — the cap grows with model capacity.

### The Growth Schedule

The same B0–B5 documents are used at every stage. What changes is how often each band is sampled:

| Stage | B0 | B1 | B2 | B3 | B4 | B5 | General | Code | CoT | Agentic |
|-------|----|----|----|----|----|----|---------|------|-----|---------|
| 1B (base) | 49% | 13% | 17% | 13% | 6% | 2% | 86% | 12% | 2% | 0% |
| 3B (harder_shift_1) | 43% | 15% | 19% | 15% | 6% | 2% | 78% | 18% | 3% | 1% |
| 8B (harder_shift_2) | 27% | 19% | 24% | 19% | 8% | 3% | 68% | 24% | 5% | 3% |
| 70B (final_adaptive) | 16% | 22% | 28% | 22% | 9% | 3% | 60% | 28% | 7% | 5% |

The pattern is clear: B0 drops from 49% to 16% as the model scales. The early model needs foundational language; the later model needs reasoning and code. This is not arbitrary — it mirrors how humans learn, and it prevents the early model from trying to absorb content it cannot yet process.

### Domain-Band Policy

Domains are inherited from Team 1's tags. B0–B5 restrict which domains are allowed:

```yaml
B0: [web, social, qa, education, language_literacy, conversation]
B1: [web, encyclopedia, news, social, qa, education, conversation]
B2: [encyclopedia, news, education, literature, web, qa, conversation]
B3: [science, math, education, code, literature, conversation]
B4: [science, math, code, instruction]
B5: [instruction, science, math, code]
```

The domain restrictions encode a principle: early bands should not contain domain-specific jargon from math or science, because the model does not have the prerequisite vocabulary. Science and math are B3+ by design.

---

## 3. Pipeline Evolution: From Tags to Production

### Phase 1: Curriculum Tags (Initial Implementation)

**Location:** `curriculum_tags/`
**Status:** Legacy, kept for reference

The first approach was a Python-based, plugin-driven extractor. The core design was sound: a `ReadOnlyRecord` wrapper enforced immutability (source records could never be mutated), and metrics were organized into levels (L0 → L1 → L2) so cheap filters ran first and expensive ones only ran on records that passed.

**What worked:**
- The immutability-first design was adopted in all subsequent versions.
- The level-based early-rejection pattern reduced wasted compute on junk records.
- 8+ metric plugins (difficulty, modality, domain, readability, structural_density, diversity) gave a clean interface.

**What went wrong:**
- Regex-heavy pattern matching for code, math, and agentic signals was brittle at scale.
- Running on Python (not Spark) meant it couldn't process terabytes efficiently.
- The band assignment logic lacked probabilistic reasoning — documents were assigned bands deterministically based on hard thresholds, which made adjacent-band decisions fragile.
- **Critical gap:** No level-wise rejection checking with documented thresholds. Patterns were overly broad (e.g., `AGENTIC_PATTERN` matched newspaper headlines with words like "Action" and "Thought").

**Key learning:** The plugin architecture was the right abstraction. The signal extraction approach needed to move from regex to statistical proxies.

---

### Phase 2: Curriculum Extractor (Reference Implementation)

**Location:** `src/curriculum_extractor/`
**Status:** Production reference, not the Spark job

This was a clean-room rewrite of the tag system, producing a modular library. The `RecordExtractor` class orchestrated 8 metric plugins via a frozen dataclass interface. The library is used for single-record analysis and testing.

Key improvements over Phase 1:
- All plugin outputs are frozen dataclasses (no mutation possible downstream).
- The L0 → L1 → L2 execution model was hardened: L0 = language + length filters, L1 = quality gates, L2 = content classification.
- Band assignment moved from hard thresholds to probability distributions (a preview of the probabilistic framework adopted in PatternRefinement r5.0).

This library is not the production Spark job — it was the design laboratory where the probabilistic banding approach was worked out. The final Spark scripts implement the same logic, rewritten as native Spark operations.

---

### Phase 3: ProbabilisticBanding r4.0 — The First Full-Scale Spark Job

**Location:** `pipeline/jobs/main_job.py` (r4.0 commit — see `git log pipeline/jobs/main_job.py`)
**Infrastructure:** AWS Glue, G.2X workers × 20, FLEX execution

This was the first full-scale Spark implementation and introduced the **probabilistic banding framework** that survives into production unchanged:
- A triangular weight function peaks at each band's centroid and falls to zero at distance ±0.20.
- Content-based nudges shift band weights based on detected signals (code pushes toward B3, agentic toward B5).
- Weights are normalized to probabilities; the final assignment is the **lowest band whose probability exceeds 15%**.

This "downgrade on uncertainty" principle is deliberate: when a document sits between two bands, assign the lower one. A model trained on content slightly below its current capacity is safer than one overwhelmed by content it cannot absorb.

**What went wrong:** Several regex patterns caused catastrophic backtracking on specific document shapes, stalling executors. The run was unstable and did not complete on the full corpus. This failure directly motivated the metric audit in r5.0.

---

### Phase 4: PatternRefinement r5.0 — Glue Baseline

**Location:** `pipeline/jobs/main_job.py` (r5.0 commit — see `git log pipeline/jobs/main_job.py`)
**Infrastructure:** AWS Glue, G.2X workers × 20, FLEX execution

PatternRefinement r5.0 was the first stable full-scale run. Its primary contribution was a rigorous audit that removed 12 metrics and Stage 3 rejections:

#### Metrics Removed in r5.0

| Metric Removed | Why |
|---------------|-----|
| `non_printable_ratio` | Blocked all Indic Unicode (Hindi, Tamil, etc.) |
| `html_tag_density` | Counted chars between `<html>` and `</html>` — 40% false positive on code tutorials |
| `risky_tld_count` | Flagged security research papers discussing `.tk`/`.ml` domains |
| `sentence_boundary_coherence` | Assumed prose — rejected code, poetry, lists (50–70% false positive) |
| `punctuation_density` | 0.02 correlation with quality (effectively random) |
| `dependency_depth_estimate` | Code legitimately has 100+ brackets — no spam signal |
| `num_numeric_tokens` | **Inverse** correlation with spam (code/science have more numbers) |
| `dialogue_turn_count`, `ellipsis_count`, `citation_count`, `step_indicator_count`, `list_marker_count` | Zero usage in any rejection or band rule |

**Stage 3 rejections were removed entirely** in r5.0. The previous Stage 3 filtered ~5% of data, but manual review showed a 60%+ false positive rate — valid training data was being discarded because readability thresholds designed for prose flagged code, structured content, and non-English scripts.

#### What Went Wrong with PatternRefinement r5.0

1. **Regex scaling:** 20+ `regexp_count()` calls scanning the full text of every row in a 4TB dataset is expensive. The Glue G.2X cluster was CPU-bound most of the run.
2. **Catastrophic backtracking:** Several patterns with `.*?` and alternation caused exponential regex engine backtracking on specific document shapes, stalling executors.
3. **Regex escaping in Spark SQL:** Patterns passed through `F.expr()` required double-escaping (`\\\\`), which introduced silent bugs — some columns produced all-zero match counts for several runs before being caught.
4. **Metadata override at 70% weight:** When source metadata contained difficulty labels ("hard", "grade: 12"), r5.0 blended them at 70% weight. This made band assignments opaque — the assigned band was dominated by upstream metadata quality, not by the text itself.

---

### Phase 5: Design Pivot — Weak Signals Approximation

**Location:** `docs/design_principles.md`

Before writing new code, the team stepped back and asked what PatternRefinement r5.0 was actually doing. The answer:

> r5.0 is using 20+ regex patterns to detect **presence** of code/math/reasoning content, and then combining these into composite scores. But detecting presence does not require regex. A document containing `"def "`, `"import "`, and `"class "` is code — no Python syntax parsing required.

The key insight is in `docs/design_principles.md`:

> Instead of detecting code with regex, use character diversity, punctuation ratio, avg word length. Instead of complex patterns: count "def", "function", "class", "import", "return". The band assignment uses **composite scores** (sums of many weak signals), so individual false positives wash out.

This is the **weak signals approximation**: replace individual strong (expensive) signals with many cheap (imprecise) signals whose aggregate is just as discriminative. The probabilistic banding framework was already designed to handle noisy inputs — the Gaussian weights and conservative assignment absorb per-signal errors.

**Empirical validation from `docs/analysis/analysis_summary_report_01.md`** (333,981 samples):
- CoT density was extremely low (average 0.0039), confirming that **binary presence flags** (`has_cot`) carry the same information as density ratios.
- Agentic density similarly low (average 0.0040). The 386 agentic samples (0.11%) were correctly floor-capped to B5 in all cases.
- T5-vocabulary content (rare/technical tokens) was 11× more likely to reach advanced bands — confirming that vocabulary proxies (like `unique_token_ratio`) are meaningful band signals.

---

### Phase 6: WeakSignals r7.1 — Production (Current)

**Location:** `pipeline/jobs/main_job.py` (current)
**Infrastructure:** EMR Serverless (Spark standalone)

#### What Changed

**Signal extraction: regex → translate + contains**

Every character-level metric moved from `regexp_replace()` to `F.translate()` — a single O(n) pass with no regex engine, no backtracking:

```python
# r5.0: regexp_replace(text, r'[^a-zA-Z0-9]', '')  →  expensive
# r7.1: translate(text, ".,;:()[]{}!?", "")         →  O(n), no backtracking
```

Every modality signal moved from regex to `F.lower(col).contains(keyword)` over curated keyword lists:

| Signal | Keywords | Examples |
|--------|----------|---------|
| `code_keyword_count` | 20 | `"def "`, `"function "`, `"class "`, `"import "`, `"return "`, `"if ("` |
| `math_keyword_count` | 11 | `"theorem"`, `"lemma"`, `"proof"`, `"integral"`, `"derivative"`, `"qed"` |
| `reasoning_keyword_count` | 10 | `"therefore"`, `"thus"`, `"hence"`, `"consequently"`, `"implies"` |
| `agentic_keyword_count` | 12 | `"execute"`, `"orchestrate"`, `"workflow"`, `"pipeline"`, `"tool use"` |
| `cot_keyword_count` | 8 | `"let's think"`, `"step by step"`, `"breaking down"`, `"analyzing"` |

61 total keyword checks vs 20+ regex patterns. Per-keyword false positives are higher, but composite scores absorb them.

**Adaptive sampling**

Instead of analyzing the full text of every document, r7.1 caps the analysis window:

| Document Size | Sample Size |
|---------------|-------------|
| < 1K chars | Full text |
| 1K–10K chars | First 5,000 chars |
| 10K–50K chars | First 15,000 chars |
| > 50K chars | First 25,000 chars |

A code file does not become prose halfway through. The signal density in the first N characters is a reliable proxy for the full document. This is safe because document type is front-loaded.

**Difficulty score: 7 components → 4 components**

The 4-component difficulty score removed the Flesch readability computation (expensive, unreliable for code/poetry), the 130-keyword rarity proxy (absorbed into keyword lists), and the 70% metadata override:

| Component | Weight | Formula |
|-----------|--------|---------|
| Vocabulary | 25% | `min(unique_token_ratio × 2.5, 1.0)` |
| Length | 25% | `min((avg_word_length − 4) / 6, 1.0)` |
| Structure | 20% | `min(punct_ratio × 3, 1.0)` |
| Specialty | 30% | `min((code_score + math_score + reasoning_score) / 60, 1.0)` |

**Progressive column dropping**

Intermediate columns are dropped at each pipeline stage to prevent memory bloat on Spark executors:

```
adaptive_sample → basic_stats → noise_metrics → DROP text
character_stats → word_stats → keyword_scores → DROP text_sample
composite_scores → difficulty_score → DROP component columns
```

#### What Stayed the Same

The probabilistic banding algorithm was **not changed**. The r5.0 algorithm was the right design; the problem was the signals feeding into it. The same triangular weights, content nudges, normalization, and lowest-credible-band selection run identically in r7.1.

The output schema was also unchanged, so downstream T3 jobs required no modifications.

---

### Infrastructure Migration: Glue → EMR Serverless

PatternRefinement r5.0 ran on AWS Glue (managed Spark). The move to EMR Serverless was driven by performance control and resource efficiency — EMR Serverless gives direct control over executor memory and parallelism without Glue's overhead pricing.

The migration required rewriting the job submission mechanism (from Glue job configs to EMR Serverless job runs) but the PySpark code itself was identical.

The intermediary migration scripts were one-off utilities used once during migration and have since been removed.

**A note on the threshold tuning in ProgressiveFilter r2.7:** The whitespace ratio threshold was raised from 0.60 to 0.75, and the non-printable ratio from 0.01 to 0.03, specifically because book-format data (with chapter breaks and Unicode formatting) was being over-rejected. This illustrates the iterative nature of the rejection policy work — every threshold was arrived at by observing false rejections on real data, not by a priori reasoning. See `docs/CHANGELOG.md` for the full r2.x history.

---

## 4. Parallel Execution Design

The three production jobs were designed from the start to run in parallel across datasets. This is not just a convenience — it is how the pipeline was validated and how production runs were executed.

### Development Strategy

Initial experimentation was done on **smaller dataset samples** (a few hundred thousand records per source) to validate the banding logic without incurring the cost of full-corpus runs. Each dataset was processed independently through the pipeline, and band distributions were inspected before moving to full scale.

### Production Execution

On EMR Serverless, the final production runs were executed **in parallel — one job run per dataset**. Each dataset was submitted as a separate EMR Serverless job run with its own `--JOB_NAME` argument and output path under `OUTPUT_BASE`. All three jobs (main, curated datasets, student data) ran concurrently, each writing to its own `band=Bx/` partition prefix.

This means:
- The three job scripts are independent — they share no state.
- Multiple datasets within the main job were also run in parallel (one EMR Serverless job per large source like RedPajama, FineWeb, Dolma).
- The output schema is identical across all jobs, so the per-source outputs can be merged by simply pointing T3's reader at the same `OUTPUT_BASE` prefix.

### Why Parallel Execution Matters for the Band Distribution

Because each dataset ran independently and in parallel, the final corpus band distribution is a sum of per-dataset distributions. The source clamping in `pipeline/jobs/curated_datasets_job.py` and the B0–B2 ceiling in `pipeline/jobs/student_data_job.py` were designed with this in mind — the aggregate B0–B5 split across all sources is controlled by (a) per-dataset clamping and (b) the sampling weights in `curriculum.yaml`, not by any cross-dataset coordination at processing time.

---

## 5. Production Architecture: Three Jobs

The final production system is three independent EMR Serverless jobs. They share the same probabilistic banding framework but differ in what data they cover and how they handle source-specific constraints.

```
T1 Output (Parquet on S3)
         │
    ┌────┴─────────────────────┐
    │                          │                          │
 Main Job               Curated Datasets        ERAv4 Student Data
 main_job.py            curated_datasets_job.py  student_data_job.py
 RedPajama, FineWeb,    HuggingFace SFT/          Student Q&A drills
 Dolma, arXiv, etc.     Math/Code datasets        Samvaad conversation
 Full B0–B5 range       Source-clamped            B0–B2 only
         │                    │                          │
         └────────────────────┴──────────────────────────┘
                              │
                    Unified Output Schema
                    (partitioned by band)
                              │
                         T3 Training Jobs
```

### Job 1: Main Job (`pipeline/jobs/main_job.py`)

Covers the large-scale web/book/code corpus. Full B0–B5 band range. Standard 4-component difficulty formula. No source clamping — the probabilistic algorithm assigns bands freely based on text content.

### Job 2: Curated Datasets (`pipeline/jobs/curated_datasets_job.py`)

Covers 17 curated instruction, preference, math, and code datasets from HuggingFace. Key differences:
- Specialty weight in difficulty formula raised to 40% (these datasets are denser with technical signals).
- Additional signals: `latex_score`, `science_score`, `conversation_score`, `step_score`.
- **Source clamping**: each dataset is clamped to a known `[floor, ceiling]` band range based on prior knowledge of the dataset.

Why clamping matters: a short GSM8K problem might score B1 on the generic difficulty formula because it uses simple language. But GSM8K requires multi-step arithmetic reasoning — it should never be below B3. Clamping preserves relative ordering within a dataset (harder GSM8K problems → B4, easier → B3) while preventing obviously wrong assignments.

Clamp ranges (selected):

| Dataset | Floor | Ceiling | Rationale |
|---------|-------|---------|-----------|
| samvaad_hi | B0 | B2 | Everyday Hindi conversation |
| gsm8k | B3 | B4 | Grade school math (capped — not PhD) |
| helpsteer | B3 | B5 | Multi-attribute instruction following |
| nemotron_math | B4 | B5 | Expert math |
| teichai / high_reasoning | B4 | B5 | High-reasoning traces |

### Job 3: ERAv4 Student Data (`pipeline/jobs/student_data_job.py`)

Covers student-generated Q&A drills and Samvaad conversation data. Completely different difficulty formula (5 components: vocabulary, length, Q&A density, language, conversation markers). Code/math/reasoning/agentic scores are hardcoded to 0 — this data doesn't contain those modalities, so the generic signals would be noise. Clamped to B0–B2 with relaxed rejection filters (the data is pre-curated, so only extreme cases like `char_length < 5` are rejected).

### Output Schema (All Three Jobs)

Every job writes the same column set:

| Group | Columns |
|-------|---------|
| Identity | `uuid`, `id`, `file_path`, `source`, `domain`, `hash`, `language`, `metadata` |
| Band | `assigned_band`, `band`, `difficulty_score`, `band_p_B0`–`band_p_B5` |
| Modality flags | `has_code`, `has_cot`, `has_reasoning`, `has_agentic` |
| Scores | `agentic_score`, `cot_score`, `reasoning_score`, `code_score`, `math_score` |
| Size | `byte_length`, `word_count`, `unique_token_ratio`, `compression_ratio`, `token_count_estimate` |
| Rejection | `is_rejected`, `rejection_reason`, `rejection_level` (rejected docs only) |

Output is partitioned by `band` as Parquet with zstd compression.

---

## 6. Quality Rejection Policy

Rejection is intentionally permissive. The design target is **95–98% pass-through** — only extreme garbage is rejected, because curriculum learning handles difficulty progression better than hard filtering.

### Stage 1: Physical Corruption

| Rule | Condition |
|------|-----------|
| Too short (bytes) | `byte_length < 50` |
| Too short (chars) | `char_length < 20` |
| Too few tokens | `token_count_estimate < 10` |

### Stage 2: Noise & Spam

| Rule | Condition | What It Catches |
|------|-----------|-----------------|
| Repetitive template | `unique_token_ratio < 0.01` AND `word_count > 200` | Same word repeated 200+ times |
| Excessive whitespace | `whitespace_ratio > 0.95` | Mostly blank |
| Link spam | `url_ratio > 0.7` AND `url_count > 50` | Link farm |
| Boilerplate | `boilerplate_ratio > 0.50` | Cookie policy / terms of service |
| Thread fragment | `thread_marker_count > 5` AND `token_count < 200` | Orphaned forum reply |

Stage 3 was removed. The previous Stage 3 had a 60%+ false positive rate — it was rejecting valid code files, poetry, and structured documents that failed prose-based quality heuristics.

---

## 7. The Probabilistic Banding Algorithm (Full Detail)

This is the core mechanism, identical across all three jobs. Full reference in `docs/band_assignment_methodology.md`.

### Step 1: Triangular Base Weights

For each band b, compute a weight that peaks at the band's centroid and falls to zero at distance WIDTH = 0.20:

```
w_b = max(0, 1 − |difficulty_score − center_b| / 0.20)
```

Example: `difficulty_score = 0.40`
- w_B2 = `max(0, 1 − |0.40 − 0.35| / 0.20)` = **0.75**
- w_B3 = `max(0, 1 − |0.40 − 0.55| / 0.20)` = **0.25**
- All others: 0.0

### Step 2: Content-Based Nudges

Composite scores add weight to specific bands, allowing content type to override raw difficulty:

| Condition | Band Nudged | +Weight | Intent |
|-----------|-------------|---------|--------|
| code_score 1–14 | B1 | +0.05 | Trivial code → at least Primary |
| code_score 15–24 | B2 | +0.08 | Intro technical → High School |
| code_score ≥ 25 | B3 | +0.10 | Meaningful code → Undergraduate |
| reasoning_score ≥ 5 | B2 | +0.05 | Implicit reasoning → structured |
| reasoning_score ≥ 8 | B3 | +0.08 | Multi-step reasoning |
| cot_score ≥ 10 | B3 | +0.05 | Chain-of-thought present |
| math_score ≥ 12 | B4 | +0.15 | Formal math → Graduate |
| code_score ≥ 40 | B4 | +0.12 | Hard code → Graduate |
| **agentic_score ≥ 8** | **B5** | **+0.20** | **Tool-use traces → PhD** |
| math_score ≥ 20 | B5 | +0.12 | Advanced math |
| reasoning ≥ 15 AND code ≥ 30 | B5 | +0.10 | Combined technical depth |

### Step 3 & 4: Normalize and Assign

```
total = sum(w_b for all bands)
band_p_Bi = w_Bi / total

# Assign: lowest band with p ≥ 0.15
assigned_band = lowest b where band_p_Bi ≥ 0.15
```

**Worked example** — 2,000-word Python tutorial with 3 code blocks:
- `code_keyword_count` = 9 → `code_score` = 46
- `reasoning_keyword_count` = 2 → `reasoning_score` = 13
- `difficulty_score` = 0.618
- Base weights: w_B3 = 0.66, w_B4 = 0.34
- After nudges (code ≥ 25 → B3 +0.10; code ≥ 40 → B4 +0.12; reasoning ≥ 12 → B4 +0.10): w_B3 = 0.76, w_B4 = 0.56
- Normalized: p_B3 = 0.576, p_B4 = 0.424
- Lowest credible: **B3** ✓ (correct — a Python tutorial with code examples is Undergraduate-level)

---

## 8. Validation and Evidence

### Band Distribution on Sample Data

From `docs/analysis/analysis_summary_report_01.md` (333,981 samples, Feb 2026):

| Band | Count | % | Notes |
|------|-------|---|-------|
| B0 | 228,154 | 68.3% | Matches the 1B-stage target weight of ~49% (larger because the full corpus is more B0-heavy than any single stage) |
| B1 | 38,222 | 11.4% | |
| B2 | 42,951 | 12.9% | |
| B3 | 12,572 | 3.8% | |
| B4 | 3,933 | 1.2% | |
| B5 | 8,149 | 2.4% | Agentic samples correctly 100% in B5 |

Key finding: T5-vocabulary content (rare/technical tokens) is **11× more likely** to reach B3–B4 than standard content. This validates that the `unique_token_ratio` proxy in the difficulty formula is tracking something real.

### The NCERT Case Study: Why Band Definitions Needed Iteration

From `docs/analysis/ncert_band_assignment_analysis_01.md`:

NCERT educational content initially had 90% of records classified as B0 (Nursery). The root cause: technical educational text uses **simpler sentence structures** (short declarative sentences) but **specialized vocabulary**. The naive difficulty formula assigned it B0 because it looked linguistically simple.

The fix was domain precedence: if `domain == math_science`, apply a soft floor of B3. This is captured in the curriculum.yaml band-domain policy (science and math are B3–B5 domains).

The iterative process for NCERT:
1. **Initial state:** 90% B0 (wrong — physics textbooks are not Nursery level)
2. **After domain precedence:** 66% B3, 21% B4, 13% B5 (more reasonable)
3. **After metadata-driven promotion:** "Hard" and "Advanced" content promoted to B4, removing the B3 saturation
4. **After complexity stratification:** B4/B5 split using `question_complexity` and `question_type`

This case study directly explains why the `source_clamping` mechanism was introduced in the curated datasets job — prior knowledge about a dataset's content type is a better signal than the generic difficulty formula for edge cases.

### Throughput

Running on EMR Serverless: ~2,200–2,500 records/second. For a 4TB corpus with average record size of ~5KB, this corresponds to ~8–16 hours of wall-clock time.

---

## 9. Upstream and Downstream

### Upstream: Team 1 (Data Engineering)

Team 2 consumes Parquet files from Team 1's pipeline. Required fields per record:

```
id, text, source, added, created, metadata, domain, language
```

The `domain` field is Team 1's classification (used directly in band-domain policy). The `language` field is used for language filtering (English primary, 11 Indic languages allowed, others dropped).

### Downstream: Team 3 (Training Pipeline)

Team 2's output is the input to T3's training job. T3 uses `band` to construct per-stage training batches. The output schema was deliberately kept stable across all versions to avoid requiring T3 to update.

The tokenizer difficulty proxy (Team 6) is intended as a secondary validation signal — the `tokenizer` constraints in `curriculum.yaml` (avg_max, max_max, p95_max per band) are defined but not yet enforced in the pipeline. This is noted as a TODO in curriculum.yaml.

The agentic trace format (Team 17) is similarly in `format_pending` state — agentic content is assigned B5 by the pipeline, but the specific format of tool-call traces is Team 17's responsibility.

---

## 10. Running the Pipeline

### Prerequisites

- AWS account with EMR Serverless application provisioned
- S3 bucket with T1 output Parquet files
- EMR Serverless execution role with S3 read/write permissions

### Configuration

Each script reads the following arguments:

```bash
--INPUT_BASE        s3://your-bucket/t1-output/
--OUTPUT_BASE       s3://your-bucket/t2-output/
--JOB_NAME          t2_main_job_r7
--LOG_LEVEL         INFO
```

See `pipeline/README.md` for the full EMR Serverless job configuration (executor memory, core count, Spark settings).

### Expected Outputs

```
s3://your-bucket/t2-output/
├── bands/
│   ├── band=B0/
│   ├── band=B1/
│   ├── band=B2/
│   ├── band=B3/
│   ├── band=B4/
│   └── band=B5/
└── rejections/
    ├── rejection_level=1/
    └── rejection_level=2/
```

### Expected Rejection Rate

95–98% of records pass both rejection stages. If rejection rate exceeds 10%, check:
1. Whether the source dataset has an unusual encoding (e.g., non-UTF-8 bytes triggering Stage 1)
2. Whether `boilerplate_ratio` calculation is appropriate for the source type (templates can inflate this)

### Logs

EMR Serverless job logs are available in CloudWatch under the application's log group. Sample logs from smaller dataset runs are in `logs/` (added after initial runs). The logs capture per-stage record counts, rejection rates, and band distribution for each job run.

### Reproducing Results

The pipeline is deterministic given the same input Parquet files. The banding algorithm uses no RNG — all scores are computed from the text and keyword hits. The same input will always produce the same `band` and `difficulty_score`.

The only non-deterministic element is the Spark shuffle order (which partition a record ends up in), but this does not affect individual record assignments — only the order of records within each partition.

---

## 11. Known Limitations and Future Work

### Limitations

**1. Weak signals have false positives at the document level**
A document containing "def " may not be code (e.g., "The definition of justice..."). The pipeline relies on composite scores absorbing individual false positives. At scale this works, but for edge-case analysis of individual documents the per-keyword signals are noisy.

**2. No tokenizer difficulty proxy integration (Team 6 TODO)**
The `curriculum.yaml` defines tokenizer constraints per band (avg_max, p95_max) but the pipeline does not enforce them. The tokenizer proxy from Team 6 is listed as a required output but was not available at pipeline build time.

**3. Language filtering relies on T1 labels**
The `language` field filtering (English/Indic only) trusts Team 1's language detection. If T1 mis-labels a document, T2 does not re-detect language.

**4. Agentic trace format is placeholder**
The pipeline can detect agentic vocabulary and assign B5, but the specific format requirements for tool-use traces (Team 17's responsibility) are not yet enforced. The `agentic_traces` modality in curriculum.yaml is marked `format_pending`.

**5. Curriculum.yaml status is DRAFT, not FROZEN**
The yaml is marked `status: "DRAFT"`. The band definitions and growth schedule are stable, but the DRAFT status means structural changes are still technically allowed. This should be updated to FROZEN before training begins.

**6. No structured persistent logs from EMR runs**
Per the PR checklist, structured JSON logging and log archiving were not implemented. EMR Serverless console logs capture run details but are not archived to S3 automatically. The Infra team needs to set up CloudWatch log export for long-term retention.

### Future Work

- Integrate Team 6's tokenizer difficulty proxy as a secondary validation signal
- Add per-source rejection reports (currently aggregated across all sources in a run)
- Make B5 agentic assignment conditional on Team 17's trace format being present
- Consider a spot-check evaluation loop using a small frozen LM to audit a sample of band assignments per run (referenced but not implemented in curriculum.yaml's `audit` section)

---

## 12. File Reference

### Production Pipeline

| File | Purpose |
|------|---------|
| `pipeline/jobs/main_job.py` | Main job — large-scale web/book/code datasets (B0–B5) |
| `pipeline/jobs/curated_datasets_job.py` | Curated HF datasets, source-clamped |
| `pipeline/jobs/student_data_job.py` | ERAv4 student data and Samvaad, B0–B2 |
| `pipeline/README.md` | How to run all three jobs on EMR Serverless |

### Policy and Configuration

| File | Purpose |
|------|---------|
| `curriculum.yaml` | Canonical curriculum policy — single source of truth |
| Team's Charter | Team mandate and band definitions |
| `src/band_assignment.yaml` | Band constraint config for `curriculum_extractor` library |
| `src/metrics_config.yaml` | Metrics plugin config for `curriculum_extractor` library |

### Documentation

| File | Purpose |
|------|---------|
| `docs/CHANGELOG.md` | Complete version history r2.1 → r7.1 |
| `docs/band_assignment_methodology.md` | Full methodology with all formulas and worked example |
| `docs/pipeline_evolution.md` | PatternRefinement r5.0 vs WeakSignals r7.1 side-by-side comparison |
| `docs/band_definitions.md` | Canonical B0–B5 definitions |
| `docs/design_principles.md` | Core design principles |

### Analysis and Validation

| File | Purpose |
|------|---------|
| `docs/analysis/analysis_summary_report_01.md` | Band distribution on 333K samples |
| `docs/analysis/ncert_band_assignment_analysis_01.md` | NCERT domain precedence case study |
| `docs/analysis/band_modality_dist_analysis_01.md` | Modality distribution findings |
| `docs/analysis/analysis_summary_report_02.md` | Tokenizer level vs difficulty band analysis |
| `docs/analysis/modality_distribution_analysis_01.md` | Modality distribution vs difficulty bands |

### Reference Libraries

| File | Purpose |
|------|---------|
| `src/curriculum_extractor/` | Python reference implementation (single-record extraction) |
| `src/curriculum_reader/` | Batch creation utilities for downstream consumers |
| `curriculum_tags/` | Phase 1 implementation — kept for historical reference |

---

## Appendix: Key Design Principles

These principles appear repeatedly across the codebase and are worth making explicit:

**Downgrade on uncertainty.** When a document has non-trivial probability in adjacent bands, assign the lower one. The cost of under-challenging the model is recoverable (the document will reappear at later stages); the cost of over-challenging it is training instability.

**Immutability first.** Source records are never modified. Metadata is stored in separate columns. This is enforced architecturally (the original `text`, `id`, `source`, `domain` columns are never overwritten).

**Early rejection, permissive thresholds.** Only reject documents that are verifiably garbage (byte-level corruption, pure link farms, blank files). Valid training data that is "low quality by prose standards" goes to B0, not to the rejection file.

**Weak signals, composite scores.** Individual signals (`code_keyword_count`, `special_ratio`) are imprecise. The composite scores (`code_score`, `math_score`) average out the noise. The probabilistic banding framework absorbs remaining uncertainty.

**Source-aware clamping for curated data.** When you know a dataset's difficulty range from prior knowledge, use it. The generic formula is designed for uncurated web text; it will mis-rate edge cases in domain-specific datasets.

**Stable output schema.** Downstream teams (T3) should not need to update when T2 changes its internal processing. The output schema is a public contract; internal implementation is private.

**Parallel execution by design.** Each dataset is processed independently, enabling parallel EMR Serverless job runs. The aggregate corpus band distribution is controlled through per-dataset clamping and the sampling weights in `curriculum.yaml`, not by cross-dataset coordination.
