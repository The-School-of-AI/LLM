# T2 Curriculum Pipeline — Version History

All version history for the band assignment pipeline, consolidated from inline docstrings.
For methodology details see `band_assignment_methodology.md`. For the PatternRefinement r5.0 → WeakSignals r7.1 architectural comparison see `pipeline_evolution.md`.

Version naming convention: `<PhaseName> r<major>.<minor>`

| Revision | Name | Infrastructure |
|----------|------|---------------|
| r7.1 | WeakSignals | EMR Serverless (current) |
| r5.0 | PatternRefinement | AWS Glue |
| r4.0 | ProbabilisticBanding | AWS Glue |
| r2.1–r2.8 | ProgressiveFilter | AWS Glue |

---

## WeakSignals r7.1 — EMR Serverless (Production, Current)

**Script:** `pipeline/jobs/main_job.py` (current)
**Infrastructure:** EMR Serverless (Spark standalone)

**What changed from PatternRefinement r5.0:**
- Signal extraction: 20+ `regexp_count()` on full text → 61 keyword `contains()` on adaptive text sample
- Character metrics: `regexp_replace()` → `F.translate()` (single O(n) pass, no backtracking)
- Difficulty: 7 components (Flesch + metadata 70% override) → 4 components (vocab, length, structure, specialty)
- Metadata override removed entirely
- Adaptive sampling: full text → first N chars based on document size (< 1K: full; 1K–10K: 5K; 10K–50K: 15K; > 50K: 25K)
- Progressive column dropping to prevent Spark executor memory bloat
- Infrastructure: AWS Glue FLEX → EMR Serverless

**What stayed the same:**
- Probabilistic banding algorithm (triangular weights → nudges → normalize → lowest credible band)
- Two-stage quality rejection (Stage 1: physical corruption; Stage 2: noise/spam)
- Output schema (downstream T3 jobs unchanged)
- Conservative "downgrade on uncertainty" principle (ε = 0.15)

---

## PatternRefinement r5.0 — Glue Baseline (Reference)

**Script:** `pipeline/jobs/main_job.py` (r5.0 commit), `glue_jobs/claude_reviewed/v1_t2_metrics_calculator_v5.py`
**Infrastructure:** AWS Glue, G.2X workers × 20, FLEX execution

**Key additions over ProbabilisticBanding r4.0:**
- Improved 9 overly broad patterns (CODE_PATTERN, MATH_PATTERN, AGENTIC_PATTERN, COT_PATTERN, REASONING_PATTERN, TABLE_PATTERN, CODE_COMMENT_PATTERN, QUESTION_PATTERN, symbol_count)
- Removed 12 ineffective metrics (see below)
- Removed all Stage 3 rejections (60%+ false positive rate on valid data)
- Expected data retention increased: 95% → 98%+

**12 metrics removed with rationale:**

| Metric | Reason |
|--------|--------|
| `non_printable_ratio` | Blocked Indic Unicode (Hindi, Tamil, etc.) — valid chars flagged as non-printable |
| `html_tag_density` | Counted chars between `<html>` and `</html>` — 40% false positive on code tutorials |
| `risky_tld_count` | Flagged security research papers discussing `.tk`/`.ml` domains |
| `sentence_boundary_coherence` | Assumed prose — rejected code, poetry, lists (50–70% false positive) |
| `punctuation_density` | 0.02 correlation with quality (random) |
| `dependency_depth_estimate` | Code legitimately has 100+ brackets; no spam signal |
| `num_numeric_tokens` | Inverse correlation with spam (code/science have more numbers) |
| `citation_count` | Never used in any rejection or band rule |
| `list_marker_count` | 0.05 correlation with quality |
| `step_indicator_count` | 60% false positive on recipes, storytelling, normal instructions |
| `ellipsis_count` | 0.10 correlation with truncation (too weak) |
| `dialogue_turn_count` | −0.02 correlation with quality (negative signal) |

**Stage 3 rejections removed:**
- `flesch_reading_ease < -50 OR > 150` → "invalid_readability_score"
- `dependency_depth_estimate > 50` → "excessive_nesting_corruption"
- `truncation_indicators > 4` → "incomplete_truncated_content"
- `code_comment_ratio > 0.9` → "code_mostly_comments"

---

## ProbabilisticBanding r4.0 — First Full-Scale Spark Job

**Script:** `pipeline/jobs/main_job.py` (r4.0 commit), source: `glue_jobs/notes/failing_job.py`
**Infrastructure:** AWS Glue, G.2X workers × 20, FLEX execution

**Philosophy shift:** FROM deterministic band labels TO soft probability distributions.

**Key additions:**
- 6 probability columns: `band_p_B0` through `band_p_B5`
- `final_band`: conservative assignment (lowest credible band, ε = 0.10)
- `difficulty_score`: single scalar [0,1] from cheap heuristics
- `fertility_estimate`: char/token ratio (re-added for analysis)
- Removed hard modality overrides → replaced with small probability nudges (+0.05 to +0.15)
- Broadcast variable for high-value keywords (~150 words) instead of expensive regex
- Column pruning: reads only `(id, text, source, domain)` initially
- Coalesce after Stage 1 to reduce partitions without full shuffle

**Band probability computation (single pass):**
1. Compute `difficulty_score` ∈ [0,1] from length, structure, reasoning, symbols, rarity
2. Map score → band probabilities using triangular weighting (fixed band centers)
3. Apply small content nudges for code/agentic/research content
4. Normalize to sum = 1, emit 6 probability columns
5. Select `final_band` = lowest band where p(band) ≥ ε

**Band centers (fixed):** B0: 0.05, B1: 0.20, B2: 0.35, B3: 0.55, B4: 0.75, B5: 0.90

**Why this failed at scale:** Regex-heavy signal extraction still present (20+ patterns on full text). Catastrophic backtracking on certain document shapes. Glue FLEX regex escaping bugs caused silent zero-match columns.

---

## ProgressiveFilter r2.8 — Training-Optimized Thresholds

**Philosophy shift:** FROM "keep only high quality" TO "reject only extreme noise."
- Comprehensive threshold analysis based on Dolma/Sangraha dataset characteristics
- Expected data retention: 85–90% (up from 60–70%)
- Expected false positive rate: < 5% (down from 20–30%)

**Phase 1 — Books recovery (90% rejection → 10% rejection):**
- `whitespace_ratio`: 0.75 → 0.85 (accommodate chapter breaks, poetry, code)
- `non_printable_ratio`: 0.03 → 0.10 (support Unicode scripts, math symbols)

**Phase 2 — Precision tuning (reduce false positives across domains):**
- `unique_token_ratio`: 0.1 → 0.05 + length check (preserve repetitive but valid content)
- `capitalization_ratio`: 0.6 → 0.7 + word_count 50→100 (reduce title false positives)
- `url_ratio`: 0.3 → 0.4 + url_count check (preserve papers with citations)
- `html_tag_density`: 0.05 → 0.10 + length check (preserve code examples)
- `boilerplate_ratio`: 0.15 → 0.25 (preserve legitimate metadata)
- `risky_tld_count`: > 0 → > 3 (allow security research mentions)
- `sentence_boundary_coherence`: 0.5 → 0.2 (reduce code/poetry false positives)
- `truncation_indicators`: 2 → 4 + length check (preserve multi-part articles)
- `code_comment_ratio`: 0.8 → 0.9 (preserve tutorial/documentation code)

---

## ProgressiveFilter r2.7 — Book-Friendly Thresholds

- Fixed excessive rejections of book content: 1610 out of 1738 books were being rejected
- `whitespace_ratio`: 0.60 → 0.75 (books have chapter breaks and structured layout)
- `non_printable_ratio`: 0.01 → 0.03 (books have Unicode formatting)
- `capitalization_ratio`: 0.50 → 0.60 (books have chapter titles)
- Added minimum `word_count` checks to capitalization and corruption rules

---

## ProgressiveFilter r2.6 — S3 Write Fix for FLEX

- Fixed `UNCLASSIFIED_ERROR: Failed to delete key intermediate data`
- Changed write mode from `overwrite` to unique timestamped paths
- Prevents S3 deletion conflicts in FLEX execution (stricter permissions)
- Timestamp format: `stage1_rejected_YYYYMMDD_HHMMSS`

---

## ProgressiveFilter r2.5 — Spark SQL Syntax Fix

- Fixed `INVALID_PARAMETER_VALUE.REGEX_GROUP_INDEX` error in `regexp_extract_all`
- Added explicit group index parameter (`idx=0`) to all 23 `regexp_extract_all` calls
- Spark SQL requires group index even for patterns without capture groups
- Without this fix, job fails immediately at first pattern extraction

---

## ProgressiveFilter r2.4 — Glue FLEX Compatibility

- Removed all forbidden `spark.conf.set()` calls for FLEX execution
- Moved `spark.network.timeout` and `spark.sql.broadcastTimeout` to CLI `--conf`
- Required `--conf` parameters documented: `spark.network.timeout=600s`, `spark.sql.broadcastTimeout=1200s`

---

## ProgressiveFilter r2.3 — 4TB Production Optimization

- Fixed `CANNOT_MODIFY_CONFIG` error (removed forbidden `spark.memory.*` configs)
- Increased shuffle partitions: 2000 → 8000 for 4TB scale (~512MB per partition)
- Enabled skew join handling for unbalanced domain/source distribution
- Added S3 fast committer (`mapreduce.fileoutputcommitter` v2)
- Removed 7 expensive global actions (count/collect calls eliminated)
- Let AQE auto-determine partition counts instead of fixed repartition

---

## ProgressiveFilter r2.2 — Regex Optimization

- Batch regex processing: reduced string traversals from 60+ to ~10 per document
- Stage 2: single-pass pattern extraction using `regexp_extract_all` (4 scans vs 15+)
- Stage 3: single-pass modality detection (8 scans vs 25+)
- 60–75% reduction in Stage 2/3 compute time

---

## ProgressiveFilter r2.1 — Initial Working Version

- Added `fertility_estimate`, `rare_word_ratio_estimate`, `mtld_estimate`, `information_density_estimate`
- Added modality detection: `has_code`, `has_math`, `has_agentic`, `primary_modality`
- Added difficulty scoring: `difficulty_score`, `difficulty_level` (L0–L5)
- Removed textbook-killing length rejections (documents up to 10MB+ now supported)
- Physical S3 writes instead of checkpointing (100% reliable for 4TB data)
- Partitioned outputs by `(domain, source)` for easy downstream joins
- Incremental processing support via `--SOURCE` parameter
