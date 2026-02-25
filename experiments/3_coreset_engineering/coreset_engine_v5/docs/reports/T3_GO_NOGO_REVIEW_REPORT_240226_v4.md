# Critical Review — T3 Coreset Engineering Implementation

**Reviewer**: Independent expert validation — first-principles architecture, code, and production-readiness audit  
**Date**: 2026-02-24  
**Scope**: Full `coreset_engine_v5/` codebase (`coreset_builder.py`, `src/`, `tools/`, `config/`, `tests/`, `schemas/`), T3 Report (`2026-02-23_T3_REPORT.md`), CI/CD pipeline, deployment automation (`commands.sh`, `shard.sh`), and design documentation (`DESIGN_AND_RECOMMENDATIONS.md`, `INFRA_RECOMMENDATIONS_DECISION_MATRIX.md`, `PROJECT_MANIFEST.py`)  
**Review Type**: Comprehensive technical critique covering architecture, selection strategy, deduplication, curriculum adherence, scalability, determinism, fault tolerance, testing coverage, report integrity, code quality, risk analysis, and production readiness


## Table of Contents

1. [Executive Assessment](#1-executive-assessment)
2. [Architecture & Design Review](#2-architecture--design-review)
3. [Selection Strategy Critique](#3-selection-strategy-critique)
4. [Deduplication Strategy Review](#4-deduplication-strategy-review)
5. [Curriculum Adherence & Band Distribution Analysis](#5-curriculum-adherence--band-distribution-analysis)
6. [Streaming & Scalability Design Review](#6-streaming--scalability-design-review)
7. [Determinism & Reproducibility Critique](#7-determinism--reproducibility-critique)
8. [Fault Tolerance & Checkpoint/Resume Review](#8-fault-tolerance--checkpointresume-review)
9. [Testing & Validation Coverage Assessment](#9-testing--validation-coverage-assessment)
10. [Report & Metrics Integrity Audit](#10-report--metrics-integrity-audit)
11. [Code Quality & Maintainability Review](#11-code-quality--maintainability-review)
12. [Risk Register & Failure Mode Analysis](#12-risk-register--failure-mode-analysis)
13. [Production Readiness Assessment](#13-production-readiness-assessment)
14. [Summary & Prioritized Recommendations](#14-summary--prioritized-recommendations)


## 1. Executive Assessment

### Overall Assessment: **Solid Engineering Foundation with Critical Analytical Gaps**

The coreset engineering pipeline represents a **well-engineered, production-oriented system** with strong fundamentals in streaming architecture, fault tolerance, and deterministic control. The codebase demonstrates mature software engineering practices including checkpoint/resume, sharded parallelism, and backwards-compatible output formats.

However, the **analytical claims in the report are significantly undermined** by the narrow execution scope (single source `C4`, single domain `web`, single language `en`), and several architectural decisions remain **unvalidated under realistic multi-source conditions**. The implementation gap between what the design documentation promises and what has been operationally verified is substantial.

| Dimension | Notes |
|-----------|-------|
| **Architecture & Design** | Clean separation, well-structured; some God-object concerns |
| **Selection Strategy** | Sound in theory; untested under multi-band/multi-domain conditions |
| **Deduplication** | Near-dedup entirely excluded; hash-based dedup is EMR-level, not in-pipeline |
| **Curriculum Adherence** | B3/B4/B5 completely absent in outputs; curriculum design cannot be validated |
| **Scalability** | Streaming + sharding + prefetch show mature scale thinking |
| **Determinism** | Strong seeding and checkpoint controls; FP nondeterminism risk remains |
| **Fault Tolerance** | Excellent checkpoint/resume with guard rails |
| **Testing** | Good unit coverage; lacks integration tests on realistic data |
| **Report Quality** | Numerically sound but analytically misleading in framing |


## 2. Architecture & Design Review

### 2.1 Strengths

**Modular Component Design**: The `src/` directory is cleanly organized into:
- `core/` (config, types) — Clean data contracts via dataclasses and enums
- `curriculum/` — Separate loader with validation
- `dedup/` — Pluggable dedup strategies
- `selection/` — Layered selection engine (base + batched)
- `io/` — Data loading, writing, and chunk store
- `error_handling.py` — Centralized error management

This follows a principled **separation of concerns** pattern and enables independent testing and evolution of each component.

**Two-Tier Builder Pattern**: The `CoresetBuilder` → `StreamingCoresetBuilder` inheritance is a sound architectural decision:
- `CoresetBuilder`: Legacy in-memory path for small-scale validation
- `StreamingCoresetBuilder`: Production streaming path with batching, sharding, and checkpointing

### 2.2 Weaknesses

#### CRITICAL: God-Object Anti-Pattern in `coreset_builder.py`

`coreset_builder.py` is **2,030 lines** containing two primary classes, a massive `main()` function (~290 lines), and the entire CLI argument parsing inline. The `StreamingCoresetBuilder._build_stage_coreset()` method alone spans **lines 950–1740** (~790 lines), making it the single most complex function in the codebase.

**Impact**: This monolithic method handles:
- Token resolution and shard scaling
- Checkpoint loading and compatibility verification
- Batch iteration with prefetch
- Per-row metadata parsing and band inference
- Non-overlap filtering via SQLite + LRU cache
- Availability accounting
- Selection engine dispatch
- Output writing (multi-format: Parquet/CSV/JSONL)
- Manifest generation
- Rolling-window stats propagation

This violates the **Single Responsibility Principle**. Any bug in the 790-line method is difficult to isolate, and unit testing individual behaviors requires mocking at extreme depths.

**Recommendation**: Extract at least the following into dedicated methods or modules:
1. Row-level parsing and band inference → `RowParser` or `ChunkParser` class
2. Output writing → dedicated writer helper in `src/io/`
3. Checkpoint validation → `CheckpointValidator` class
4. Manifest construction → standalone builder

#### MODERATE: Hardcoded Fallbacks Throughout

Multiple locations use hardcoded fallback values without centralized defaults:
- Band defaults to `"B0"` when missing (line 1219, 1287)
- Language defaults to `"en"` (line 1304)
- Domain defaults to `"unknown"` (line 1238–1239)
- Dataset ID defaults to `"ds"` (line 1295)

These defaults are scattered across the codebase rather than being centralized constants, making it easy for inconsistencies to emerge.

#### MINOR: Inline Imports

Several imports appear inside method bodies (e.g., `import pandas as pd` at line 1063, `from collections import Counter` at line 1049, `from src.core.types import ChunkMetadata, DifficultyBand` at line 1064). While sometimes necessary for circular dependency avoidance, here they appear to be for lazy loading convenience. This hurts readability and makes dependency tracking harder.


## 3. Selection Strategy Critique

### 3.1 The Stated Strategy

The pipeline uses **"stratified density-aware selection"**: it creates `(band, domain)` buckets, assigns token budgets from curriculum ratios, scores chunks within buckets, then greedily selects highest-scored chunks until each bucket's budget is met.

### 3.2 Analysis

#### STRENGTH: Proportional Batch-Level Budget Allocation

The `_process_batch()` method in `engine_batched.py` (lines 285–511) is well-designed:
- Batch-level token budgets are proportioned from `stage_target_tokens * (batch_tokens_raw / total_input_tokens_estimate)`
- Stage-level carryover accounting (`_remaining_band_tokens`, `_remaining_stage_tokens`) ensures batches that underperform a band's quota can be compensated in later batches
- Per-bucket targets are capped by remaining quotas via `_cap_bucket_targets_by_remaining`

This is a **correct streaming approximation** of global stratified selection.

#### CRITICAL: Scoring Function Is Effectively Random Under Current Configuration

Inspecting `_score_chunks_in_bucket()` in `engine.py` (lines 208–228), the composite score depends on:
1. **Token-level rarity** (via `DiversityScorer`) — but the report explicitly states *"Token-level rarity scoring is optional and only applies when tokenizer artifacts like token_ids exist; otherwise it is skipped by design."* Since the streaming pipeline receives metadata-only parquet rows **without** `token_ids`, token rarity scoring is **completely inactive**.
2. **Domain diversity** and **language diversity** — but C4 has only `domain=web` and `language=en`, yielding zero diversity signal.
3. **Band score** / **difficulty score** — extracted via `_extract_band_score()`. If present in the input parquet columns, this is the only active scoring signal.

**Under the current C4-only run, if `band_score` / `difficulty_score` columns are present, chunks within the same bucket are ranked by that score. If not, selection within buckets is effectively random (seed-controlled).**

This means the "5.62x compression" claim is technically correct but the **quality signal of the compression is unknown** — the pipeline may be randomly subsampling rather than selecting higher-quality content.

**Recommendation**: 
- Verify and document which scoring columns are present in the C4 input data
- If no meaningful scoring signal exists, reframe the compression as "stratified subsampling" rather than "selection"

#### MODERATE: Protected Slice Enforcement Has Zero Effect in Current Run

The protected slice rules protect `B4`, `B5`, `code`, `agentic`, and `indic` domains. In the C4-only run:
- B4 has 365 documents (0.0000009% of tokens) — trivially insufficient
- B5 has zero documents
- `code`, `agentic`, `indic` domains do not exist in C4

**All protected slice logic is dead code in this run**. While the implementation is present and structurally correct, it has not been exercised or validated.

#### MODERATE: `_estimate_protected_preservation()` Returns Hardcoded Values

```python
def _estimate_protected_preservation(self):
    return ProtectedSlicesPreserved(
        B4_preservation_ratio=0.95,
        B5_preservation_ratio=0.95,
        code_preservation_ratio=0.90,
        agentic_preservation_ratio=0.90,
        indic_preservation_ratio=0.85,
    )
```

This method (lines 313–323) **always returns the target ratios as if they were achieved**, regardless of actual preservation. This is misleading — the manifest will show 95% B4 preservation when the actual number is unknowable (or trivially 0%).

**Recommendation**: Compute actual preservation ratios from selection results, or clearly annotate the manifest as containing *target* ratios rather than *achieved* ratios.


## 4. Deduplication Strategy Review

### 4.1 What Was Executed

- **Exact deduplication**: Performed at the **EMR level** (upstream, before coreset_builder.py runs), using `dropDuplicates(["hash"])` in PySpark.
- **In-pipeline exact dedup**: `engine_batched.py::_apply_batch_deduplication()` (lines 1044–1079) runs XXHash64 content-addressable dedup **per-batch only**, not globally.
- **Near-dedup**: Entirely excluded.

### 4.2 Critical Issues

#### CRITICAL: Batch-Level Dedup Is Insufficient at Scale

The batched dedup only detects duplicates **within a single batch** (typically 80,000 rows). Cross-batch duplicates are invisible. At 249M documents, this means:

- If duplicate document A appears in batch 1 and batch 50, both copies pass dedup
- The probability of cross-batch duplicates increases with the number of batches

The report claims 87.5% chunk reduction but this is primarily from **selection** (budget-constrained sampling), not deduplication. The report comingles these two mechanisms by describing chunk reduction as "deduplication impact" (Section 6), which is analytically misleading.

#### HIGH: Near-Dedup Exclusion Undermines Compression Quality

Near-dedup is where most meaningful redundancy lives in web corpora. The C4 corpus, while cleaned, still contains:
- Boilerplate-heavy pages from the same sites
- Template text with minor variations
- Re-published content across different URLs

The team's justification — *"Dolma slices are already substantially cleaned/deduplicated"* — is a citation of upstream documentation, not an empirical validation. Without running near-dedup or sampling to estimate residual similarity, the claim is unverifiable.

#### MODERATE: `find_near_duplicates()` Has O(n²) Complexity

The `NearDeduplicator.find_near_duplicates()` method (lines 217–242 of `deduplicator.py`) uses **brute-force pairwise comparison**:

```python
for i in range(len(chunk_ids)):
    for j in range(i + 1, len(chunk_ids)):
```

This is O(n²) in the number of chunks, making it **computationally infeasible** at the 249M-document scale even within batches. The production-ready alternative (LSH bucketing for SimHash/MinHash) is not implemented. If near-dedup is ever enabled at scale, this will require a complete rewrite using approximate nearest neighbor structures.


## 5. Curriculum Adherence & Band Distribution Analysis

### 5.1 The Core Problem: Curriculum Mismatch

The curriculum defines progressive difficulty across 6 bands (B0–B5) with the following stage profiles:

| Band | 1B Target | 3B Target | 8B Target | 70B Target |
|------|-----------|-----------|-----------|------------|
| B0 | 49% | 43% | 27% | 16% |
| B1 | 13% | 15% | 19% | 22% |
| B2 | 17% | 19% | 24% | 28% |
| B3 | 13% | 15% | 19% | 22% |
| B4 | 6% | 6% | 8% | 9% |
| B5 | 2% | 2% | 3% | 3% |

**Actual output distributions (from ablation report):**

| Band | 1B Actual | 3B Actual | 8B Actual | 70B Actual |
|------|-----------|-----------|-----------|------------|
| B0 | 57.29% | 5.12% | ~0% | 0.92% |
| B1 | 18.71% | 41.91% | 43.21% | 55.68% |
| B2 | 24.00% | 52.97% | 56.79% | 43.40% |
| B3 | 0% | 0% | 0% | 0% |
| B4 | 0% | 0% | 0% | 0% |
| B5 | 0% | 0% | 0% | 0% |

#### CRITICAL: Complete Absence of B3/B4/B5

**Zero tokens** were selected for bands B3, B4, and B5 across all stages. This is explained by the input data:
- C4 has only 788K B3 documents (0.79% of tokens) and 365 B4 documents (negligible)
- C4 has zero B5 documents

However, the curriculum expects 13–22% B3, 6–9% B4, and 2–3% B5 allocations. This means:
- **The entire upper difficulty spectrum is missing** — reasoning, graduate-level content, and PhD-level complexity
- The B3 + B4 + B5 deficit represents roughly 21–34% of the intended curriculum at each stage
- The curriculum-based progressive difficulty increase from 1B → 70B **cannot be validated** on C4 alone

#### CRITICAL: Band Distributions Do Not Track Curriculum Targets

Even for bands present (B0, B1, B2):
- **1B stage**: B0 is 57.29% vs target 49% (+8.29pp over-represented)
- **3B stage**: B2 is 52.97% vs target 19% (+33.97pp over-represented); B0 is 5.12% vs target 43% (-37.88pp under-represented)
- **70B stage**: B1 is 55.68% vs target 22% (+33.68pp over-represented)

These deviations vastly exceed the ±1% tolerance defined in the curriculum's validation rules. The pipeline correctly applies curriculum ratios to allocate budgets, but when the **input pool cannot satisfy the distribution**, the excess budget is redistributed to available bands. This is a reasonable fallback mechanism, but the report should **explicitly flag these deviations as curriculum violations, not successes**.

**Recommendation**: 
- Add a dedicated "Curriculum Deviation Report" section that quantifies the gap between target and actual distributions per band per stage
- Classify deviations as `CONFORMANT`, `MINOR_DEVIATION`, or `CRITICAL_VIOLATION` based on thresholds
- The current report lists band distributions without comparing against targets — this is a significant omission


## 6. Streaming & Scalability Design Review

### 6.1 Strengths

**Well-Designed Streaming Architecture**: The streaming pipeline demonstrates several best practices:

1. **SQLite-backed Non-Overlap Store** (`UsedChunksStore`): Using SQLite with WAL mode and temporary table joins for batch membership checks is pragmatic and correct for single-writer scenarios. This scales to billions of chunk IDs without memory pressure.

2. **LRU Cache Layer**: The optional in-memory LRU cache (`_used_cache`) on top of SQLite provides O(1) amortized lookups for recently processed chunks, reducing SQLite I/O. The hit-rate logging (`_used_cache_hit_rate()`) enables operators to tune cache size empirically.

3. **Batch Prefetch with Auto-Detection**: The threaded prefetch mechanism (`_iter_with_prefetch()`) overlaps I/O and compute. The auto-detection logic (checking batch size and shard-to-CPU ratio) prevents prefetch from hurting performance on small runs.

4. **Shard-Level Parallelism via Shell Script**: The `shard.sh` orchestrator cleanly manages N parallel Python processes, handles signal propagation, and collects exit statuses. The Python detection logic is robustly cross-platform.

### 6.2 Weaknesses

#### MODERATE: Shard Determinism Assumption

Each shard reads a deterministic subset of files (via file listing sorted by name and modular assignment), but the **file-level partitioning assumes that input file boundaries are fixed**. If the upstream EMR job produces different file counts or sizes between runs, shard assignments will change, breaking run-to-run reproducibility.

**Recommendation**: Document the exact input file set as part of the reproducibility manifest, or hash the sorted file listing and include it in checkpoint metadata.

#### MINOR: Single-Writer SQLite Assumption

`UsedChunksStore` uses SQLite with `PRAGMA journal_mode=WAL` which is safe for concurrent readers but only supports a single writer. If multiple shards share the same SQLite file (e.g., via NFS), data corruption is possible. The current design avoids this by creating `used_chunks_shard{shard_id:03d}.sqlite` per shard, but the cross-shard non-overlap guarantee then requires post-processing merger checks.

**Recommendation**: Add a cross-shard overlap validation tool, or document that cross-shard non-overlap depends on the input file partitioning being disjoint.


## 7. Determinism & Reproducibility Critique

### 7.1 Strengths

- **Fixed seed** (42) used for all randomized operations
- **Config hash + curriculum hash** embedded in manifests  
- **RNG state** (both Python `random` and NumPy) serialized in checkpoint state
- **Checkpoint compatibility guards** prevent resume with changed `num_shards`, `shard_id`, or `stage_target_tokens`

### 7.2 Weaknesses

#### MODERATE: Floating-Point Nondeterminism Risk

The `DESIGN_AND_RECOMMENDATIONS.md` document correctly identifies FP nondeterminism (Section 1 of Gotchas). However, the production code **does not follow its own recommendation**:
- `engine_batched.py` uses `float()` arithmetic for band ratio calculations (line 776), share computations (lines 656, 668), and budget allocations
- `_extract_band_score()` uses `float()` conversions throughout
- Band distribution normalization divides by `float(selected_tokens)` (lines 1615–1621)

None of these use the integer-arithmetic approach recommended in the design documentation. On different hardware (Intel vs ARM), these floating-point divisions may produce different rounding, which could cause occasionally different selection at bucket boundaries.

**Recommendation**: For tie-breaking and budget allocation, use integer arithmetic or fixed-point representation. At minimum, add a note in the manifest documenting the hardware architecture used for the run.

#### HIGH: Curriculum Status Is "DRAFT" Not "FROZEN"

The curriculum YAML (`config/curriculum.yaml`) states:
```yaml
status: "DRAFT"  # "FROZEN"
change_policy: "STRUCTURE_IMMUTABLE"  # "NO_CHANGES_ALLOWED"
```

The pipeline warns but does **not halt** on non-frozen curriculum (line 68–71 in `coreset_builder.py`):
```python
if not self.curriculum.validate_curriculum_frozen():
    logger.warning("Curriculum is not frozen - reproducibility may be compromised")
```

Per the curriculum's own `global_contract.enforcement.violation_action: "REJECT_OR_HALT"`, a non-frozen curriculum should halt the pipeline. The warning-only behavior contradicts the contract.

**Recommendation**: Either freeze the curriculum (`status: "FROZEN"`, `change_policy: "NO_CHANGES_ALLOWED"`) or promote the warning to an error for production runs.


## 8. Fault Tolerance & Checkpoint/Resume Review

### 8.1 Assessment: **Excellent**

This is the strongest aspect of the implementation.

- **Checkpoint metadata includes shard_id, num_shards, stage_target_tokens, and engine state** — enabling strict compatibility validation on resume
- **Engine state preservation** includes RNG states, remaining budgets per band/domain, rolling-window deque, and selected/removed chunk sets — deterministic resume is achievable
- **Crash-test hooks** via environment variables (`CORESET_SIMULATE_CRASH_AFTER_CHECKPOINT_STAGE`, `CORESET_SIMULATE_CRASH_AFTER_CHECKPOINT_BATCH`) enable controlled crash testing
- **Final checkpoint** is always written at stage-end when batches were processed but the last batch wasn't a checkpoint cadence boundary

### 8.2 Minor Issues

- The `last_checkpoint_state` is always the **last batch's state** regardless of whether a checkpoint was written. In the final flush (line 1593–1603), this is correct. However, if the process is killed between batch processing and checkpoint writing, the per-shard state could be advanced beyond the last persisted checkpoint. This is inherent to any checkpoint-based system and is acceptable.

- **`engine_state` serialization failure is silently swallowed** (line 1523–1527): If `get_checkpoint_state()` raises, the checkpoint is written without engine state. Resume from this checkpoint would work but would **not** be bitwise deterministic. This should at minimum log a warning.


## 9. Testing & Validation Coverage Assessment

### 9.1 Test Inventory

The test suite contains 13 test files:

| Test File | Focus | Assessment |
|-----------|-------|------------|
| `test_pipeline.py` (16 KB) | End-to-end pipeline runs | Good coverage of basic paths |
| `test_coreset_outputs.py` (10 KB) | Output format validation | Solid output contract tests |
| `test_optimizations.py` (13 KB) | Performance optimization validation | Good for regression detection |
| `test_checkpoint_resume_equivalence.py` (8.5 KB) | Crash + resume determinism | **Critical and well-designed** |
| `test_batch_determinism.py` (5.4 KB) | Batch ordering determinism | Addresses key correctness property |
| `test_protected_slice_domain_eligibility.py` (2.3 KB) | Protected slice with domain gating | Small but targeted |
| `test_domain_mismatch.py` (1.4 KB) | Domain eligibility edge cases | Minimal coverage |
| `test_lang_config.py` (0.5 KB) | Language configuration | Very minimal |
| `test_check_indices_determinism_tool.py` (3.7 KB) | Determinism checking tool | Tests tooling, not core logic |
| `test_merge_selected_indices.py` (2 KB) | Post-shard merge | Tests merge tool |
| `test_verify_outputs_compare.py` (3.9 KB) | Output comparison | Tests verification tooling |
| `test_parquet_optional_columns.py` (1.5 KB) | Missing column handling | Small edge case test |

### 9.2 Gaps

#### HIGH: No Multi-Source / Multi-Domain Integration Test

All tests appear to use synthetic data or C4-only data. There is no integration test that validates the pipeline with:
- Multiple sources (e.g., C4 + books + ArXiv)
- Multiple domains (web + code + science + math)
- Multiple languages (en + Indic languages)
- Multiple active bands (B0–B5)

This is the **single highest priority testing gap** given the pipeline's purpose.

#### HIGH: No Curriculum Violation Detection Test

There is no test that verifies the pipeline correctly **detects and reports** when output distributions deviate from curriculum targets. Given the critical deviations observed in the C4 run, this is essential.

#### MODERATE: No Near-Dedup Integration Test

Since near-dedup was excluded, there is no validated integration path for enabling it. The `deduplicator.py` has O(n²) `find_near_duplicates()` which will fail at scale, but no test exercises it at realistic sizes.


## 10. Report & Metrics Integrity Audit

### 10.1 Token Accounting

The report states:
- **136,932,109,554** — C4 post-dedup corpus total
- **458,727,376,426** — cumulative stage exposure (sum of per-stage input totals)
- **81,560,691,927** — selected tokens

**Arithmetic verification:**

Sum of stage inputs: 136,932,109,554 + 125,162,236,032 + 113,649,882,764 + 82,983,148,076 = **458,727,376,426** ✓

The per-stage inputs form a decreasing chain:
- 136.9B → 125.2B (−11.7B, reflects 1B selection) 
- 125.2B → 113.6B (−11.5B, reflects 3B selection)
- 113.6B → 83.0B (−30.7B, reflects 8B selection)

**These check out**: each stage's input = prior stage's input − prior stage's selected tokens. ✓

Sum of selected: 11,769,873,522 + 11,512,353,268 + 30,666,734,688 + 27,611,730,449 = **81,560,691,927** ✓

### 10.2 Compression Ratio Framing

#### CRITICAL: Misleading Compression Ratio Denominator

The report's headline "5.62x compression" uses **458B cumulative stage exposure** as the denominator. This inflates the perceived compression by counting the same tokens multiple times (the 70B stage sees tokens that already passed through 1B, 3B, and 8B eligibility windows).

The **actual single-pass compression** is:
- 136.9B input / 81.6B output = **1.68x** compression

This is a dramatically different narrative:
- "5.62x compression" suggests aggressive data curation
- "1.68x compression" reveals the pipeline is selecting approximately 60% of available tokens

Neither is wrong, but the choice of denominator fundamentally changes the interpretation. **The report should prominently display both metrics** and clearly label which is the single-pass compression and which is the cumulative exposure compression.

### 10.3 Stage-Wise Compression Paradox

| Stage | Compression | What It Means |
|-------|-------------|---------------|
| 1B | 11.63x | Selects 8.6% of C4 — very selective |
| 3B | 10.87x | Selects 9.2% of remaining — still very selective |
| 8B | 3.71x | Selects 27% of remaining — moderately selective |
| 70B | 3.01x | Selects 33.3% of remaining — takes a third of what's left |

The compression drops steeply from 1B to 70B. This is **expected** (larger stages need more data), but it means the 70B coreset is essentially "everything B1/B2 that passed earlier stages" rather than a curated subset. **For the 70B stage, the coreset is primarily a residual after earlier stage removals, not an active quality-based selection.**

### 10.4 Section 6: "Deduplication Impact" Misattribution

Section 6 claims:
> Chunks removed / excluded in final coreset selection: **784,626,370** (896,870,901 - 112,244,531)
> **87.5% chunk reduction** and **7.99x chunk compression**

This frames the entire chunk reduction as "deduplication impact." In reality, most of the reduction is from **budget-constrained selection** (the pipeline selects chunks until token budgets are met, then stops). The actual deduplication impact is the EMR-level hash dedup, which is not quantified separately in this section.

**Recommendation**: Separate "chunks excluded by deduplication" from "chunks not selected due to budget exhaustion" for accurate attribution.


## 11. Code Quality & Maintainability Review

### 11.1 Positive Observations

- **Consistent type annotations** throughout the codebase
- **Comprehensive docstrings** on public methods
- **Logging at appropriate granularity** — info for batch progress, warning for non-critical issues, error for failures
- **Backward compatibility patterns** (e.g., dual `token_count` / `token_count_estimate` field emission)
- **Cross-platform considerations** in `shard.sh` (Windows/Git Bash Python detection)

### 11.2 Issues

#### HIGH: `coreset_builder.py` Line Count (2,030 lines)

As discussed in Section 2.2. This single file contains more logic than all of `src/` combined in some dimensions. Refactoring is overdue.

#### MODERATE: Defensive `getattr()` Overuse

The codebase extensively uses `getattr(obj, "field", default)` even where the object's type is known and the field is part of the dataclass definition. Example:
```python
getattr(stage_bands.band_ratios, "B4", 0.0) > 0
```

If `band_ratios` is a `BandDistribution` dataclass (which it is), `stage_bands.band_ratios.B4` is always valid. The defensive `getattr` suggests either past schema instability or an overly cautious coding style. It obscures intent and makes static analysis harder.

#### MODERATE: Silent Exception Swallowing

Multiple locations catch broad `Exception` and continue silently:
- `_extract_band_score._to_float()` (lines 694–700)
- `_extract_band_from_band_p._to_float()` (lines 755–761)
- `_infer_band_from_score()` error handling (lines 631–634, 669–672)
- Row parsing catch-all (lines 1370–1377)

While per-row resilience is valuable at 2T scale (one malformed row shouldn't crash the pipeline), the **count of silently-handled exceptions** should be tracked and surfaced in the stage manifest to detect systemic data quality issues.

#### MINOR: Multiple Curriculum YAML Versions in Config

The `config/` directory contains:
- `curriculum.yaml` (current)
- `curriculum_old.yaml`
- `curriculum - 0.4.yaml`
- `curriculum - 0.6.yaml`

These should be version-controlled or removed. Having multiple undifferentiated versions is a configuration management anti-pattern.


## 12. Risk Register & Failure Mode Analysis

| Risk | Severity | Likelihood | Impact | Mitigation Status |
|------|----------|------------|--------|-------------------|
| **Multi-source run reveals curriculum infeasibility** | CRITICAL | HIGH | Band targets impossible across heterogeneous sources | ❌ Not mitigated — only C4 tested |
| **Near-dedup exclusion causes redundancy in coreset** | HIGH | MEDIUM | Model sees near-duplicate content, degrading training efficiency | ⚠️ Deferred, justified but unverified |
| **Protected slices (B4/B5/code) absent at production scale** | HIGH | MEDIUM | Model lacks reasoning/code capability emergence data | ❌ Protected slice enforcement untested |
| **Curriculum remains DRAFT status in production** | HIGH | HIGH | Curriculum changes between runs, breaking reproducibility | ⚠️ Warning only, no enforcement |
| **FP nondeterminism across hardware** | MODERATE | LOW | Different selections on Intel vs ARM | ⚠️ Documented but not mitigated |
| **Cross-shard overlap in output** | MODERATE | LOW | Duplicated chunks across shard outputs | ⚠️ Per-shard SQLite; no cross-shard check |
| **EC2 Spot interruptions** | MODERATE | HIGH | Extended runtime, wasted compute | ✅ Checkpoint/resume handles this |
| **SQLite corruption under NFS** | LOW | LOW | Used-chunk store loses consistency | ✅ Per-shard DB design avoids this |


## 13. Production Readiness Assessment

This section evaluates the pipeline's readiness for production deployment at full 2T→400B scale, cross-referencing against the conditions identified in the Go/No-Go review.

### 13.1 Production Readiness Scorecard

| Dimension | Status | Verdict |
|-----------|--------|--------|
| **Operational Tooling** | ⚠️ Partial | Validation tool exists; no production run checklist or runbook |
| **CI/CD Pipeline** | ✅ Present | GitHub Actions workflow with self-hosted and SSH modes |
| **Observability & Monitoring** | 🔴 Missing | No metrics export, no alerting, no dashboard integration |
| **Data Contract & Schema** | 🔴 Missing | No formal upstream schema enforcement; behavior on missing columns is ad-hoc |
| **Infrastructure Dependencies** | ⚠️ Open | EMR + EC2 capacity not confirmed; EBS sizing documented but not provisioned |
| **Dependency Management** | ⚠️ Risky | Unpinned `requirements.txt`; `torch>=2.0.0` adds ~2GB for a CPU-only pipeline |
| **Learning-Quality Validation** | 🔴 Missing | Zero convergence/benchmark evidence that coreset preserves learning quality |
| **Production Run Playbook** | 🔴 Missing | No step-by-step checklist for operators |
| **Security & Secrets** | ✅ Sound | GitHub Secrets for S3/SSH; IAM Role-based auth; no hardcoded credentials |

### 13.2 Detailed Assessment

#### 13.2.1 Operational Tooling — Partial ⚠️

**What exists (good):**
- `tools/validate_coreset_outputs.py` — 1,140-line validator with 20 checks per stage across 8 categories
- `tools/check_stage_overlaps.py` — Cross-stage overlap checker (reads `selected_indices.jsonl` per stage, reports pairwise overlaps)
- `tools/estimate_total_tokens.py` — Computes `total_input_tokens_estimate` for a given input path
- `tools/merge_sharded_ablation_reports.py` and `tools/merge_sharded_manifests.py` — Post-run shard merging
- `tools/verify_batch_determinism.py` — Validates that runs produce identical outputs
- `tools/check_indices_determinism.py` — Compares selected indices across runs

**What is missing (gaps):**
- **No production run checklist**: `commands.sh` handles deployment automation, but there is no operator-facing checklist covering pre-flight validation (curriculum frozen? total_tokens verified? checkpoint-dir unique per shard? used_chunks store preserved?)
- **No failure-conditions runbook**: The T3 report lists a "failure conditions watchlist" but there is no actionable runbook mapping each condition to a tool invocation and escalation path
- **No post-run validation gate**: The pipeline does not automatically invoke `validate_coreset_outputs.py` after completion. It should be a mandatory post-step in `commands.sh` or `shard.sh`, with non-zero exit on critical failures
- **Cross-shard overlap validation is incomplete**: `check_stage_overlaps.py` checks cross-*stage* overlaps, but there is no tool to check cross-*shard* overlaps within the same stage. Each shard writes its own `selected_indices`, but nothing verifies that shard 0 and shard 1 don't accidentally select the same chunk when input partitioning changes

#### 13.2.2 CI/CD Pipeline — Present ✅

**Strengths:**
- `.github/coreset_engine.yml` provides a `workflow_dispatch`-triggered pipeline with parameterized inputs (num_shards, stages, input_path, total_tokens, resume)
- Two deployment modes: self-hosted EC2 runner and SSH-to-EC2
- Unit tests run as a prerequisite job before deployment
- `commands.sh` supports `--dry-run`, `--foreground`, and `--skip-repo-setup` flags

**Gaps:**
- **No integration test job** in CI: only unit tests run (`uv run pytest coreset_engine_v5/tests/`). There is no CI step that validates a small-scale end-to-end coreset build
- **No artifact upload**: The CI workflow runs the pipeline but does not upload manifests, validation reports, or logs as GitHub Actions artifacts for post-run review
- **No post-pipeline validation step**: After `commands.sh` completes, there should be a step that runs `validate_coreset_outputs.py` and fails the pipeline if critical issues are detected
- **No rollback mechanism**: If a production run produces a bad coreset, there is no documented process for reverting to the previous known-good coreset

#### 13.2.3 Observability & Monitoring — Missing 🔴

**Current state:**
- The pipeline writes to `coreset_selection.log` and stdout — standard Python logging at INFO/WARNING/ERROR levels
- Per-batch progress logging and LRU cache hit-rate logging exist
- No metrics export, no structured logging (JSON), no external observability integration

**What a production pipeline needs:**
- **Structured JSON logging** for machine-parsable log aggregation (e.g., CloudWatch, Datadog)
- **Metrics export**: tokens_processed, chunks_selected, chunks_deduplicated, batch_processing_time, cache_hit_rate — as time-series metrics (CloudWatch Metrics, Prometheus)
- **Alerting**: Trigger alerts on curriculum deviation > threshold, stage completion failure, checkpoint corruption, or silent exception spike
- **Progress tracking**: For a 12–14h production run, operators need a dashboard showing estimated completion percentage, tokens processed vs. target, and current stage/batch

#### 13.2.4 Data Contract & Upstream Schema — Missing 🔴

**Current state:**
- The pipeline accepts Parquet or JSONL input with flexible column handling. It uses `getattr` and `get()` with defaults everywhere, silently falling back when columns are absent
- `schemas/integration_schema.json` exists but is not enforced at runtime
- No input schema validation step runs before or during pipeline execution

**What is needed:**
- **Schema validation gate**: Before processing, verify that input files contain required columns (`chunk_id`, `band`, `domain`, `language`, `token_count`) and log which optional columns are present (`band_score`, `difficulty_score`, `band_p_B0..B5`)
- **One-page data contract**: Document the expected upstream schema, listing required vs. optional fields and how the pipeline behaves when optional fields are absent (e.g., "scoring falls back to diversity + chunk_id")
- **Behavior on absent scoring columns**: Currently undocumented at the system level. The Go/No-Go review specifically flags this as a condition for full GO (§5.3)

#### 13.2.5 `total_input_tokens_estimate` Correctness — Caution ⚠️

The streaming budget formula is `stage_target × (batch_tokens_raw / total_input_tokens_estimate)`. An incorrect total **silently skews** all per-batch budgets:
- **Total too high** → systematic under-selection (pipeline ends before filling budgets)
- **Total too low** → systematic over-selection (early batches consume too much quota)

`tools/estimate_total_tokens.py` exists to compute this value, but:
- There is no enforcement that the estimate was recomputed for the current input path
- The value is not automatically validated against the actual input size at startup
- The Go/No-Go review lists this as a **Caution** condition (§5.4)

**Recommendation**: At pipeline startup, sample the first N files to estimate total tokens and compare against the provided `total_input_tokens_estimate`. Log a `WARNING` if the deviation exceeds 10%, `ERROR` if it exceeds 25%.

#### 13.2.6 Dependency Management — Risky ⚠️

**Current state (`requirements.txt`):**
```
numpy>=1.24.0
torch>=2.0.0
pyyaml>=6.0
...
ray>=2.10.0
faiss-cpu>=1.7.4
```

**Issues:**
- **All dependencies are unpinned** (minimum-version only). In production, a `pip install` could pull a breaking update to any dependency. The `uv sync` path via `pyproject.toml` may have a lockfile, but `requirements.txt` does not
- **`torch>=2.0.0` is a ~2GB dependency** for a pipeline that runs on CPU only. The codebase doesn't appear to use PyTorch at runtime in the streaming path. If it's only needed for `torchhash`, consider replacing it with a lightweight alternative
- **`ray>=2.10.0`** is listed but not used by the core pipeline (no `import ray` in `coreset_builder.py` or `src/`). Removing unused heavy dependencies reduces install time and surface area
- **`faiss-cpu>=1.7.4`** — similarly not imported in the core pipeline path. May be used by auxiliary tools but should not be a core requirement

**Recommendation**: Create a `requirements-core.txt` (minimal dependencies for production pipeline) and `requirements-dev.txt` (adds tools, testing, and optional heavy deps). Pin all production dependencies via a lockfile.

#### 13.2.7 Learning-Quality Validation — Missing 🔴

The Go/No-Go review (§5.6) identifies this as a key condition:
> "To validate quality, we need at least one comparison where the only change is the dataset (coreset vs full or random), with the same model size and setup — then compare training loss over time and/or benchmark scores."

**Current state:** Zero convergence or benchmark evidence exists. The T3 report ties "efficiency without learning degradation" to compression metrics, not downstream model quality. 

**Impact:** Without this evidence, the coreset is an **unvalidated hypothesis** — it may select data that achieves 1.68x compression but degrades model quality compared to training on the full corpus or a random subsample.

**Recommendation**: Before full-scale production, run at least one controlled comparison:
1. Train a small proxy model (e.g., 125M or 350M params) on a 10% subset of the 1B coreset
2. Train the same model on a 10% random sample of C4
3. Compare: training loss curves and at least one benchmark (e.g., HellaSwag, ARC-Easy)
4. Publish a short "Coreset Quality Validation" note in `docs/reports/`

If time does not permit, explicitly document: *"Learning-quality validation is deferred to Phase 2"* as a risk acceptance.

#### 13.2.8 Production Run Playbook — Missing 🔴

`commands.sh` automates the deployment, but there is no **operator-facing production run playbook**. At 2T→400B scale with 12–14h runtimes and multiple shards, operators need:

1. **Pre-flight checklist**:
   - [ ] Curriculum status is `FROZEN` (not DRAFT)
   - [ ] `total_input_tokens_estimate` recomputed for current input path via `tools/estimate_total_tokens.py`
   - [ ] Unique `--checkpoint-dir` per shard (e.g., `output/checkpoints_1B/shard000`)
   - [ ] `.used_chunks/` directory preserved from previous stages
   - [ ] EC2 instance type confirmed (c7gd.16xlarge)
   - [ ] EBS volume ~600GB attached and mounted
   - [ ] On-demand vs. Spot decision documented
   - [ ] `--num-shards` / `--shard-id` / `--stage-target-scale` unchanged from last checkpoint (or fresh `--checkpoint-dir` if changed)

2. **During-run monitoring**:
   - Tail `shard_run.log` or individual shard logs
   - Watch for `WARNING` (non-critical) vs `ERROR` (requires action) patterns
   - Monitor checkpoint cadence (expect checkpoint every N batches)
   - Check EBS disk usage periodically (large parquet outputs)

3. **Post-run validation**:
   - Run `tools/validate_coreset_outputs.py --stages 1B 3B 8B 70B --format both`
   - Run `tools/check_stage_overlaps.py` to verify cross-stage disjointness
   - Verify manifest hashes and `total_input_tokens_estimate` in output manifests
   - Review band/domain/language distributions in validation reports
   - Archive manifests and validation reports to S3

4. **Failure recovery**:
   - If a shard dies: restart with `--resume` using the same `--checkpoint-dir`
   - If Spot termination: same `--resume` flow; prefer on-demand for production
   - If checkpoint corruption: delete corrupt checkpoint file, restart from the previous checkpoint batch
   - If curriculum deviation > threshold: escalate to Curriculum Architects team

#### 13.2.9 Security — Sound ✅

- GitHub Secrets used for `S3_BUCKET`, `EC2_SSH_KEY`, `EC2_HOST`, `EC2_USER` — no hardcoded credentials
- SSH key cleaned up in `always()` step
- IAM Role-based S3 access (no access keys in code)
- `set +x` in SSH steps prevents credential leakage in logs
- No external API calls or network dependencies beyond S3

#### 13.2.10 Operational Reliability (Spot, Long Runs) — Caution ⚠️

**Finding:** The T3 report documents **~20 interruption/restart events** on a ~10 GB C4 run using EC2 Spot instances, resulting in nearly a full day of wall-clock time for what should have been a much shorter computation. Checkpoint/resume handled every interruption correctly, but the operational cost (human monitoring, extended runtime, wasted partial-batch compute) was significant.

**At 2T→400B scale** (estimated 12–14h on-demand), Spot-based runs could face:
- Multiple Spot terminations within a single stage, each requiring ~5 min restart overhead (re-loading checkpoint, re-initializing SQLite, re-reading file listing)
- Potential for cascading delays if all shards are Spot and terminate near-simultaneously
- Extended wall-clock time making it harder to meet delivery timelines

**Recommendation:**
- **Prefer on-demand instances for production coreset runs** (as recommended in the project's own `INFRA_RECOMMENDATIONS_DECISION_MATRIX.md`)
- If Spot is used, document the chosen strategy (e.g., "Spot with 3x retry budget") in the run playbook and set a wall-clock timeout after which the run is escalated
- Consider using a mix: on-demand for shard 0 (critical path), Spot for remaining shards

#### 13.2.11 Additional Improvements (Non-Blocking)

The following are non-blocking improvements identified during the review that would strengthen operational confidence:

1. **A/B prefetch and batch-size benchmarking**: Run a controlled comparison of `prefetch=auto` vs. `prefetch=off` on a representative data subset to quantify I/O-compute overlap gains. Record throughput (chunks/sec, tokens/sec) and stability metrics. This would validate the auto-detection heuristic and inform production batch_size tuning.

2. **"Availability-limited" interpretation note**: When input data has few or no chunks for a band/domain (e.g., C4 has 365 B4 documents), the validator should label target/ratio shortfalls as "availability-limited" rather than "pipeline failure." Add a short note to the report template or `REPORT_GENERATION_GUIDE.md` clarifying this distinction, so downstream reviewers understand that "B4 target not met" means "not enough B4 data in C4" rather than a pipeline bug.

3. **B4/B5 and domain coverage in manifests**: Ensure each stage manifest includes `availability_stats` — how much eligible data remained per band/domain — so shortfalls can be attributed to data availability vs. pipeline logic.

4. **Scoring and token_ids documentation note**: In production streaming mode, `token_ids` (tokenizer output per chunk) are typically absent; token-level rarity scoring in the `DiversityScorer` is then skipped and a length-based proxy is used. Add a one-sentence note to the T3 report or `REPORT_GENERATION_GUIDE.md` that scoring is metadata/column-driven and token-level rarity applies only when `token_ids` are present, to prevent future confusion about scoring effectiveness.

### 13.3 Production Readiness — Go/No-Go Cross-Reference

The Go/No-Go review (v4) evaluates 11 criteria for full production readiness. Here is how the codebase and this review address each:

| Go/No-Go Criterion | Ref | Go/No-Go Status | Critical Review Assessment | Gap | T3 Comments/Evidences |
|--------------------|-----|-----------------|---------------------------|-----|-----------------------|
| Report accuracy & completeness | §7 | **Pass** | Numerically sound; framing issues identified (§10) | Compression ratio framing (P0-2) | T3 Report §Token Accounting Context explicitly distinguishes single-pass corpus (136.9B) from cumulative stage exposure (458.7B). Arithmetic verified: per-stage inputs form correct decreasing chain. Section 6 "Deduplication Impact" conflates selection reduction with dedup — needs reframing. |
| Charter/deliverables alignment (executed scope) | §3, §7 | **Pass** | Aligned for C4-only scope; untested for full charter | Multi-source validation needed (P0-1) | T3 Report §Source Scope Note: execution explicitly scoped to `source=C4` due to EMR capacity. Deliverables section confirms pipeline is implemented for full scope (curriculum loading, protected slices, multi-stage). NCERT small-dataset validation also executed on EC2 Spot. Charter's ~400B target acknowledged as "program-level" pending production config. |
| Determinism & reproducibility | §4.2, §7 | **Pass** | Strong seeding/checkpoint controls; FP risk remains (§7) | Freeze curriculum (P0-3) | T3 Report §Determinism: fixed seed, config/curriculum hashes in manifests, checkpoint compatibility guards, engine state persisted for deterministic resume. |
| Infra & access (EMR, EC2, EBS) | §5.1 | **Open** | `commands.sh` handles EC2 setup; no EMR orchestration (§13.2.8) | Infra not confirmed | T3 Report §Current State: "coordinate with AWS admin to provision and validate Team 3 access.AWS EC2 instance "c7gd.16xlarge for 24-48 hours, 1 TB EBS with higher IOPS documented. EMR target: dedup+chunk+stats on ~2 TB input in ~4–5h. Full infra depends on AWS admin scheduling. `INFRA_DESIGN.md` and `validate_infra.sh` created for pre-flight checks. |
| Token accounting & data contract | §5.2 | **Open** | No formal data contract document; token accounting is stats-based (§13.2.4) | No "Token Accounting Scope" doc (P1-15) | T3 Report §Known Constraints: "Team 3 received data without token-level allocation and rejection summary tables." Stats-level logic was added to derive per-band token signals after EMR dedup. T3 states "final authoritative token-level rejection statistics remain an upstream dependency." No formal data contract document exists. (Link to [T3 Report](https://github.com/The-School-of-AI/LLM/blob/p3/feat/stage-wise-coreset-selection_v2/experiments/3_coreset_engineering/coreset_engine_v5/docs/reports/2026-02-23_T3_REPORT.md#input-distribution-csv-backed)) Total Tokens Aggregated at a Source Level Provides Token Accounting for the coresets process. Token verification report is generated by tool validate_corset_outputs.py. sample - [Verification Report](https://github.com/The-School-of-AI/LLM/blob/p3/feat/stage-wise-coreset-selection_v2/experiments/3_coreset_engineering/coreset_engine_v5/validation_output.txt)) . Upstream Data Contract file has been added [Upstream_data_contract](https://github.com/The-School-of-AI/LLM/blob/p3/feat/stage-wise-coreset-selection_v2/experiments/3_coreset_engineering/coreset_engine_v5/docs/UPSTREAM_DATA_CONTRACT.md)|
| Upstream scoring schema (band_score/difficulty_score) | §5.3 | **Open** | `getattr` fallbacks everywhere; no schema validation at load (§3.2, §13.2.4) | No schema enforcement (P2-22) | T3 Report §Pipeline: "Band inference + scoring (column-driven)" — uses `band_score`, `difficulty_score`, or `band_p_*` columns when present. Token-level rarity "only applies when tokenizer artifacts like `token_ids` exist; otherwise it is skipped by design." No runtime schema validation gate; behavior on missing columns is ad-hoc `getattr` fallbacks. |
| `total_input_tokens_estimate` verification | §5.4 | **Caution** | `tools/estimate_total_tokens.py` exists; no auto-verification at startup (§13.2.5) | Manual step, not enforced (P1-10) | T3 Run Config: `--total-tokens 136932109554` explicitly passed. T3 Report §Token Accounting: value sourced from "C4 post-dedup corpus total from T3 EMR stats." `tools/estimate_total_tokens.py` available but not auto-invoked at pipeline startup. Manual step — no enforcement or deviation check. |
| Checkpoint & used_chunks ops discipline | §5.8 | **Caution** | Code handles per-shard dirs; no operator checklist (§8, §13.2.8) | Needs playbook (P0-5) | T3 Report §Determinism: "Checkpoint compatibility guards on resume (shards, shard-id, stage target). Persistent non-overlap store ensures stage disjointness." Run Config shows `--checkpoint-every-n-batches 50`. §Known Constraints: ~20 Spot interruption/restarts — checkpoint/resume handled every one. No operator-facing playbook or pre-flight checklist documented. |
| Near-dedup & B4/B5 coverage | §5.5 | **Deferred** | Deferred — documented as Phase 2 (§4) | Accepted deferral | T3 Report §Known Constraints: near-dedup excluded due to data size/compute constraints and Dolma upstream dedup claims. §Coverage: C4 only has web data and web as a domain is not allowed in B4/B5 (this is as per the curriculum band policies). (Link: [curriculum.yaml](https://github.com/The-School-of-AI/LLM/blob/p3/feat/stage-wise-coreset-selection_v2/experiments/3_coreset_engineering/coreset_engine_v5/config/curriculum.yaml)) |
| Convergence / benchmark validation | §5.6 | **Pending** | Zero evidence (§13.2.7) | **Largest gap** (P2-16) | T3 Report §Efficiency: "Efficiency evidence is positive from compression and throughput behavior. Final learning-quality confirmation remains benchmark-dependent." §Pending Final Sign-off: "Convergence comparison vs full-data training." No proxy model training or benchmark comparison has been executed. This is part of training teams charter |
| Operational reliability (Spot, long runs) | §5.7 | **Caution** | ~20 restarts on 10GB Spot run documented; on-demand preferred (§13.2.10) | Needs documented strategy | T3 Report §Known Constraints: "~10 GB run experienced at least 20 interruption/restart events and required nearly a full day due to spot termination behavior." §Next Milestones: "provision on-demand EC2 capacity for ~24–48 hours." Team prefers on-demand for production. |

### 13.4 Production Readiness Audit

**Status: Conditional GO for current scope; NOT READY for full-scale 2T→400B production.**

The pipeline's **engineering foundation** (streaming, checkpointing, sharding, determinism) is production-grade. However, the **operational wrapper** — monitoring, runbooks, schema enforcement, dependency hygiene, and learning-quality evidence — is insufficient for a 12–14h, multi-shard production run at 2T→400B scale without significant operational risk.

**To reach full production readiness, the following must be addressed:**

1. Create production run playbook (§13.2.8)
2. Add post-pipeline validation gate to `commands.sh` (auto-run `validate_coreset_outputs.py`)
3. Add structured logging or metrics export for observability (§13.2.3)
4. Validate or explicitly accept `total_input_tokens_estimate` at startup (§13.2.5)
5. Pin production dependencies and remove unused heavy deps (§13.2.6)
6. Produce at least one convergence/benchmark check or explicitly defer with documented risk acceptance (§13.2.7)
7. Document upstream data contract (§13.2.4)


## 14. Summary & Prioritized Recommendations

### Summary

The coreset engineering pipeline is a **well-engineered system with strong operational foundations** (streaming, fault tolerance, determinism controls) but it has been **validated only under trivially simple conditions** that exercise less than half of its design features. The T3 report accurately presents numerical results but makes analytical claims (compression ratios, deduplication impact, curriculum adherence) that are either **misleading in framing** or **untestable under the current scope**.

From a **production readiness** standpoint (§13), the engineering core is solid but the operational envelope — monitoring, runbooks, schema enforcement, dependency management, and downstream quality evidence — has significant gaps that must be closed before deploying at full 2T→400B scale.

### Prioritized Recommendations

#### P0 — Must Fix Before Production

1. **Run multi-source validation** (C4 + at least one STEM source + one code source) to exercise B3/B4/B5 bands, multi-domain stratification, and protected slice enforcement
2. **Correct compression ratio framing** in the report: prominently display single-pass compression (1.68x) alongside cumulative exposure compression (5.62x)
3. **Freeze curriculum** or enforce `REJECT_OR_HALT` on DRAFT status
4. **Add curriculum deviation section** to report with per-band per-stage target-vs-actual comparison
5. **Create production run playbook** with pre-flight checklist, monitoring guidance, post-run validation steps, and failure recovery procedures
6. **Add post-pipeline validation gate** — auto-run `validate_coreset_outputs.py` at pipeline end; fail on critical issues

#### P1 — Should Fix Before Scale-Up

1. **Implement cross-shard overlap validation** tool to verify disjointness across shard outputs within the same stage
2. **Compute actual preservation ratios** in `_estimate_protected_preservation()` instead of returning target values
3. **Add structured logging or metrics export** for CloudWatch/Datadog integration during long-running production runs
4. **Auto-verify `total_input_tokens_estimate`** at startup by sampling input files; warn if deviation > 10%
5. **Pin production dependencies** via lockfile; split `requirements.txt` into core vs. dev; remove unused heavy deps (`torch`, `ray`, `faiss-cpu`)
6. **Refactor `_build_stage_coreset()`** into smaller, testable methods (row parsing, output writing, manifest building)
7. **Add integration test** with multi-source, multi-domain, multi-language synthetic dataset
8. **Track and surface silent exception counts** per stage in the manifest
9. **Document upstream data contract** — required vs. optional input schema and fallback behavior

#### P2 — Should Fix for Long-Term Health

1. **Run at least one convergence/benchmark comparison** (coreset vs. random baseline); publish results or explicitly document deferral as risk acceptance
2. **Replace O(n²) near-dedup** with LSH-based approximate algorithm before enabling at scale
3. **Use integer arithmetic** for scoring tie-breaks and budget allocation to guarantee cross-hardware determinism
4. **Remove stale curriculum versions** from `config/` directory
5. **Extract `coreset_builder.py`** into smaller modules (parser, writer, checkpoint validator)
6. **Add sampling-based near-dedup estimation** to quantify residual redundancy in C4 without running full near-dedup
7. **Add input schema validation gate** — verify required columns exist before processing begins

## Meeting Notes

### (25-02-2026) Audience - Pankaj, Sid, Balaji, Varsha, Smita, Sualeh

### Key Points Discussed

1. Discussion over major gaps and open points highlighted in the report
2. P3 Working Team covered the entire process and design of coresets
3. Questions were asked on the report
4. Input datasets (T1/T2) and limitations were explained by P3 working team
5. EMR script go through by Balaji
6. Sharding process was covered and explained in detail by Pankaj
7. Token Allocation and Rejection strategy was explained
8. Band Inference using probability and difficulty scores was explained to the team by Sid
9. Infra level challenges were discussed and solutions worked upon were explained
10. Problems associated with data size and compute constraints were discussed
11. Why C4 was choosen was covered and explained focusing on the distribution of data

### Outcomes

1. Share the output structure with evidences on the process (from local disk not actual S3 structure, S3 structure would resemble this with different buckets and prefixes) - Pankaj (Link: [Sample Representation of Output Structure on S3 from T1/T2/T3](https://github.com/The-School-of-AI/LLM/blob/p3/feat/stage-wise-coreset-selection_v2/experiments/3_coreset_engineering/coreset_engine_v5/docs/reports/T1_T2_T3_sample_output_structure.png))
2. Tokenization team is trying to use the T3 dataset for there validations and process
3. Entire coresets pipeline was covered and explained with sharding, batch streaming, checkpointing, logging, sqllite DB, deterministic processing, etc.
4. Curriculum policies were explained once again
5. Deterministic processing was discussed and confirmed
6. Infra level challenges were discussed and solutions worked upon were explained
