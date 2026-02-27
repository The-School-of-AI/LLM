# T3 Coreset Engineering — Go/No-Go Review Report

**Review type:** Independent expert validation of the 2026-02-23 T3 report and underlying pipeline.  
**Scope:** Readiness for production use and for scaling to full charter scope.  
**Review date:** 2026-02-24.  

---

## 1. Executive Summary

**Verdict: Conditional GO.**

The T3 report and the coreset engine it describes are technically sound and aligned with the charter for the scope that was run: C4 only, exact dedup only, pretraining stages 1B–3B–8B–70B. The pipeline is deterministic (same config, input, and seed yield the same selected indices), respects the curriculum (e.g. band mix per stage), and correctly enforces non-overlap (no chunk in more than one stage) and protected-slice rules (B4/B5, code, agentic, Indic). 

A full, unconditional GO for production at full 2T→400B scale may have to wait until: (1) infra and access are unblocked (EMR and EC2 capacity, permissions), (2) token-accounting and data-contract gaps are closed—i.e. either we have a clear end-to-end count of tokens at each step and an agreed input/output schema between teams, or we document current limitations, (3) upstream scoring fields (band_score / difficulty_score) are specified and either validated or explicitly accepted as optional, and (4) at least one controlled comparison exists: same model and setup, only the dataset differs (coreset vs full or random), with convergence (loss over time) and/or benchmark scores compared to show the coreset does not hurt quality.

---

## 2. What Was Reviewed

- **Primary:** `docs/reports/2026-02-23_T3_REPORT.md` (content, claims, alignment with charter and REPORT_GENERATION_GUIDE).
- **Codebase:** `coreset_builder.py`, `src/selection/engine_batched.py`, `src/selection/engine.py`, `src/diversity/scorer.py`, `src/curriculum/loader.py`, `src/io/used_chunks_store.py`, `src/io/loaders.py`, `src/io/batch_processor.py`, configs (`pipeline.yaml`, `curriculum.yaml`), README, DELIVERABLES, DESIGN_AND_RECOMMENDATIONS.
- **Companion:** CORESET_ENGINEERING_PROCESS_GUIDE.md (score logic, band targets, token accounting, upstream/downstream).
- **Cross-checks:** DELIVERABLES.md, REPORT_GENERATION_GUIDE.md, INFRA_RECOMMENDATIONS_DECISION_MATRIX.md.

This review does not re-run the pipeline or validate numerical results against raw data; it assesses consistency of the report with the code, completeness of documentation, and readiness for next steps.

---

## 3. Charter and Deliverables Alignment

| Charter expectation | Status | Notes |
|---------------------|--------|-------|
| Deterministic, configurable coreset selection | **Met** | A fixed seed plus the same config and input produce the same indices; config and curriculum hashes in manifests allow verification of which config was used. Pipeline and curriculum YAML control stages, targets, dedup, and diversity. Checkpoint logic blocks resume when shard count, stage target, or scale change. |
| Curriculum-aligned, stage-specific coresets (1B/3B/8B/70B) | **Met** | Each stage has its own token target and band mix from the curriculum; the engine builds the selected chunk set per stage and enforces band/domain/language rules (e.g. B0=49% for 1B). |
| Manifests + ablation/validation reports | **Met** | Per-stage manifests (JSON with target/actual tokens, composition, seed, hashes) and ablation/validation report structure match REPORT_GENERATION_GUIDE. Ablation runs turn off or change one ingredient to measure impact. |
| Exact + optional near-dedup, stratified selection, protected slices | **Partially met** | Exact dedup (same-content chunks removed) is in the pipeline; near-dedup (similar-but-not-identical content) was deferred and is documented. Chunks are grouped by (band, domain) and each group is filled up to its token target; B4/B5, code, agentic, and Indic are kept to minimum ratios. |
| Non-overlap across stages | **Met** | A SQLite store records chunk IDs already selected in earlier stages; when building 3B/8B/70B those chunks are skipped so no chunk appears in two stages. |
| CPU-first streaming, batching, checkpoint/resume | **Met** | Data is read in batches (e.g. 80k chunks) rather than all at once; after every N batches, state is saved so the run can resume after a stop. Multiple workers (shards) each process a subset of data via `shard.sh`. |
| Metadata-driven scoring, configurable band-inference | **Met** | Order within each (band, domain) bucket uses upstream columns (e.g. band_score, difficulty_score) plus a diversity score. When a chunk’s (band, domain) is not allowed by the curriculum, the engine can reassign band from a score column; this is configurable via CLI. |
| Runnable builder, modular layout, production + ablation configs | **Met** | `coreset_builder.py` is the main entry; `src/` is split into selection, dedup, curriculum, io. Production and ablation configs (e.g. pipeline.yaml, ablation_no_neardup.yaml) are in place. |

**Conclusion:** The report and implementation satisfy the charter for the executed scope. Deferred items (near-dedup, SFT/ALIGNMENT, full multi-source 2T) are explicitly called out and do not contradict the charter.

---

## 4. Strengths (Why a GO Is Justified)

1. **Clear scoping**  
   The report clearly states: C4 only, exact dedup only, no near-dedup, raw text dropped for feasibility, token accounting derived from stats. No overclaim.

2. **Reproducibility and determinism**  
   A fixed seed (e.g. 42) and config/curriculum hashes in manifests make it possible to confirm which config was used. Checkpoint compatibility checks prevent resuming with a different shard count or stage target. The used-chunk store keeps the same chunk set excluded across runs, so same config and input yield the same indices.

3. **Pipeline design matches 2T-scale intent**  
   Streaming and per-batch budgeting avoid loading 2T tokens into memory. Stage-level remaining budgets let later batches fill shortfalls (e.g. if one batch had little B2, the next can take more). Sharding spreads work across workers. INFRA_RECOMMENDATIONS_DECISION_MATRIX correctly flags the parts that must stay identical if the pipeline is ever ported (engine_batched, used_chunks, checkpoint semantics) and recommends keeping them in Python until a Spark path is proven equivalent.

4. **Filters and banding are documented**  
   The “Identified Filters” table describes what runs before the coreset engine (Stage 1/2 rejects, banding gate). The engine’s role—band inference when the label is missing or ineligible, plus curriculum gating—is clearly separate.

5. **Compression and reduction metrics are interpretable**  
   The report separates the single-pass corpus size (136.9B tokens in C4 after EMR dedup—the real input pool) from cumulative stage exposure (458.7B, the sum of “tokens seen” across all four stages, a process metric). The 5.62x ratio is 458.7B ÷ 81.56B selected, so the denominator is clear and misinterpretation is avoided.

6. **Failure conditions and next actions**  
   Curriculum violations, domain/difficulty spikes, B4/B5 dilution, non-determinism, and infra/access dependencies are listed, along with next steps (freeze curriculum, A/B prefetch, infra matrix).

7. **Band targets vs scoring are separable in design**  
   Band percentages (e.g. “B0 must be 49%” for 1B) are enforced by bucket token targets from the curriculum; band_score (or difficulty_score) is used only to order chunks inside a bucket. So the desired mix (policy) is separate from which chunks are preferred within that mix (ranking). See process guide for details.

---

## 5. Gaps, Risks, and Conditions for Full GO

### 5.1 Infra and access (blocking for full-scale production)

- **Finding:** Full-scale execution depends on AWS admin support: EMR (managed Spark cluster for dedup/stats), EC2 capacity (servers where `coreset_builder.py` runs), permissions, and scheduling. The report estimates ~4–5 h for EMR on ~2 TB input and ~12–14 h on EC2 for coreset generation. On-demand instances are stable; Spot is cheaper but can be interrupted (the report notes ~20 restarts on a ~10 GB Spot run). Persistent disk (EBS) of ~600 GB is needed for intermediate and output data.
- **Risk:** Without confirmed capacity and access, production timelines are uncertain.
- **Condition for GO:** Confirm with infra/AWS admin: (a) EMR run on ~2 TB input is scheduled or schedulable, (b) EC2 profile (e.g. c6i.32xlarge / r6i.16xlarge) and ~600 GB EBS are provisionable for the duration of the run.

### 5.2 Token-level accounting and data contract

- **Finding:** Team 3 did not receive token-level allocation and rejection summary tables; counts are derived from stats and on-the-fly logic. A single, authoritative count of how many tokens exist at each step (raw → after dedup → after filters → selected) is still a dependency. Similarly, the agreed shape of data between teams—which columns upstream must provide and what outputs mean—is not fully written down.
- **Risk:** Downstream or audit may need a full rejection breakdown; current numbers may be hard to reconcile. Ambiguity on required vs optional input columns (and behavior when optional ones are missing) can lead to inconsistent runs.
- **Condition for GO:** Either (a) restore or derive token-level allocation and rejection summaries in the T3 workflow and document them, or (b) formally accept the current stats-based accounting and document limitations in a short “Token accounting scope” note. For the data contract, document required and optional fields and how the pipeline behaves when optional fields (e.g. band_score) are absent.

### 5.3 Upstream scoring fields (band_score / difficulty_score)

- **Finding:** Within each (band, domain) bucket, chunks are ordered by: presence and value of band_score (or the configured band_score_source), then diversity composite, then chunk_id. Those score columns are supplied by upstream (e.g. Team 2 / EMR); the engine does not compute difficulty. If they are missing, ordering is only by diversity and chunk_id. For instance, in a bucket with target 1M tokens, chunks A (band_score=0.9), B (0.5), C (missing) would be picked in that order when scores exist; without scores, order depends only on diversity and chunk_id.
- **Risk:** If the production schema does not specify or validate these columns, some runs may have scores and others not, or selection may not match intended “quality-first” behavior.
- **Condition for GO:** (a) Document the expected upstream schema: at minimum chunk_id, band, domain, language, token_count (or equivalent); optionally band_score, difficulty_score, or band_p_B0..B5. (b) Either validate presence of at least one score column for production inputs or explicitly accept “no score” and document that ordering is then diversity + chunk_id only. (c) If using band inference (e.g. infer_if_ineligible), document which band_score_source is used and that upstream provides the corresponding column.

### 5.4 total_input_tokens_estimate correctness

- **Finding:** In streaming mode, the per-batch token budget is stage_target × (batch_tokens_raw / total_input_tokens_estimate). So for stage 1B target 20B, total 137B, and a batch with 500M raw tokens, the batch budget is 20B × (0.5B / 137B) ≈ 73M tokens. If the total were wrongly set to 200B (e.g. from another source), the same batch would get only 50M—systematically under-selecting. README and commands.sh state that this total must match the actual token count at the input path.
- **Risk:** A wrong total silently skews selection and can invalidate reproducibility or charter targets.
- **Condition for GO:** (a) Recompute the total (e.g. via tools/estimate_total_tokens.py) when the input path or source changes. (b) Put “total_input_tokens_estimate verified for this input path” on the production run checklist and record the value in the run manifest or report.

### 5.5 Near-dedup and B4/B5 coverage on C4

- **Finding:** Near-dedup (removing near-duplicate content, e.g. same article with small edits) was not run. C4 has very little B4 (365 docs in the reported stats). The curriculum allows B4 only for domains science, math, code, instruction; B4+web is not allowed, so that slice is filtered or re-banded.
- **Risk:** For the full charter (multi-source, 400B), near-dedup and richer B4/B5 coverage may matter; C4-only runs do not validate those.
- **Condition for GO:** Accept C4-only and exact-dedup-only as sufficient for this delivery milestone; document “Near-dedup and B4/B5-heavy sources” as a follow-up when data and compute allow.

### 5.6 Convergence / benchmark validation

- **Finding:** The report ties “efficiency without learning degradation” to compression and throughput; it states that final learning-quality confirmation remains benchmark-dependent and lists convergence comparison vs full-data training as “Pending final sign-off.” To validate quality, we need at least one comparison where the only change is the dataset (coreset vs full or random), with the same model size and setup—then compare training loss over time and/or benchmark scores (e.g. MMLU).
- **Risk:** Without such a comparison, there is no empirical evidence that the selected coreset preserves learning quality.
- **Condition for GO:** Plan (and ideally run) one controlled A/B and publish a short summary (e.g. loss curve, one or two benchmarks). If time does not allow before handoff, state explicitly that “learning-quality validation is deferred to Phase 2” and add it to the failure-conditions watchlist.

### 5.7 Operational reliability (Spot, long runs)

- **Finding:** The report mentions ~20 interruption/restart events on a ~10 GB run on EC2 Spot and nearly a full day of runtime due to Spot terminations.
- **Risk:** At 2T/400B scale, Spot-only runs may be unreliable or slow; checkpoint/resume helps but does not remove the risk of repeated restarts and long wall-clock time.
- **Recommendation:** Prefer on-demand for production coreset runs (as in the infra matrix) or use Spot with a clear retry budget and monitoring; document the chosen strategy in the run playbook.

### 5.8 Checkpoint and used_chunks store (ops)

- **Finding:** If multiple shards share the same --checkpoint-dir, they overwrite each other’s checkpoints. Each shard needs its own directory (e.g. output/checkpoints_1B/shard000, shard001, …). The used_chunks store (SQLite under output/coresets/.used_chunks/) holds chunk IDs already selected in previous stages; if it is deleted before running a later stage, chunks chosen for an earlier stage could be chosen again, breaking non-overlap. With four shards, four checkpoint dirs and four used_chunks DBs (e.g. used_chunks_shard000.sqlite through shard003) are required.
- **Risk:** A shared checkpoint-dir corrupts or loses state; deleting .used_chunks between stages breaks cross-stage disjointness.
- **Condition for GO:** Add to the production checklist: (a) unique checkpoint-dir per shard, (b) do not delete .used_chunks between stages, (c) do not change num_shards/shard_id/stage_target_scale without a fresh checkpoint-dir.

---

## 6. Suggestions (Non-Blocking)

1. **One-page “Token accounting scope”**  
   Describe what is counted (e.g. post-dedup corpus total, cumulative stage exposure, selected tokens), where each number comes from (which job, which table), and what is not yet available (e.g. authoritative rejection breakdown). This will reduce ambiguity for downstream and audit.

2. **Upstream data contract (one-pager)**  
   Document expected input schema: required fields (chunk_id, band, domain, language, token_count or equivalent), optional scoring fields (band_score, difficulty_score, band_p_B0..B5), and behavior when score columns are missing (ordering falls back to diversity + chunk_id). Reference in README or REPORT_GENERATION_GUIDE.

3. **A/B prefetch and batch-size run**  
   The report suggests A/B: prefetch=auto vs off, fixed num-shards and batch-size. Running this on a representative subset and recording throughput and stability would strengthen operational confidence. Prefetch loads the next batch while the current one is processed, overlapping I/O and compute.

4. **Explicit “Production run checklist”**  
   A short checklist covering: curriculum FROZEN; total_input_tokens_estimate set and verified for input path; unique checkpoint-dir per shard; on-demand vs Spot documented; used_chunks store preserved across stages; no change to num_shards/shard_id/stage_target_scale without new checkpoint-dir; manifest hashes and (if applicable) total_input_tokens_estimate recorded in report or manifest.

5. **B4/B5 and domain coverage in manifests**  
   Ensure each stage manifest includes band and domain composition and, where relevant, availability_stats (how much eligible data remained, so “target not met” can be read as “not enough data” vs “pipeline bug”). Making these part of the production manifest contract will help downstream and debugging.

6. **“Availability-limited” interpretation note**  
   When input has few or no chunks for a band/domain (e.g. C4 and B4), the validator can label target/ratio shortfalls as availability-limited. A short note in REPORT_GENERATION_GUIDE or the report template would clarify that such “failures” mean “not enough data” rather than a pipeline bug.

7. **Failure-conditions runbook**  
   Turn the “Failure conditions watchlist” into a one-page runbook: what to check (e.g. curriculum ratio violations, overlap check), which tools to run (e.g. validate_coreset_outputs.py, check_stage_overlaps.py, merge_sharded_ablation_reports.py, generate_verification_artifacts.py), and when to escalate.

8. **Scoring and token_ids (optional doc note)**  
   In production streaming, token_ids (tokenizer output per chunk) are often absent; token-level rarity in the diversity scorer is then skipped and a length-based proxy is used when needed. A one-sentence note in the T3 report or REPORT_GENERATION_GUIDE that scoring is metadata/column-driven and token-level rarity applies only when token_ids are present would avoid future confusion.

---

## 7. Go/No-Go Summary

| Criterion | Status | Condition for full GO |
|-----------|--------|------------------------|
| Report accuracy and completeness | **Pass** | — |
| Charter/deliverables alignment (executed scope) | **Pass** | — |
| Determinism and reproducibility | **Pass** | Freeze curriculum for production runs |
| Infra and access | **Open** | Confirm EMR + EC2 capacity and permissions |
| Token accounting / data contract | **Open** | Restore/derive or document scope and limitations |
| Upstream scoring (band_score/difficulty_score) | **Open** | Document schema; validate or accept “no score” behavior |
| total_input_tokens_estimate | **Caution** | Verify for input path; add to checklist and manifest |
| Checkpoint / used_chunks ops | **Caution** | Per-shard checkpoint-dir; preserve .used_chunks; document in checklist |
| Near-dedup / B4/B5 (full charter) | **Deferred** | Accept as Phase 2; document in scope note |
| Learning-quality / convergence evidence | **Pending** | One controlled A/B or explicit deferral + watchlist |
| Operational reliability | **Caution** | Prefer on-demand; document strategy |

**Final recommendation**

- **Conditional GO** for:  
  - The scope defined in 2026-02-23 T3 report.  
  - Using the current Python streaming pipeline as the **production baseline** for coreset generation.  
  - Proceeding with merge/publish of the report and artifacts, with the understanding that full-scale production depends on infra, data-contract, upstream-schema, and token-accounting conditions above.

- **Before full production at 2T/400B scale:**  
  Satisfy the conditions in §5 (infra, token accounting, curriculum freeze, upstream scoring contract, total_input_tokens_estimate verification, checkpoint/used_chunks discipline, and either one convergence/benchmark check or an explicit deferral). Implementing the suggestions in §6 will improve robustness and clarity.

---
