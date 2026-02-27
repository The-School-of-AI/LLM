# Infra Recommendations Decision Matrix — 2026-02-23

## Executive Summary

This memo recommends a **phased execution path**: keep the current Python streaming pipeline as the production baseline, improve infra reliability/performance on AWS first, and run a **bounded 2-3 day PySpark feasibility spike** before any migration decision.

- **Recommendation:** Production runs should continue on current deterministic Python flow with EC2 on-demand for coreset generation and EMR for upstream data prep.
- **Why now:** Existing implementation already supports deterministic resume, shard-level parallelism, and non-overlap guarantees.
- **Migration stance:** PySpark is promising for file listing/scan throughput, but should be adopted only if it meets explicit success thresholds without regressions in determinism and policy behavior.

---

## Scope and Evidence Policy

- This document separates claims into:
  - **Artifact-validated:** directly confirmed from repository code and run behavior.
  - **Estimate/assumption:** operational estimates requiring run-time verification.
- Goal is decision support for upcoming ~2 TB preprocessing and ~500-600 GB downstream coreset generation scope.

---

## Current Pipeline Facts (Artifact-validated)

1. **Sharded orchestration exists and is operational**
   - `shard.sh` launches parallel shard workers with shard-specific checkpoints and resume support.
2. **Deterministic stage-wise streaming selection exists**
   - `StreamingCoresetBuilder` performs parse → stage gating → batched selection → write → checkpoint loop.
3. **Checkpoint/resume compatibility guards are implemented**
   - Stage target, shard identity, and runtime compatibility checks are enforced on resume.
4. **Cross-stage non-overlap is enforced**
   - `UsedChunksStore` (SQLite-backed) prevents reused chunk IDs across stages.
5. **S3/parquet batch iteration is implemented in current flow**
   - `BatchProcessor` supports streamed listing/filtering and deterministic batch iteration.
6. **Selection policy controls are implemented in batched engine**
   - Stage budgets, domain/language gating, and protected-slice constraints are present in `engine_batched` logic.

### Parity-critical controls that must be preserved in any Spark integration

- **Deterministic sharding and stage targeting:** current flow splits stage target by `num_shards` and persists shard identity.
- **Fault tolerance and resume safety:** checkpoint state includes shard identity, stage target, counters, and selection-engine internal state.
- **Cross-stage non-overlap:** `UsedChunksStore` (SQLite) prevents reused chunk IDs across stages.
- **Policy parity:** domain/language/band gating and protected-slice logic in `BatchedSelectionEngine` must stay semantically identical.
- **Manifest parity:** output manifests include deterministic metadata, composition distributions, and shard-scoped identity.

### Important implementation note (Artifact-validated)

- In the currently inspected entrypoint, sharding/checkpoint controls are explicit.
- Prefetch/auto CPU-ratio behavior appears in run logs and prior run documentation, but should be treated as **configuration/runtime behavior to re-validate** when planning migration.

---

## Explicit PySpark Migration Map (Module-by-Module)

### A) Can migrate right away (low parity risk, high I/O leverage)

| Current module/process | Current responsibility | PySpark target | Feasibility now | Parity risk |
| --- | --- | --- | --- | --- |
| `src/io/batch_processor.py` (`list_input_files`, parquet scanning) | File discovery and batch reads | Spark DataFrame read/list + partition scan | **High** | Low-Medium |
| Input normalization pre-stage (`chunk_id`, schema projection, null cleanup) | Row prep before metadata conversion | Spark projection/select/cast + deterministic sort key | **High** | Low-Medium |
| Pre-selection eligibility materialization (language/domain/band eligibility flags) | Gating support before selection | Spark-side derived eligibility columns | **High** | Medium |
| Output part-file writing for selected rows | Batch part writes (`parquet/csv/jsonl`) | Spark write path with fixed schema/partitioning | **Medium-High** | Medium |

### B) Feasible, but only after parity harness is in place

| Current module/process | Current responsibility | PySpark target | Feasibility now | Parity risk |
| --- | --- | --- | --- | --- |
| `src/io/used_chunks_store.py` | Cross-stage overlap prevention via SQLite membership checks | Spark Delta/Hudi/Iceberg anti-join membership table (or external KV) | **Medium** | Medium-High |
| Checkpoint persistence in `BatchProcessor` + stage loop state in `StreamingCoresetBuilder` | Resume-safe recovery with engine state | Spark checkpoint + deterministic state snapshots + stage resume index | **Medium** | High |
| Shard orchestration in `shard.sh` | Process-level shard fanout | Spark partition/task orchestration + explicit shard contract layer | **Medium** | Medium-High |

### C) Keep in Python for now (do not migrate immediately)

| Current module/process | Why keep now | Migration timing |
| --- | --- | --- |
| `src/selection/engine_batched.py` (`_process_batch`, rolling-window/state restore) | This is parity-critical decision logic; drift risk is highest here | Migrate only after A/B paths are proven equivalent |
| Stage-level orchestration + manifest semantics in `StreamingCoresetBuilder._build_stage_coreset` | Controls deterministic resume contracts and coreset accounting semantics | Refactor last, after data-plane parity is stable |
| Protected-slice and curriculum policy interpretation path | Silent semantic drift here can invalidate benchmark comparability | Keep Python source-of-truth until exhaustive parity tests pass |

---

## Feasibility Confirmation: Is PySpark Really an Option?

### Short answer

- **Yes, PySpark is a real option**, but **as incremental integration**, not full replacement in one step.

### Why yes

- Your workload has clear Spark-friendly surfaces: large parquet scans, file listing, projection/filter transforms, and distributed writes.
- These can be integrated without immediately touching parity-critical selection semantics.

### Why not full immediate migration

- Deterministic selection, resume state fidelity, and cross-stage non-overlap currently rely on tight Python engine + checkpoint contracts.
- Full port in one pass materially increases risk of semantic drift and operational regression.

### Final feasibility stance

- **Feasible to integrate now:** I/O and preprocessing surfaces (Section A).
- **Feasible later with controls:** overlap-store/checkpoint orchestration surfaces (Section B).
- **Not recommended now:** core selection semantics and stage decision engine (Section C).

---

## What can move to PySpark right away (explicit list)

1. File enumeration and parquet read layer currently handled by `BatchProcessor`.
2. Schema normalization/projection before selection metadata conversion.
3. Eligibility-flag derivation for language/domain/band pre-gating support.
4. Selected-row output writing path (part files) with strict schema parity checks.

## What should not move right away

1. `BatchedSelectionEngine` token-budget and rolling-window selection state machine.
2. Checkpoint state contract that restores deterministic engine state on resume.
3. `UsedChunksStore` overlap semantics until anti-join replacement is validated under crash/restart scenarios.

---

## Decision Matrix

| Option | Throughput upside | Determinism risk | Engineering effort | Cost predictability | Recommended use |
| --- | --- | --- | --- | --- | --- |
| **A. Current Python + EC2 on-demand (production baseline)** | Medium (with tuning) **(Estimate/assumption)** | Low **(Artifact-validated controls exist)** | Low-Medium | High | **Primary path now** |
| **B. Keep Python core, use EMR/EMR Serverless for prep-heavy transforms only** | Medium-High **(Estimate/assumption)** | Low-Medium | Medium | Medium-High | **Secondary near-term enhancement** |
| **C. Partial PySpark migration for streaming/selection path** | Potentially High **(Estimate/assumption)** | Medium-High (porting semantics) | High | Medium | **Feasibility spike only** |

### Verdict Summary

- **Choose Option A now**, incorporate selected infra improvements, and run **Option C as a time-boxed feasibility spike**.
- Revisit production migration decision only after spike metrics are collected and compared against current baseline.

---

## Recommended AWS Baseline (Now)

### Compute

- **Coreset generation:** EC2 on-demand instances for long-running deterministic jobs.
- **Preferred profiles:**
  - `c6i.32xlarge` for CPU-heavy throughput runs.
  - `r6i.16xlarge` for memory-sensitive stages.

### Storage

- Attach **~600 GB gp3 EBS** as a baseline for checkpoints, manifests, and intermediate writes.
- Tune gp3 IOPS/throughput only after first profiling pass (avoid premature over-provisioning).

### Runtime configuration guidance

- Maintain shard-parallel execution through `shard.sh`.
- Keep checkpoint cadence conservative during validation, then tune for throughput once resume confidence is established.
- Preserve deterministic seed/config hash tracking in manifests for every production run.

### Confidence labels

- **Artifact-validated:** deterministic controls, sharding, checkpoint/restart pattern, and non-overlap architecture.
- **Estimate/assumption:** exact wall-clock runtime, final cost, and optimal instance/storage tuning at full production load.

---

## PySpark Feasibility Spike (3-4 Days, Time-boxed)

## Objective

Evaluate whether selective PySpark adoption can improve throughput/cost **without** changing selection semantics or deterministic reproducibility.

## Entry criteria

- Frozen input sample (same source window and seed strategy).
- Agreed acceptance metrics (below).
- Side-by-side baseline run from current Python path captured first.

## Experiment boundaries

- Port only high-I/O and partition-scan components first.
- Do **not** change scoring/policy semantics during spike.
- Keep output schema and manifest contract identical.

## Success thresholds (must all pass)

1. **Output parity:** selected token/chunk totals and composition distributions stay within agreed tolerance.
2. **Determinism parity:** rerun stability under same inputs/seeds is preserved.
3. **Performance gain:** measurable wall-clock and/or cost-per-token improvement versus baseline.
4. **Operational resilience:** checkpoint/restart behavior remains reliable under interruption.

## Rollback rule

If any threshold fails, stop migration work and continue with current Python production path plus infra tuning.

---

## Risks and Trade-offs

- **Risk:** PySpark port may introduce subtle semantic drift in gating/selection behavior.
  - **Mitigation:** strict parity tests and schema/manifest diff checks.
- **Risk:** Spot interruptions can stretch runtime unpredictably.
  - **Mitigation:** prefer on-demand for critical long-running production runs.
- **Risk:** Over-tuning infra before baseline profiling can increase spend without benefit.
  - **Mitigation:** phased tuning after first measured run.

---

## Metrics to Track (for Go/No-Go)

1. End-to-end wall-clock per stage and total pipeline.
2. Cost-per-selected-token (or cost per 1M selected tokens).
3. Checkpoint count, average checkpoint write time, resume recovery time.
4. Determinism parity across reruns (hashes/manifests/state).
5. Output composition stability (band/domain/language) vs baseline.

---

## Two-Week Execution Plan

### Week 1

1. Provision AWS access dependencies (IAM, EC2, EMR readiness).
2. Run production-style baseline on current Python path and collect metrics.
3. Apply low-risk infra/runtime tuning (instance/storage/checkpoint cadence) and re-measure.

### Week 2

1. Execute 3-4 day PySpark feasibility spike with fixed scope.
2. Run parity + determinism + cost/perf evaluation.
3. Hold go/no-go review:
   - **Go:** only if all success thresholds pass.
   - **No-go:** retain Python baseline and continue incremental infra optimization.

---

## Final Verdict

### Decision

- **Verdict:** PySpark is **suitable as a conditional optimization path**, not as the immediate default for coreset generation.
- **Production now:** continue with the current deterministic Python pipeline as primary.
- **Near-term strategy:** execute a bounded PySpark feasibility spike and adopt only on evidence.

### Go/No-Go Criteria (all required for GO)

1. **Output parity:** selected token/chunk totals and composition remain within agreed tolerance.
2. **Determinism parity:** repeated runs under fixed input/seed produce stable reproducible outputs.
3. **Operational reliability:** checkpoint/restart behavior is stable under interruption scenarios.
4. **Efficiency gain:** wall-clock and/or cost-per-token shows a clear improvement versus baseline.

### Final recommendation if criteria are not met

- Keep Python pipeline as system-of-record for coreset generation.
- Use EMR/PySpark only for upstream preprocessing and transform-heavy stages.
- Continue infra tuning on EC2/EBS baseline with measured iteration.

### Decision Sign-off

- **Decision owner:** ____________________
- **Date:** ____________________
- **Baseline run ID:** ____________________
- **PySpark spike run ID(s):** ____________________
- **Outcome:** GO / NO-GO
- **Notes:** ____________________

---

## References

### Internal artifacts

- `shard.sh`
- `coreset_builder.py`
- `src/io/batch_processor.py`
- `src/io/used_chunks_store.py`
- `src/selection/engine_batched.py`
- `docs/T3_REPORTS/2026-02-23_T3_REPORT.md`

### External references used during due-diligence

- EMR Serverless documentation (service capabilities and fit).
- EC2 `c6i`/`r6i` instance family documentation.
- EBS `gp3` performance/cost model documentation.
- Spark tuning guidance for scan/listing/memory/parallelism behavior.
