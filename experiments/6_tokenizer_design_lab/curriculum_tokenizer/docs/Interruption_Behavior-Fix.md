# Interruption Behavior — Bug Analysis & Fix

**Date:** 2026-03-05
**File affected:** `tokenize_curriculum.py`
**Severity:** High — data integrity issue in sequential processing path

---

## 1. Bug Description

When a Spot interruption (SIGTERM / SIGINT / Ctrl+C) fires **during** the processing of a
coreset batch file (i.e., mid-batch, while not all T1 files have been processed), the pipeline
incorrectly marks that batch as **completed** in `progress_state.json`.

On the next run, the batch is skipped entirely. The tokens from the T1 files that were not
yet processed are permanently lost.

At production scale (150M tokens per batch, 5 T1 files per batch), an interruption after the
first T1 file means **~80% of that batch's tokens are silently dropped**.

---

## 2. Root Cause

### Code location: `tokenize_curriculum.py` lines 880–892 (sequential loop)

```python
for idx, uri in enumerate(pending_files):
    if _TERMINATION_DETECTED.is_set():          # ← checks flag BETWEEN batches only
        print("\n[INTERRUPT] Stopping file loop.")
        break
    stats = process_coreset_file(...)            # ← signal may fire INSIDE this call
    if stats:
        all_stats.append(stats)
        completed_set.add(uri)                  # ← unconditionally marks as completed
        progress["completed"] = list(completed_set)
        save_progress_state(...)
```

**Why the check is insufficient:**

- The `_TERMINATION_DETECTED.is_set()` guard at the top of the loop catches signals that
  arrive *between* batch files.
- When the signal fires *during* `process_coreset_file()`, the inner T1-file loop in that
  function breaks early (line 575–577), returns partial results, and execution falls back to
  the outer loop.
- At that point, `_TERMINATION_DETECTED` is already set, but the outer loop never re-checks
  it before executing `completed_set.add(uri)`.

**Why `if stats:` does not protect against this:**

- `process_coreset_file()` always returns a non-empty dict (line 719), even when only 1 of 5
  T1 files was processed.
- So `if stats:` evaluates to `True` in all cases, including interrupted ones.

### Observed evidence (from local halt-resume test)

```
--- File 3/10 ---
Processing Coreset: selected_indices_small_batch000002.parquet
  Unique source files to fetch: 5
  [UPLOADED] shard_001: 1 blocks, 0.0 MB       ← only T1 file 1 processed
Map (num_proc=2):   0%|...                      ← T1 file 2 starting
[SIGNAL] Received signal 2. Initiating graceful shutdown...
[INTERRUPT] Stopping before next T1 file due to termination signal.
  Completed: 3/10 files                         ← batch000002 wrongly counted
```

`progress_state.json` after the interrupt:
```json
{
  "completed": [
    "dataset/final/small/t3/selected_indices_small_batch000002.parquet",
    "dataset/final/small/t3/selected_indices_small_batch000001.parquet",
    "dataset/final/small/t3/selected_indices_small_batch000000.parquet"
  ]
}
```

`batch000002` has only `shard_001` (1 T1 file's worth of data). 4 of 5 T1 files were
never processed. On resume, `batch000002` would be skipped entirely — data loss confirmed.

---

## 3. Proposed Fix

### Change 1 — Sequential loop: check `_TERMINATION_DETECTED` after the call

**File:** `tokenize_curriculum.py`, lines 885–892

```python
# Before (incorrect):
stats = process_coreset_file(...)
if stats:
    all_stats.append(stats)
    completed_set.add(uri)                       # always runs, even on interrupt
    progress["completed"] = list(completed_set)
    save_progress_state(s3, args.dst_uri, progress, tmp_dir)

# After (correct):
stats = process_coreset_file(...)
was_interrupted = _TERMINATION_DETECTED.is_set()  # check AFTER the call returns
if was_interrupted:
    # Batch may have partial shards on disk — track as interrupted, NOT completed
    interrupted_set.add(uri)
    progress["interrupted"] = list(interrupted_set)
    save_progress_state(s3, args.dst_uri, progress, tmp_dir)
    print("\n[INTERRUPT] Stopping file loop.")
    break
else:
    if stats:
        all_stats.append(stats)
        completed_set.add(uri)
        progress["completed"] = list(completed_set)
        save_progress_state(s3, args.dst_uri, progress, tmp_dir)
```

Also remove the redundant early-loop break block (the one at line 881) since the
post-call check subsumes it, OR keep it as an optimization to skip files when
termination is already known before starting.

### Change 2 — `load_progress_state()`: add `interrupted` key to default state

**File:** `tokenize_curriculum.py`, line 229

```python
# Before:
return {"completed": [], "failed": []}

# After:
return {"completed": [], "interrupted": [], "failed": []}
```

### Change 3 — Resume startup: clean up interrupted batch dirs before re-queuing

**File:** `tokenize_curriculum.py`, after line 870 (progress state load)

```python
completed_set = set(progress.get("completed", []))
interrupted_set = set(progress.get("interrupted", []))

# Clean up partial shard directories from interrupted batches
if interrupted_set:
    print(f"  [RESUME] Cleaning up {len(interrupted_set)} interrupted batch(es)...")
    for uri in interrupted_set:
        batch_stem = os.path.splitext(os.path.basename(uri))[0]
        partial_dir = f"{args.dst_uri.rstrip('/')}/{batch_stem}"
        if partial_dir.startswith("s3://"):
            _delete_s3_prefix(s3, partial_dir)           # S3 path (helper needed)
        elif os.path.isdir(partial_dir):
            shutil.rmtree(partial_dir)                   # local path
            print(f"    Cleared: {batch_stem}")
    progress["interrupted"] = []                         # reset after cleanup
    save_progress_state(s3, args.dst_uri, progress, tmp_dir)

# interrupted batches are NOT in completed_set → they fall into pending naturally
pending_files = [uri for uri in target_files if uri not in completed_set]
```

### Change 4 — New helper: `_delete_s3_prefix(s3, prefix_uri)`

Required for cleaning up interrupted batch dirs on S3. Example skeleton:

```python
def _delete_s3_prefix(s3_client, prefix_uri: str) -> None:
    """Delete all S3 objects under a given prefix URI."""
    bucket, prefix = _parse_s3_uri(prefix_uri)
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if objects:
            s3_client.delete_objects(Bucket=bucket, Delete={"Objects": objects})
```

---

## 4. Updated `progress_state.json` Schema

```json
{
  "completed": [
    "s3://bucket/coresets/batch_000000.parquet",
    "s3://bucket/coresets/batch_000001.parquet"
  ],
  "interrupted": [
    "s3://bucket/coresets/batch_000002.parquet"
  ],
  "failed": [],
  "last_updated": "2026-03-05T14:30:00Z"
}
```

On the next resume run:
- `completed` entries → skipped (not reprocessed)
- `interrupted` entries → partial dirs deleted, then re-queued for full reprocessing
- `interrupted` field is reset to `[]` after cleanup

---

## 5. Gaps and Edge Cases

| # | Gap | Severity | Resolution |
|---|-----|----------|------------|
| 1 | **Signal fires after last T1 file completes, before outer loop re-checks flag** | Low | Batch is truly complete but will be marked `interrupted`. It gets deleted and re-processed on resume. Safe: shard-level skip (`metadata.json` check) makes re-processing idempotent. Slight waste. Acceptable. |
| 2 | **Parallel path (`--file-parallelism N`) not covered by this fix** | Medium | See §6 below. Data integrity is not at risk in the parallel path; the bug is specific to the sequential path. |
| 3 | **S3 cleanup needs `_delete_s3_prefix()` helper** | Medium | Helper does not exist yet. Must be added (Change 4 above). Local path uses `shutil.rmtree()`. |
| 4 | **`num_source_files` in returned stats is inflated on interrupt** | Low | `len(grouped_sources)` counts total T1 files in batch regardless of how many were processed. Misleading in logs; does not affect correctness. Out of scope for this fix. |
| 5 | **If interrupt fires before any T1 file is processed** | Low | `stats` is returned with `num_shards=0`; no partial dir exists; cleanup is a no-op. Handled gracefully. |

---

## 6. Parallel Path Analysis

**The sequential-path bug does NOT exist in the parallel path.**

In the parallel path (lines 893–913):
```python
with ctx.Pool(processes=args.file_parallelism) as pool:
    results = pool.map(_worker_process_coreset, worker_inputs)

for uri, stats in zip(pending_files, results):
    if stats:
        completed_set.add(uri)

progress["completed"] = list(completed_set)
save_progress_state(...)
```

When SIGTERM fires during `pool.map()`:
- The pool is terminated; `pool.map()` raises an exception.
- The `for uri, stats in zip(...)` and `save_progress_state()` calls never execute.
- `progress_state.json` retains only the state from before the parallel run started.
- No batches are incorrectly marked as `completed`.

**Residual concern (efficiency, not integrity):**
Workers that completed their batch fully before the pool was killed will have valid shards
on disk. Since `progress_state.json` was not updated, those batches will be re-processed
on the next run. The shard-level skip (`key_exists(metadata.json)`) will detect existing
valid shards and skip re-writing them, so no data is duplicated. Only compute time is wasted.

**No code change needed in the parallel path for data integrity.**

---

## 7. Fix Priority

| Priority | Change | Risk |
|----------|--------|------|
| P0 | Change 1 — post-call interrupt check in sequential loop | Low — isolated change |
| P0 | Change 2 — `interrupted` key in default state | Low — backwards compatible |
| P0 | Change 3 — resume cleanup logic | Medium — file deletion |
| P1 | Change 4 — `_delete_s3_prefix()` helper | Medium — S3 API calls |

---

## 8. Test Plan (after fix)

1. Run small profile tokenizer with `--shard-size-mb 0.025`
2. Press Ctrl+C after ~3 batches complete
3. Verify `progress_state.json`:
   - `"completed"` has 2 entries (batches 000000 and 000001)
   - `"interrupted"` has 1 entry (batch 000002)
4. Verify `dataset/final/small/tok_out_resume/selected_indices_small_batch000002/` exists
   with partial shard(s)
5. Re-run with same args
6. Verify log shows cleanup: `[RESUME] Cleared: selected_indices_small_batch000002`
7. Verify log shows: `Resuming: 2 already complete, 8 remaining`
8. Verify batch000002 is fully reprocessed (all 5 T1 files, all shards)
9. Run `validate_shards.py` — all 10 batches PASS