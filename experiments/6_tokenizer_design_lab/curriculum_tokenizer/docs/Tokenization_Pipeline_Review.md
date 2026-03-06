# Tokenization Pipeline Review & AWS Strategy

**Document version:** 2026-03-06
**Pipeline:** `tokenize_curriculum.py` — S3 Curriculum Tokenization Pipeline
**Data scale:** 9.896B tokens (Stage 1B actual) · 12,231 T3 batch files · 25,420 unique T1 files · 4,305 GB T1 source data
**Source data region:** us-east-1 (already available)

---

## Table of Contents

1. [Pipeline Review — Pending Items Checklist](#1-pipeline-review--pending-items-checklist)
2. [Local Testing Strategy](#2-local-testing-strategy)
3. [Parallel Tokenization — Current State & Enhancement](#3-parallel-tokenization--current-state--enhancement)
4. [AWS Deployment Guide](#4-aws-deployment-guide)
   - [4.1 Execution Strategy](#41-execution-strategy)
   - [4.2 Cost and Duration Breakdown](#42-cost-and-duration-breakdown)
   - [4.3 S3 Setup and IAM](#43-s3-setup-and-iam)
   - [4.4 Spot Instance Interrupt Handling](#44-spot-instance-interrupt-handling)

---

## 1. Pipeline Review — Pending Items Checklist

### P0 — BLOCKERS (pipeline cannot run correctly without these)

- [x] **[ARCH-01] Script downloads T2 instead of T1 — 2-level migration required.**
  - **Supersedes**: BUG-01 (T2 path construction no longer needed; T2 is entirely bypassed).
  - **Location**: `process_coreset_file()` — the entire source-file groupby and download block.
  - **Problem**: The script groups T3 rows by `source_url + source_doc_id` (a T2 band directory path), downloads T2 band files, and filters T2 by `id == chunk_id`. T2 has no `text` column, so the script always hits the `SKIP` branch and produces zero tokens.
  - **Fix**: Group by `t1_file_path` column (hardcoded — fixed by T3 schema). Build T1 URI as `args.t1_base_uri.rstrip("/") + "/" + t1_file_path`. Download T1 directly. Filter `T1.id == chunk_id`. Extract `T1.text`.
  - **New arg**: `--t1-base-uri` (default: `s3://t1-dataacquisition-datasets/processed_dataset/normalized_data`; override to `dataset/final/t1` for local testing).
  - **Dead args to remove**: `--src-doc-col`, `--url-col`, `--src-id-col`, `--coreset-id-col`, `--text-col`, `--band-col`, `--domain-col`.
  - **Hardcoded constants**: `t1_file_path` (T3 column), `chunk_id` (T3 ID column), `id` (T1 ID column), `text` (T1 text column), `band`, `domain`.

- [x] **[BUG-02] `key_exists()` uses wrong exception class.**
  - **Location**: Line 105: `except s3.exceptions.ClientError`
  - **Problem**: `boto3.client("s3").exceptions.ClientError` does not exist; `ClientError` is a `botocore` base class, not a service-specific exception. This raises `AttributeError` at runtime, breaking the checkpoint-skip logic entirely.
  - **Fix applied**: Added `from botocore.exceptions import ClientError` at top; changed to `except ClientError as e: if e.response["Error"]["Code"] in ("404", "NoSuchKey"): return False; raise`.

### P1 — CRITICAL (required for training team delivery)

- [x] **[META-01] `tokenizer_hash` missing from `metadata.json`.**
  - Required by `TOKENIZER_TEAM_RECOMMENDATIONS.md` §2. Must be SHA256 of `tokenizer.json` + `special_tokens_map.json` (sorted filenames prepended to hash input).
  - **Fix applied**: Added `compute_tokenizer_hash(tokenizer_dir)` function; pass hash into `ShardWriter.__init__()`; include in every `metadata.json`.

- [x] **[META-02] `band` and `band_distribution` missing from `metadata.json`.**
  - Required for curriculum sampler. Coreset parquets have a `band` column (B0/B1/B2).
  - **Fix applied**: Column name `band` is hardcoded (fixed by T3 schema — not a CLI arg); compute `value_counts()` per batch group; pass distribution into `ShardWriter`; write dominant band + full distribution dict.

- [x] **[META-03] `domain` missing from `metadata.json`.**
  - Same pattern as band. Column name `domain` is hardcoded (fixed by T3 schema — not a CLI arg).

- [x] **[META-04] `stage` missing from `metadata.json`.**
  - **Fix applied**: Added `--stage` arg (integer, e.g., `1`); pass into `ShardWriter`.

- [x] **[META-05] `source_file` missing from `metadata.json`.**
  - Must be the coreset batch parquet URI for full traceability. **Fix applied** in `ShardWriter`.

- [x] **[META-06] `created_at` missing from `metadata.json`.**
  - **Fix applied**: `datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")` in `flush_shard()`.

- [x] **[META-07] `tokenizer_version` missing from `metadata.json`.**
  - **Fix applied**: Added `--tokenizer-version` arg (string, e.g., `"v1"`).

- [x] **[AUDIT-01] `rows_input` not tracked per shard.**
  - Added counter `shard_rows_input` in `ShardWriter`; incremented in the token-packing loop.

- [x] **[AUDIT-02] `rows_with_eos` not tracked per shard.**
  - EOS IS correctly appended per row at line 440 (not a logic bug). The count was not persisted to `metadata.json`.
  - **Fix applied**: Added `shard_rows_with_eos` counter; incremented alongside the `buffer.append(eos)` call.

- [x] **[AUDIT-03] `rows_dropped` computed but not written to `metadata.json`.**
  - Lines 469-476 calculated `dropped_rows` but only `print()` it. **Fix applied**: persisted to metadata.

- [x] **[AUDIT-04] `tokens_dropped` computed but not written to `metadata.json`.**
  - Same: `dropped_tokens = len(buffer)` was computed but not persisted. **Fix applied**.

- [x] **[AUDIT-05] `drop_reason` not written to `metadata.json`.**
  - Must be `"tail_truncation_at_block_boundary"` or `"padded"`. **Fix applied**.

### P2 — IMPORTANT (for production AWS run)

- [x] **[PERF-01] File-level parallelism missing.**
  - 133 coreset batch files were processed sequentially. All CPU cores sat idle while one file downloaded/tokenized.
  - **Fix applied**: Added `--file-parallelism N` arg; uses `multiprocessing.get_context("spawn").Pool(N)`; worker function creates its own `boto3.client` and `AutoTokenizer` (cannot share across `fork`); each worker gets isolated `worker_tmp` subdirectory.
  - **Recommended settings**: c5.9xlarge → `--file-parallelism 12 --num-proc 3` (36 vCPU fully used).

- [x] **[SPOT-01] No Spot interrupt handling.**
  - **Fix applied**: Added IMDS polling daemon thread (checks `http://169.254.169.254/latest/meta-data/spot/termination-time` every 5s); added `signal.signal(SIGTERM, handler)` handler; sets a `threading.Event` flag; checks flag at the start of each source-file loop iteration; on termination: discards partial `accumulated_blocks`, only completed shards are preserved.

- [x] **[SPOT-02] No cross-interrupt progress state file.**
  - **Fix applied**: Writes `progress_state.json` to `dst_uri` after each batch file completes. On startup, reads this file and skips already-completed batch URIs (faster than per-shard S3 `head_object` checks).

### P3 — QUALITY FIXES

- [x] **[CORR-01] `tmp_dir` cleanup uses `os.rmdir()` which fails silently on non-empty dirs.**
  - Line 616. **Fix applied**: Uses `shutil.rmtree(tmp_dir, ignore_errors=True)`.

- [x] **[CORR-02] Global `manifest.json` uses Unix epoch float for timestamp.**
  - Line 600: `"timestamp": time.time()`. **Fix applied**: Uses ISO 8601 string for consistency with shard metadata.

- [x] **[CORR-03] `src_id_col` default `"id"` may not match source parquet column name.**
  - Resolved — T1 schema confirmed: the ID column is `id` (matches default). Hardcoded as a constant in the 2-level migration (ARCH-01); `--src-id-col` arg removed.

### P4 — TOOLING

- [x] **[VALID-01] No standalone validation script.**
  - `TOKENIZER_TEAM_RECOMMENDATIONS.md` §4 defines an 8-point checklist. Created `validate_shards.py` that: reads each shard's `metadata.json`, checks `total_tokens == file_size/4`, checks `len(idx_offsets)-1 == num_blocks`, verifies `tokenizer_hash`, validates `max(token_ids) < vocab_size`, checks `rows_dropped + rows_with_eos == rows_input`, and reports pass/fail per shard.

### Files Modified / Created

| File | Action |
|------|--------|
| `tokenize_curriculum.py` | Modified — all P0/P1/P2/P3 fixes applied |
| `Tokenization-Strategy-AWS.md` | Created — full AWS deployment guide |
| `scripts/create_mock_sources.py` | Redesigned — reads real T2 `file_path` values from local band files in `datafiles/`; generates mock T3 (with `t1_file_path` column) and mock T1 files with full schema; no T2 mocks created |
| `validate_shards.py` | Created — post-run 8-point validation script |

---

### Post-Parallel-Run Bug Fixes (2026-03-05)

The following bugs were discovered and fixed during local parallel and halt-and-resume testing.
All fixes are in `tokenize_curriculum.py` only — no changes to any other file.

---

#### [BUG-P1] Doubled `tmp_dir` Path — First 3 Workers Crash Immediately

**Symptom:**
Workers 0, 1, and 2 crashed immediately on the first parallel run with:
```
[Errno 2] No such file or directory:
  '...tokenize_curriculum_tmp\tokenize_curriculum_tmp\worker_001\tokens.bin'
```
Workers 3–9 completed successfully. All 10 shards from those workers passed validation, confirming the logic was correct — only the path construction was wrong.

**Root cause:**
`main()` constructs `tmp_dir` as:
```python
tmp_dir = os.path.join(base_tmp, "tokenize_curriculum_tmp")
args_dict["tmp_dir"] = tmp_dir   # already includes the suffix
```
`_worker_process_coreset` then did:
```python
worker_tmp = os.path.join(
    args_dict.get("tmp_dir"),
    "tokenize_curriculum_tmp",   # ← BUG: re-appended the suffix
    f"worker_{worker_id:03d}",
)
```
This doubled the `tokenize_curriculum_tmp` segment. Three concurrent workers raced to create the same non-existent nested parent directory on NTFS, all failing with `FileNotFoundError`.

**Fix:**
Removed the extra `"tokenize_curriculum_tmp"` segment from `_worker_process_coreset`:
```python
# Fix: args_dict["tmp_dir"] already includes the "tokenize_curriculum_tmp" suffix
# set in main(). Do NOT re-append it here or the path is doubled.
worker_tmp = os.path.join(
    args_dict.get("tmp_dir") or tempfile.gettempdir(),
    f"worker_{worker_id:03d}",
)
```

**Decision:** Fix applied directly. No backward-compat shim needed — this was a clean path construction bug with no ambiguity.

---

#### [BUG-P2] Orphan Shards After Mid-Batch Interruption Cause Data Duplication on Resume

**Symptom:**
After pressing Ctrl+C mid-batch during a parallel run, the `shards/` directory contained shard directories that had no corresponding `.done` marker. On resume, those batches were re-queued and re-processed, producing new shard numbers for the same documents. The orphan shards from the interrupted run remained on disk, resulting in duplicate token data across two shard numbers.

**Root cause:**
The parallel path used per-coreset `.done` marker files to track completion. A batch was only considered complete once its `.done` file was written. However, `flush_shard()` writes individual shard directories eagerly as soon as enough blocks accumulate. If a worker is killed between its last `flush_shard()` call and its `write_done_marker()` call, those shard directories exist on disk with no claim.

On resume, the shard counter advanced past the orphan numbers (since the counter is initialised from `max(existing shard numbers) + 1`), so the orphan shards were never overwritten — they just silently coexisted with the new ones produced for the same documents.

**Fix — three changes:**

*Change 1:* Add `shard_names` to the `process_coreset_file` return dict:
```python
"shard_names": [s["shard_name"] for s in shard_stats],
```

*Change 2:* Store the full stats dict (including `shard_names`) inside the `.done` marker file via `data.update(stats)`, replacing the previous hard-coded subset of fields.

*Change 3:* Add `purge_orphan_shards()` — called once on resume before the pool starts. It loads all `.done` markers, collects the union of all claimed `shard_names`, then deletes any `shards/shard_NNN/` directory that is not in that claimed set:
```python
def purge_orphan_shards(s3, dst_uri: str, markers: list) -> int:
    if not markers:
        return 0
    if not all("shard_names" in m for m in markers):
        print("[RESUME] Skipping orphan-shard purge: some .done markers predate shard_names tracking.")
        return 0
    claimed = set()
    for m in markers:
        claimed.update(m.get("shard_names", []))
    # delete any shard_NNN/ not in claimed ...
```

**Backward compatibility guard:** If any `.done` marker on disk lacks `shard_names` (written by an older version of the script), the purge is skipped entirely and a warning is printed. This prevents false deletions when resuming a run that was started before this fix was applied.

**Decision:** The purge-on-resume approach was chosen over alternatives (e.g., staging directory per worker, write-ahead log) because:
- It requires no change to the hot write path — `flush_shard()` is untouched.
- It is idempotent — running it twice produces the same result.
- It is cheap — only reads `.done` markers (tiny JSON files) and deletes a small number of directories.
- The staging+flatten alternative would add a per-worker temp prefix and a post-pool merge step, adding complexity for a scenario (local testing) that is not the production path.

---

#### [BUG-P3] Manifest Missing Batches Completed in Prior Runs

**Symptom:**
After a two-run sequence (run 1 interrupted after 3 batches, run 2 completing the remaining 7), `manifest.json` listed only 7 `processed_files` — the batches completed in run 2. The 3 batches completed in run 1 were absent, making `total_shards` and `total_tokens` incorrect.

**Root cause:**
`all_stats` was accumulated only from the return values of the current run's pool workers. Batches that were already in `completed_set` at the start of run 2 were skipped by the pool (correct) but never added to `all_stats` (wrong). The manifest was written from `all_stats` alone.

**Fix:**
After the pool exits, load all `.done` markers and merge any batch whose `coreset_name` is not already in `all_stats`:
```python
final_markers = load_all_markers(s3, args.dst_uri)
current_run_names = {s.get("coreset_name") for s in all_stats}
for m in final_markers:
    cname = m.get("coreset_name")
    if cname and cname not in current_run_names:
        all_stats.append({
            "coreset_file":      m.get("coreset_file", cname),
            "coreset_name":      cname,
            "num_shards":        m.get("num_shards", 0),
            "total_tokens":      m.get("total_tokens", 0),
            "tokens_dropped":    m.get("tokens_dropped", 0),
            "shard_names":       m.get("shard_names", []),
            "elapsed_seconds":   m.get("elapsed_seconds", 0),
            "num_docs":          m.get("num_docs", 0),
            "num_source_files":  m.get("num_source_files", 0),
        })
        current_run_names.add(cname)
```

**Decision:** The `.done` marker was already the source of truth for resume correctness. Extending it to also carry the full stats dict (Change 2 from BUG-P2) made this fix trivial — no additional I/O, no new data structures. The manifest merge is a post-pool, single-threaded operation with no race conditions.

---

#### Halt-and-Resume Test Results (2026-03-05)

Validated locally using the small profile (10 batches) with `--shard-size-mb 0.025`:

| Run | Action | Outcome |
|-----|--------|---------|
| Run 1 | Ctrl+C after 3 batches completed | 3 `.done` markers written; shards 001–011 on disk; no `manifest.json` |
| Resume | Identical command re-run | `Resuming: 3 already complete, 7 remaining`; orphan purge executed (0 orphans in this sequence); shards 012–036 written |
| Validation | `validate_shards.py` on full output | **36/36 PASS** — all 8 checks green; `manifest.json` shows `processed_files: 10`, `total_shards: 36`, `total_tokens: 147456` |

---

## 2. Local Testing Strategy

### Step 1 — Create mock source parquets (no S3 needed)

The generator builds a 2-level mock dataset under `dataset/final/` using the real T1 files that
already exist locally at `dataset/source/t1_rawdata/normalized_data/source=C4/` (5 files available).

```python
# scripts/create_mock_sources.py — redesigned for 2-level architecture
#
# Source T1 files (read-only, never modified):
#   dataset/source/t1_rawdata/normalized_data/source=C4/part-0000{0..4}-8299c866-...parquet
#
# Generation steps:
# 1. Read T3 coresets from coresets/1B/; collect all unique source_doc_id groups
#    and the chunk_ids that belong to each group.
# 2. For each unique source_doc_id group, assign a real T1 filename from the pool above
#    (round-robin; if more groups than available files, create copies named part-00005-..., etc.)
#    t1_file_path = "source=C4/<assigned_t1_filename>"
# 3. Write mock T3 parquets to dataset/final/t3/ — same as real T3 but with
#    t1_file_path column added (value = t1_file_path assigned in step 2).
# 4. For each assigned T1 file:
#      - Copy it to dataset/final/t1/<t1_file_path>
#      - Replace the id column values with the chunk_ids from T3 that map to this file
#        (preserves all other columns — hash, text, domain, language, metadata, etc.)
#    This ensures T1.id == T3.chunk_id filter always succeeds.
#    One decoy row (original id not in T3) is kept as-is to verify filtering works.
```

```bash
python scripts/create_mock_sources.py \
  --profile        minimal|small|parallel \
  --t3-source-dir  dataset/source/t3_coresets \
  --t1-source-dir  dataset/source/t1_rawdata/normalized_data \
  --output-dir     dataset/final
```

Output structure:
```
dataset/final/
  t1/
    source=C4/
      part-00000-8299c866-c99b-45fc-92d0-4d8b5c1f7503-c000.zstd.parquet  ← ids replaced
      part-00001-8299c866-c99b-45fc-92d0-4d8b5c1f7503-c000.zstd.parquet  ← ids replaced
      ...  (one file per unique source_doc_id group in T3; copies created if needed)
  t3/
    selected_indices_part_shard000_batch000000.parquet  ← real T3 + t1_file_path column
    ...
```

### Step 2 — Minimal smoke test (single T3 batch file)

```bash
python tokenize_curriculum.py \
  --coreset-uri   dataset/final/t3/selected_indices_part_shard000_batch000000.parquet \
  --dst-uri       dataset/final/tok_out \
  --tokenizer-path ./tsai_131k_tokenizer \
  --t1-base-uri   dataset/final/t1 \
  --block-size    4096 \
  --shard-size-mb 512 \
  --num-proc      2 \
  --drop-remainder \
  --stage         1 \
  --tokenizer-version v1 \
  --tmp-dir       /tmp/tok_tmp
```

### Step 3 — Verify output

```bash
# Check structure
find dataset/final/tok_out -type f

# Verify token count math
python -c "
import numpy as np, json, pathlib
shard_dir = next(pathlib.Path('dataset/final/tok_out').rglob('shard_000'))
meta = json.load(open(shard_dir / 'metadata.json'))
tokens = np.fromfile(shard_dir / 'tokens.bin', dtype=np.uint32)
print('total_tokens match:', meta['total_tokens'] == len(tokens))
print('max token id valid:', tokens.max() < 131072)
print('EOS count:', (tokens == 130717).sum())
print('band:', meta.get('band'), 'domain:', meta.get('domain'))
print('tokenizer_hash present:', 'tokenizer_hash' in meta)
"
```

### Step 4 — Resume test

```bash
# Delete one shard to simulate interrupted run
python -c "
import shutil, pathlib
shard = next(pathlib.Path('dataset/final/tok_out').rglob('shard_000'))
shutil.rmtree(shard); print('Deleted:', shard)
"
# Re-run with same args — shard_000 regenerated, completed shards skipped
python tokenize_curriculum.py \
  --coreset-uri   dataset/final/t3/selected_indices_part_shard000_batch000000.parquet \
  --dst-uri       dataset/final/tok_out \
  --tokenizer-path ./tsai_131k_tokenizer \
  --t1-base-uri   dataset/final/t1 \
  --block-size    4096 \
  --shard-size-mb 512 \
  --num-proc      2 \
  --drop-remainder \
  --stage         1 \
  --tokenizer-version v1 \
  --tmp-dir       /tmp/tok_tmp
```

### Step 5 — Directory-level run with all mock T3 files

```bash
python tokenize_curriculum.py \
  --coreset-uri    dataset/final/t3 \
  --dst-uri        dataset/final/tok_out_full \
  --tokenizer-path ./tsai_131k_tokenizer \
  --t1-base-uri    dataset/final/t1 \
  --file-parallelism 4 \
  --num-proc       2 \
  --block-size     4096 \
  --shard-size-mb  512 \
  --drop-remainder \
  --stage          1 \
  --tokenizer-version v1 \
  --tmp-dir        /tmp/tok_tmp
```

### Verification Steps

1. **Local smoke test**: Run single-file tokenization against mock source data → verify `metadata.json` has all required fields → run `validate_shards.py`
2. **Resume test**: Delete one shard → re-run → verify shard regenerated, others skipped
3. **Parallel test**: Run `--file-parallelism 4` on 4 batch files → verify 4 separate output directories with correct per-shard metadata
4. **AWS dry run**: Launch c5.4xlarge spot → run 2-3 batch files only → validate output → terminate
5. **Full AWS run**: Launch c5.9xlarge spot → process all 133 batch files → run validation → terminate

---

## 3. Parallel Tokenization — Current State & Enhancement

### Current capability

- **Within each file**: `datasets.Dataset.map()` with `num_proc=min(args.num_proc, 4)` — parallel HF tokenization (up to 4 subprocess workers per batch).
- **Across files**: Sequential loop (`for idx, uri in enumerate(target_files)`) — **no parallelism**. 133 batch files processed one at a time.

### Enhancement: `--file-parallelism N`

Architecture change (added to `main()`):

```python
ctx = multiprocessing.get_context("spawn")  # safe with boto3 + HF tokenizers
with ctx.Pool(processes=file_parallelism) as pool:
    results = pool.map(worker_process_coreset, worker_inputs)
```

Worker function (`_worker_process_coreset`):

- Creates its own `boto3.client("s3")` after fork (not safe to share across processes)
- Loads its own `AutoTokenizer` instance
- Uses isolated `worker_tmp = os.path.join(tmp_dir, f"worker_{worker_id:03d}")`

### Memory per worker

- Source parquet in memory: ~50–200 MB
- `accumulated_blocks` (ShardWriter, pre-flush): up to 512 MB
- HF Dataset object: proportional to filtered subset
- **Peak per worker**: ~700–900 MB

### Recommended parallelism settings

| Instance | vCPU | RAM | `--file-parallelism` | `--num-proc` |
|----------|------|-----|---------------------|-------------|
| c5.4xlarge | 16 | 32 GB | 8 | 2 |
| **c5.9xlarge** | **36** | **72 GB** | **12** | **3** |
| r5.4xlarge | 16 | 128 GB | 8 | 2 |

### What `--file-parallelism N` Parallelizes — Exact Mechanism

The parameter name can be misleading. It does **not** parallelize T1 files within a single batch. It parallelizes **entire coreset batches** (one per T3 parquet file) against each other.

```
--file-parallelism 3
        │
        ▼
Pool(processes=3)         ← multiprocessing.get_context("spawn").Pool
        │
        ├── Worker 0  → processes coreset batch_000000 end-to-end
        │     ├── downloads T1 file(s) referenced by that coreset
        │     ├── filters rows (T1.id == T3.chunk_id)
        │     ├── tokenizes via dataset.map(..., num_proc=min(args.num_proc, 4))
        │     └── writes shards to <dst>/shards/shard_NNN/ (atomic counter)
        │
        ├── Worker 1  → processes coreset batch_000001 end-to-end  (parallel)
        │
        └── Worker 2  → processes coreset batch_000002 end-to-end  (parallel)
```

**`--num-proc` role:** Inside each worker, `dataset.map(..., num_proc=min(args.num_proc, 4))` is the HuggingFace internal parallelism for **tokenizing rows within a single T1 file**. It spawns up to 4 sub-processes per worker to tokenize the filtered row set faster.

**Full concurrency picture with `--file-parallelism 3 --num-proc 2`:**
```
Main process
 ├── Worker A (batch 0)  → HF tokenizer with min(2, 4) = 2 sub-procs
 ├── Worker B (batch 1)  → HF tokenizer with min(2, 4) = 2 sub-procs
 └── Worker C (batch 2)  → HF tokenizer with min(2, 4) = 2 sub-procs

Total OS processes = 1 main + 3 workers + up to 6 HF sub-procs = up to 10
```

**Why NTFS is slow with parallelism > 1:** All 3 workers write `shard_NNN/` subdirectories under the same `shards/` parent simultaneously. Windows NTFS serializes directory-entry updates under a shared parent, so concurrent `mkdir` calls to `shards/` serialize — causing ~2× per-batch slowdown locally. On S3 there is no such contention: concurrent `PutObject` calls to keys under the same prefix are fully independent.

**Token drop rate is unaffected by shard count.** `--drop-remainder` discards the tail partial block **once per batch** (at the very end, after all T1 files in that batch are exhausted). The drop is at most `block_size − 1 = 4095` tokens per batch regardless of how many shards that batch produces. Smaller `--shard-size-mb` creates more shards but does not increase token drops.

---

### Known Platform Behavior: Local Filesystem Contention (Windows / NTFS)

When running with `--file-parallelism > 1` against a **local (non-S3) destination**, all workers
write their output shards to the same `<dst>/shards/` parent directory concurrently.  On Windows
NTFS this causes filesystem-level serialization of directory-entry updates, resulting in
approximately **2× slower throughput per batch** compared to sequential execution.

**Observed numbers (small profile, 10 batches, `--file-parallelism 3`, Windows 11):**

| Mode | Workers | Per-batch time | Total time |
|------|---------|----------------|------------|
| Sequential | 1 | ~55 s | ~550 s |
| Parallel (direct write) | 3 | ~110 s | ~183 s |

The parallel run is still ~3× faster wall-clock than sequential despite the per-batch slowdown,
because 3 batches run simultaneously.  However, per-core throughput drops by ~2×.

**This does NOT affect S3 runs.**  S3 is an object store with no directory-lock semantics —
concurrent `PutObject` calls to keys under the same prefix are fully parallel and independent.
The observed contention is entirely a local-NTFS artifact.

**The same behavior is expected on macOS HFS+/APFS** when multiple processes create
subdirectories under the same parent simultaneously (though APFS contention is generally lighter
than NTFS).

**Recommendation:** For local testing, the 2× slowdown is acceptable — local runs use tiny
profiles and finish in minutes.  For production (S3, EC2), parallel throughput is unaffected.
If local parallel throughput becomes a bottleneck, each worker could be given an isolated
per-batch staging directory; however, this adds a flatten step and is not worth implementing
for a test-only path.

---

## 4. AWS Deployment Guide

> **Platform prep** (S3 bucket creation, IAM role, EC2 launch commands) is in [Appendix A](#appendix-a-s3-setup--iam) and [Appendix B](#appendix-b-ec2-launch--run). This section covers strategy, feasibility, instance selection, and cost only.

---

### 4.1 Scale & Feasibility Analysis

#### Actual Dataset Numbers (Stage 1B)

| Metric | Value |
|--------|-------|
| T3 coreset batch files | 12,231 |
| T3 total rows (documents) | 19,882,213 |
| Total tokens (after filtering) | **9,895,522,858 (~9.896B)** |
| Unique T1 source parquets | 25,420 |
| T1 total size on S3 | 4,305 GB |
| T1 avg / max file size | 173 MB / **2.68 GB** |
| Avg T1 files per T3 batch | ~2.1 |
| Output shards at 512 MB | **~74** |
| Output shards at 1 GB | **~37** |
| Output shards at 2 GB | **~19** |
| Total output size (all shard sizes) | **~40–42 GB** |

> Output size is constant across shard sizes — it only changes how many files the training dataloader opens.

#### Can the Script Handle This Scale?

**Yes — no modifications required.** Confirmed scalability at each layer:

| Concern | Scale at 12,231 files | How it's handled |
|---------|-----------------------|-----------------|
| Pool task queue | 12,231 `apply_async` submissions | Python multiprocessing handles this without issue; `Pool(N)` processes N concurrently |
| `.done` marker loading on resume | 12,231 S3 `GetObject` calls (~13 paginated `list_objects_v2` pages) | One-time cost of ~30–90 seconds on resume startup; acceptable |
| `progress_state.json` size | 12,231 completed URI strings ≈ 1.5–3 MB JSON | Trivial for S3 read/write |
| Shard counter scanning | Only ~74 shard folder names listed (not contents) | `get_next_shard_idx_from_existing` uses S3 `Delimiter` listing, very fast |
| Output shard count | 74 shards at 512 MB | Minimal; Megatron loaders handle thousands of shards |

#### Why Single Instance (Not Distributed)

- S3 → EC2 intra-region reads are **free** and fast (10–25 Gbps ENA networking)
- The bottleneck is **CPU tokenization** (BPE is Rust-based, memory-bound, not I/O-bound)
- A single instance with `--file-parallelism 8–10` saturates all cores
- Distributed coordination (EMR/ECS) would add hours of setup for a ~5-hour job
- The checkpoint system already handles Spot interruptions natively — no orchestration needed

#### Why Not GPU Instances?

BPE tokenization (HuggingFace `tokenizers` library, Rust-based) is **CPU-only**. There is no CUDA-accelerated BPE implementation available. GPU instances (g4dn, p3, p4) would provide **zero throughput improvement** and cost significantly more than CPU-optimized instances.

The correct scaling axis is **vCPU count**, not GPU cores.

---

### 4.2 Instance Selection & Efficiency

#### Memory Constraint: The 2.68 GB T1 File

This is the most important sizing consideration. The pipeline reads each T1 file fully into RAM as a pandas DataFrame before filtering. Parquet with zstd typically expands 3–6× in memory:

- **Average T1 file**: 173 MB compressed → ~600 MB–1 GB in RAM
- **Largest T1 file**: 2.68 GB compressed → **~8–16 GB in RAM**

Per-worker peak memory = T1 DataFrame + HF Dataset (filtered subset) + ShardWriter accumulated_blocks buffer:

| `--shard-size-mb` | ShardWriter buffer/worker | Per-worker peak (avg) | Per-worker peak (worst — 2.68 GB T1) |
|-------------------|--------------------------|----------------------|--------------------------------------|
| 512 MB | up to 512 MB | ~1.5–2 GB | ~9–17 GB |
| 1 GB | up to 1 GB | ~2–3 GB | ~10–18 GB |
| 2 GB | up to 2 GB | ~3–4 GB | ~11–19 GB |

**Implication**: with `--file-parallelism 12` and 512 MB shards, worst-case RAM = 12 × 17 GB = 204 GB — far beyond any single instance. However, the probability of all 12 workers simultaneously loading the 2.68 GB outlier file is extremely low. With `--file-parallelism 8`, the realistic peak is 8 × 2 GB = 16 GB (typical) with occasional spikes.

**Recommendation: use `--file-parallelism 8` or less** on instances with ≤ 96 GB RAM.

#### Recommended Instance Configurations

Duration basis: 12,231 batches ÷ `--file-parallelism` = pool rounds × ~13 s/round (download + tokenize + write).

| Instance | vCPU | RAM | `--file-parallelism` | `--num-proc` | Est. Duration | Spot $/hr | Cost (512 MB) | Cost (1 GB) | Cost (2 GB) |
|----------|------|-----|---------------------|-------------|--------------|-----------|--------------|------------|------------|
| c5.4xlarge | 16 | 32 GB | 4 | 4 | ~14 hrs | $0.15–0.22 | $2.10–3.08 | same | same |
| c5.9xlarge | 36 | 72 GB | 8 | 4 | ~6 hrs | $0.35–0.55 | $2.10–3.30 | same | same |
| c5.18xlarge | 72 | 144 GB | 16 | 4 | ~3 hrs | $0.65–1.00 | $1.95–3.00 | same | same |
| **c6a.12xlarge** | **48** | **96 GB** | **10** | **4** | **~4.5 hrs** | **$0.25–0.40** | **~$1.13–1.80** | **same** | **same** |
| c6i.12xlarge | 48 | 96 GB | 10 | 4 | ~4.5 hrs | $0.30–0.50 | $1.35–2.25 | same | same |

> **Cost is identical across all shard sizes** — shard size affects output file count and peak RAM, not tokenization speed or wall time.

**c6a.12xlarge Spot is the recommended choice:** 48 vCPU, 96 GB RAM (comfortable margin for large T1 files with 10 workers), AMD EPYC (competitive BPE throughput), ~30–40% cheaper per vCPU than c5.9xlarge.

**c5.9xlarge** is a solid fallback with historically higher Spot availability in us-east-1.

#### Shard Size Recommendation

| Shard size | Shards | RAM impact | Recommendation |
|------------|--------|-----------|----------------|
| 512 MB | 74 | Lowest | **Default — use this.** Megatron standard; compatible with all training tooling |
| 1 GB | 37 | +~500 MB/worker | Use only if training team explicitly requests fewer shard files |
| 2 GB | 19 | +~1.5 GB/worker | Minimal additional benefit; risk of tight RAM on smaller instances |

#### Recommended Production Command

```bash
python tokenize_curriculum.py \
  --coreset-uri    s3://t2-datacurriculum-353/coreset_outputs/coresets/1B \
  --dst-uri        s3://your-training-bucket/tokenized/run_$(date +%Y%m%d) \
  --tokenizer-path ./tsai_131k_tokenizer \
  --t1-base-uri    s3://t1-dataacquisition-datasets/processed_dataset/normalized_data \
  --block-size     4096 \
  --shard-size-mb  512 \
  --num-proc       4 \
  --file-parallelism 10 \
  --drop-remainder \
  --stage          1 \
  --tokenizer-version v1 \
  --tmp-dir        /tmp/tok_tmp \
  2>&1 | tee ~/tokenize_$(date +%Y%m%d_%H%M%S).log
```

> Adjust `--file-parallelism` to match your instance: 8 for c5.9xlarge, 10 for c6a.12xlarge/c6i.12xlarge, 16 for c5.18xlarge.

---

### 4.3 Cost & Duration Breakdown

#### S3 Data Flow

```
S3 us-east-1 (T3 — read-only, already exists):
  s3://t2-datacurriculum-353/coreset_outputs/coresets/1B/
    12,231 × selected_indices_*.parquet   (~few GB total)

S3 us-east-1 (T1 — read-only, already exists):
  s3://t1-dataacquisition-datasets/processed_dataset/normalized_data/
    25,420 × part-*.zstd.parquet          (~4,305 GB total)
    avg 173 MB · max 2.68 GB

S3 us-east-1 (output — created during run):
  s3://your-training-bucket/tokenized/run_YYYYMMDD/
    progress_state.json                   ← resume state (updated after each batch)
    manifest.json                         ← global summary (written on clean completion)
    completed/
      selected_indices_*_batch000000.done ← per-batch completion marker (12,231 files)
      ...
    shards/
      shard_001/
        tokens.bin    ← uint32 token IDs (512 MB per shard)
        tokens.idx    ← spdl-compatible binary byte offsets
        metadata.json ← per-shard metadata (tokenizer_hash, band, domain, source_file, ...)
      shard_002/ … shard_074/             (~74 shards total at 512 MB)
```

#### Throughput Model

- T1 download per batch: ~2.1 files × 173 MB / 100 MB/s per worker ≈ **3–4 seconds**
- Tokenization per batch: ~809K tokens / (3M tokens/min/worker × 4 HF procs) ≈ **8–10 seconds**
- Shard write: few MB to S3 ≈ **< 1 second** (overlaps with next download)
- **Per batch: ~12–15 seconds end-to-end**
- S3 intra-region transfer: 4,305 GB / (10 workers × 100 MB/s) ≈ **72 minutes** (overlaps fully with tokenization — not on critical path)

#### Duration Estimates (9.896B tokens)

| Instance | `--file-parallelism` | Pool rounds | Est. wall time |
|----------|---------------------|-------------|----------------|
| c5.4xlarge | 4 | 3,058 | ~12–14 hrs |
| c5.9xlarge | 8 | 1,529 | ~5–6 hrs |
| **c6a.12xlarge** | **10** | **1,224** | **~4–5 hrs** |
| c5.18xlarge | 16 | 765 | ~2.5–3 hrs |

#### Output Size (Post-Processing)

Every shard directory (`shards/shard_NNN/`) contains three files:

| File | Per-shard size | 74 shards total | Description |
|------|---------------|-----------------|-------------|
| `tokens.bin` | 512 MB (exact) | **37.9 GB** | uint32 token IDs, packed 4096-token blocks |
| `tokens.idx` | ~2–3 MB | **~150–220 MB** | spdl-compatible binary index: block count, byte offsets, doc offsets |
| `metadata.json` | ~2–5 KB | **~150–370 KB** | tokenizer hash, band/domain distribution, shard stats |

Runtime artifacts (at `<dst>/`):

| File | Size | Description |
|------|------|-------------|
| `.done` markers (12,231 files) | ~12–24 MB total | Per-batch completion markers with token/row stats |
| `progress_state.json` | ~1.5–3 MB | Resume state; lists completed batch URIs |
| `manifest.json` | ~few KB | Global summary (total shards, tokens, processed files) |

**Total output: ~38–40 GB** (dominated by `tokens.bin`; all other artifacts are < 250 MB combined)

> Same total output size regardless of `--shard-size-mb` — only the file count and per-shard size change.

#### Cost Estimates (us-east-1, Spot)

| Component | c5.9xlarge (6 hrs) | c6a.12xlarge (4.5 hrs) | c5.18xlarge (3 hrs) |
|-----------|-------------------|------------------------|---------------------|
| Compute (Spot) | $2.10–3.30 | **$1.13–1.80** | $1.95–3.00 |
| EBS gp3 100 GB | $0.05 | $0.05 | $0.05 |
| S3 intra-region reads (T1, T3) | **$0.00** | **$0.00** | **$0.00** |
| S3 PUT — shard files (222 PUTs) | $0.001 | $0.001 | $0.001 |
| S3 PUT — `.done` markers (12,231) | $0.06 | $0.06 | $0.06 |
| S3 PUT — `progress_state.json` (~12,231 updates) | $0.06 | $0.06 | $0.06 |
| S3 GET — T1 GetObject (25,420) | $0.01 | $0.01 | $0.01 |
| **Total one-time run cost** | **~$2.28–3.48** | **~$1.30–1.97** | **~$2.12–3.17** |
| S3 output storage — Standard (~39 GB/month) | $0.90/mo | $0.90/mo | $0.90/mo |
| S3 output storage — Infrequent Access (~39 GB/month) | $0.49/mo | $0.49/mo | $0.49/mo |

> **S3 API cost breakdown**: ~24,685 PUT requests × $0.005/1000 ≈ **$0.12**; ~25,420 GET requests × $0.0004/1000 ≈ **$0.01**. Total S3 API overhead per run: **~$0.13**. S3 data transfer is $0.00 for EC2→S3 within us-east-1.

---

### 4.4 Spot Instance Interrupt Handling

#### How AWS Spot Interrupts Work

1. AWS reclaims capacity → **2-minute warning** via IMDS: `GET http://169.254.169.254/latest/meta-data/spot/termination-time` returns HTTP 200
2. **SIGTERM** sent ~30 seconds before hard shutdown
3. Instance terminated

#### What `tokenize_curriculum.py` Does on Interruption

**Layer 1 — IMDS polling thread**: polls every 5 s; sets `_TERMINATION_DETECTED` event on HTTP 200.

**Layer 2 — SIGTERM handler**: registered at startup; sets the same event. Also captures SIGINT (Ctrl+C) for local testing.

**Layer 3 — Graceful shutdown**:
- The `apply_async` polling loop calls `pool.terminate()` within 2 seconds of detection
- Workers mid-batch: discard partial `accumulated_blocks` — only fully uploaded shards (those with `metadata.json`) are kept
- Fully completed batches already have `.done` markers → skipped on resume
- Orphan shards (from interrupted workers) are purged automatically on the next resume startup (`purge_orphan_shards()`)

#### Resume at 12,231-File Scale

Re-running with identical arguments resumes safely:
1. Reads `progress_state.json` → skips completed batch URIs
2. Loads all `.done` markers (~13 paginated S3 list calls + JSON reads, ~30–90 s one-time cost)
3. Purges any orphan shards from prior interrupted workers
4. Initialises shard counter from `max(existing shard numbers) + 1`
5. Submits only pending batches to the pool

**At <5% Spot interruption probability for a 5-hour job:**
- Expected extra cost per run: 5% × ~$0.45 (one Spot-hour) ≈ **~$0.02**
- Work lost per interruption: at most 1 partially-accumulated shard; all completed batches preserved

#### Manual Resume After Interruption

```bash
# Check completed batch count
aws s3 cp s3://${BUCKET}/tokenized/${RUN_ID}/progress_state.json - | \
  python3 -c "import json,sys; s=json.load(sys.stdin); \
  print(f'Completed: {len(s[\"completed\"])} / 12231 batches')"

# Re-run with identical arguments — resumes automatically
python tokenize_curriculum.py \
  --coreset-uri    s3://t2-datacurriculum-353/coreset_outputs/coresets/1B \
  --dst-uri        s3://${BUCKET}/tokenized/${RUN_ID} \
  # ... same args as original run
```

---

## Quick Reference

### Recommended Command (Production Run — c6a.12xlarge)

```bash
python tokenize_curriculum.py \
  --coreset-uri    s3://t2-datacurriculum-353/coreset_outputs/coresets/1B \
  --dst-uri        s3://your-training-bucket/tokenized/run_$(date +%Y%m%d) \
  --tokenizer-path ./tsai_131k_tokenizer \
  --t1-base-uri    s3://t1-dataacquisition-datasets/processed_dataset/normalized_data \
  --block-size     4096 \
  --shard-size-mb  512 \
  --num-proc       4 \
  --file-parallelism 10 \
  --drop-remainder \
  --stage          1 \
  --tokenizer-version v1 \
  --tmp-dir        /tmp/tok_tmp
```

### Validation Command (After Run)

```bash
python validate_shards.py \
  --shards-dir /tmp/tok_out_synced \
  --tokenizer-path ./tsai_131k_tokenizer \
  --verbose
```

### Cost Summary (9.896B tokens, Stage 1B)

| What | Value |
|------|-------|
| Recommended instance | c6a.12xlarge Spot, us-east-1 |
| Fallback instance | c5.9xlarge Spot (higher availability) |
| Estimated duration | ~4.5 hrs (c6a.12xlarge, 10 workers) |
| Compute cost (Spot) | **~$1.13–1.80** |
| S3 transfer cost | **$0.00** (intra-region EC2↔S3) |
| S3 API cost (PUTs + GETs per run) | **~$0.13** |
| **Total one-time run cost** | **~$1.30–1.97** |
| Output size | **~38–40 GB** (74 × tokens.bin + idx + markers) |
| Output storage — S3 Standard | ~$0.90/month |
| Output storage — S3 Infrequent Access | ~$0.49/month |
| Spot interruption risk | <5% for 5-hour job in us-east-1 |
| Expected extra cost from interruption | **~$0.02** |

---

## Appendix A: S3 Setup & IAM

### IAM Policy for EC2 Instance

Save as `tokenization-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadCoresetIndex",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:HeadObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::t2-datacurriculum-353",
        "arn:aws:s3:::t2-datacurriculum-353/coreset_outputs/*"
      ]
    },
    {
      "Sid": "ReadRawTextData",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:HeadObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::t1-dataacquisition-datasets",
        "arn:aws:s3:::t1-dataacquisition-datasets/processed_dataset/normalized_data/*"
      ]
    },
    {
      "Sid": "ReadWriteTrainingBucket",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject", "s3:PutObject", "s3:HeadObject",
        "s3:ListBucket", "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::your-training-bucket",
        "arn:aws:s3:::your-training-bucket/*"
      ]
    }
  ]
}
```

### Create IAM Role and Instance Profile (one-time)

```bash
cat > /tmp/ec2-trust.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]
}
EOF

aws iam create-role \
  --role-name TokenizationInstanceRole \
  --assume-role-policy-document file:///tmp/ec2-trust.json

aws iam create-instance-profile --instance-profile-name TokenizationRole
aws iam add-role-to-instance-profile \
  --instance-profile-name TokenizationRole \
  --role-name TokenizationInstanceRole

aws iam put-role-policy \
  --role-name TokenizationInstanceRole \
  --policy-name TokenizationPolicy \
  --policy-document file:///tmp/tokenization-policy.json
```

### Destination Bucket Setup (one-time)

```bash
BUCKET="your-training-bucket"
aws s3api create-bucket --bucket ${BUCKET} --region us-east-1
aws s3api put-public-access-block --bucket ${BUCKET} \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
aws s3api put-bucket-encryption --bucket ${BUCKET} \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
```

### Verify Access

```bash
aws sts get-caller-identity
aws s3 ls s3://t2-datacurriculum-353/coreset_outputs/coresets/1B/ --region us-east-1
aws s3 ls s3://t1-dataacquisition-datasets/processed_dataset/normalized_data/ --region us-east-1
```

### Upload Tokenizer and Code

```bash
BUCKET="your-training-bucket"
aws s3 sync tsai_131k_tokenizer/ s3://${BUCKET}/tsai_131k_tokenizer/ --region us-east-1
aws s3 sync . s3://${BUCKET}/tokenizer-code/ --region us-east-1 \
  --exclude ".git/*" --exclude "*.pyc" --exclude "__pycache__/*" \
  --exclude "dataset/*" --exclude "tsai_131k_tokenizer/*"
```

### metadata.json Schema (per shard)

```json
{
  "format": "megatron_bin_idx",
  "idx_format": "spdl_v1",
  "token_dtype": "uint32",
  "bytes_per_token": 4,
  "block_size": 4096,
  "vocab_size": 131072,
  "pad_token_id": 130718,
  "eos_token_id": 130717,
  "num_blocks": 32768,
  "total_tokens": 134217728,
  "file_size_bytes": 536870912,
  "shard_name": "shard_001",
  "tokenizer_hash": "sha256-of-tokenizer.json+special_tokens_map.json",
  "tokenizer_version": "v1",
  "band": "B0",
  "band_distribution": {"B0": 0.574, "B1": 0.187, "B2": 0.239},
  "domain": "web",
  "domain_distribution": {"web": 1.0},
  "stage": 1,
  "source_file": "s3://t2-datacurriculum-353/coreset_outputs/coresets/1B/selected_indices_*_batch000000.parquet",
  "rows_input": 245000,
  "rows_with_eos": 244997,
  "rows_dropped": 3,
  "tokens_dropped": 8192,
  "drop_reason": "tail_truncation_at_block_boundary",
  "created_at": "2026-03-06T10:00:00Z"
}
```

---

## Appendix B: EC2 Launch & Run

### Launch EC2 Spot Instance

```bash
BUCKET="your-training-bucket"
RUN_ID="run_$(date +%Y%m%d)"
KEY_NAME="your-key-pair"
SUBNET_ID="subnet-xxxxxxxx"
SG_ID="sg-xxxxxxxx"

aws ec2 run-instances \
  --region us-east-1 \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type c6a.12xlarge \
  --key-name ${KEY_NAME} \
  --security-group-ids ${SG_ID} \
  --subnet-id ${SUBNET_ID} \
  --iam-instance-profile Name=TokenizationRole \
  --block-device-mappings '[{
    "DeviceName": "/dev/xvda",
    "Ebs": {"VolumeSize": 100, "VolumeType": "gp3", "Iops": 3000, "DeleteOnTermination": true}
  }]' \
  --instance-market-options '{
    "MarketType": "spot",
    "SpotOptions": {"SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}
  }' \
  --tag-specifications \
    'ResourceType=instance,Tags=[{Key=Name,Value=tokenization-run},{Key=RunId,Value='${RUN_ID}'}]' \
  --query 'Instances[0].InstanceId' --output text
```

> If `c6a.12xlarge` Spot capacity is unavailable, substitute `c5.9xlarge` or `c6i.12xlarge`.

### SSH, Bootstrap, and Run

```bash
# SSH
INSTANCE_IP=$(aws ec2 describe-instances \
  --instance-ids i-xxxxxxxxxxxxxxxxx \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
ssh -i ${KEY_NAME}.pem ec2-user@${INSTANCE_IP}

# Bootstrap (run once on instance)
BUCKET="your-training-bucket"
RUN_ID="run_$(date +%Y%m%d)"
sudo yum install -y python3.11 python3.11-pip tmux htop 2>/dev/null || \
  sudo apt-get install -y python3 python3-pip tmux htop 2>/dev/null || true
pip3 install numpy pandas pyarrow transformers datasets boto3 botocore tokenizers
aws s3 sync s3://${BUCKET}/tokenizer-code/ ~/tokenizer/ --region us-east-1
aws s3 sync s3://${BUCKET}/tsai_131k_tokenizer/ ~/tokenizer/tsai_131k_tokenizer/ --region us-east-1
cd ~/tokenizer

# Run inside tmux (survives SSH disconnect)
tmux new -s tokenize
python tokenize_curriculum.py \
  --coreset-uri    s3://t2-datacurriculum-353/coreset_outputs/coresets/1B \
  --dst-uri        s3://${BUCKET}/tokenized/${RUN_ID} \
  --tokenizer-path ./tsai_131k_tokenizer \
  --t1-base-uri    s3://t1-dataacquisition-datasets/processed_dataset/normalized_data \
  --block-size     4096 \
  --shard-size-mb  512 \
  --num-proc       4 \
  --file-parallelism 10 \
  --drop-remainder \
  --stage          1 \
  --tokenizer-version v1 \
  --tmp-dir        /tmp/tok_tmp \
  2>&1 | tee ~/tokenize_$(date +%Y%m%d_%H%M%S).log
# Detach: Ctrl+B then D  |  Reattach: tmux attach -t tokenize
```

### Monitor Progress

```bash
tail -f ~/tokenize_*.log
aws s3 ls s3://${BUCKET}/tokenized/${RUN_ID}/shards/ --recursive | grep metadata.json | wc -l
aws s3 cp s3://${BUCKET}/tokenized/${RUN_ID}/progress_state.json - | \
  python3 -c "import json,sys; s=json.load(sys.stdin); print(f'Completed: {len(s[\"completed\"])}/12231')"
```

### Validate Output

```bash
# Sync metadata only for fast check
aws s3 sync s3://${BUCKET}/tokenized/${RUN_ID}/ /tmp/tok_synced/ \
  --exclude "*.bin" --include "*/metadata.json"
python validate_shards.py --shards-dir /tmp/tok_synced --tokenizer-path ./tsai_131k_tokenizer --verbose
```

### Terminate Instance

```bash
# Only after validating output
aws ec2 terminate-instances --instance-ids i-xxxxxxxxxxxxxxxxx --region us-east-1
```

### Auto-Restart with ASG (Optional — fully automated)

For unattended restart after Spot interruption:

```bash
# Create launch template
aws ec2 create-launch-template \
  --launch-template-name tokenization-lt \
  --launch-template-data '{
    "InstanceType": "c6a.12xlarge",
    "ImageId": "ami-0c02fb55956c7d316",
    "IamInstanceProfile": {"Name": "TokenizationRole"},
    "KeyName": "your-key-pair",
    "BlockDeviceMappings": [{"DeviceName": "/dev/xvda",
      "Ebs": {"VolumeSize": 100, "VolumeType": "gp3", "Iops": 3000}}],
    "UserData": "'$(base64 -w 0 userdata_auto_restart.sh)'"
  }'

# Create ASG — min/max/desired all 1; scale to 0 when job finishes
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name tokenization-asg \
  --launch-template LaunchTemplateName=tokenization-lt,Version='$Latest' \
  --min-size 1 --max-size 1 --desired-capacity 1 \
  --vpc-zone-identifier "subnet-xxxxxxxx" \
  --mixed-instances-policy '{
    "InstancesDistribution": {
      "OnDemandBaseCapacity": 0, "OnDemandPercentageAboveBaseCapacity": 0,
      "SpotAllocationStrategy": "capacity-optimized"
    },
    "LaunchTemplate": {"LaunchTemplateSpecification":
      {"LaunchTemplateName": "tokenization-lt", "Version": "$Latest"},
      "Overrides": [{"InstanceType": "c6a.12xlarge"}, {"InstanceType": "c5.9xlarge"},
                    {"InstanceType": "c6i.12xlarge"}]}
  }'

# Scale down to 0 once manifest.json is confirmed on S3
aws autoscaling set-desired-capacity --auto-scaling-group-name tokenization-asg --desired-capacity 0
```

**`userdata_auto_restart.sh`:**

```bash
#!/bin/bash
set -e
BUCKET="your-training-bucket"
RUN_ID="run_20260306"   # Fixed: same RUN_ID always resumes the same run

yum install -y python3.11 python3.11-pip tmux 2>/dev/null || \
  apt-get install -y python3 python3-pip tmux 2>/dev/null || true
pip3 install numpy pandas pyarrow transformers datasets boto3 botocore tokenizers

aws s3 sync s3://${BUCKET}/tokenizer-code/ /home/ec2-user/tokenizer/ --region us-east-1
aws s3 sync s3://${BUCKET}/tsai_131k_tokenizer/ /home/ec2-user/tokenizer/tsai_131k_tokenizer/ --region us-east-1
cd /home/ec2-user/tokenizer

python tokenize_curriculum.py \
  --coreset-uri    s3://t2-datacurriculum-353/coreset_outputs/coresets/1B \
  --dst-uri        s3://${BUCKET}/tokenized/${RUN_ID} \
  --tokenizer-path ./tsai_131k_tokenizer \
  --t1-base-uri    s3://t1-dataacquisition-datasets/processed_dataset/normalized_data \
  --block-size 4096 --shard-size-mb 512 --num-proc 4 --file-parallelism 10 \
  --drop-remainder --stage 1 --tokenizer-version v1 --tmp-dir /tmp/tok_tmp \
  2>&1 | tee /home/ec2-user/tokenize_$(date +%Y%m%d_%H%M%S).log

if [ $? -eq 0 ]; then
  aws s3 cp /home/ec2-user/tokenize_*.log s3://${BUCKET}/tokenized/${RUN_ID}/logs/ --region us-east-1
  INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
  aws ec2 terminate-instances --instance-ids ${INSTANCE_ID} --region us-east-1
fi
# Non-zero exit: instance stays up for investigation (continues billing — check manually)
```
