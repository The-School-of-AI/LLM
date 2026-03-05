# Folder Structure Change Analysis

**Date:** 2026-03-05
**Requested by:** User
**Scope:** `tokenize_curriculum.py`, `validate_shards.py`, `flatten_shards.py` (new)

---

## 1. Request

> "lets rename tok_out to 'shards', under which lets not have folders like
> selected_indices_small_batch000000, selected_indices_small_batch000001 ...
> lets have continuous folders like shard_001, shard_002
> Thoroughly analyze this for impacts.
> Also impacts to parallel processing.
> Impacts to interruption and resume"

---

## 2. Current vs Proposed Structure

### Current
```
<dst-uri>/                                          e.g. tok_out/
  progress_state.json
  manifest.json
  selected_indices_small_batch000000/               ← per-batch subdir (coreset stem)
    shard_001/ → tokens.bin, tokens.idx, metadata.json
    shard_002/ → ...
    shard_003/ → ...
  selected_indices_small_batch000001/
    shard_001/ → ...
    shard_005/ → ...
  ...
```

### Proposed
```
<dst-uri>/shards/                                   ← renamed + flattened
  shard_001/ → tokens.bin, tokens.idx, metadata.json
  shard_002/ → ...
  shard_003/ → ...
  shard_004/ → ...   ← batch 2 starts here (no separator)
  ...
  shard_NNN/ → ...
```

---

## 3. Impact Analysis by Layer

### 3.1 `ShardWriter` — Low complexity

Two changes required:

**a) Add `start_shard_idx` parameter**
Currently `self.shard_idx = 1` is hardcoded — every new batch starts at shard_001.
With global numbering, the writer needs to start at the current global counter.

```python
# Current (line 332):
self.shard_idx = 1

# New:
def __init__(self, ..., start_shard_idx: int = 1):
    self.shard_idx = start_shard_idx
```

**b) Remove per-batch subdir from `dst_uri` construction**

```python
# Current (line 547):
dst_uri=f"{dst_base_uri.rstrip('/')}/{coreset_name}"

# New:
dst_uri=f"{dst_base_uri.rstrip('/')}/shards"
```

---

### 3.2 Sequential main loop — Medium complexity

A global shard counter must persist across batches:

```python
next_shard_idx = progress.get("next_shard_idx", 1)   # restored on resume

for uri in pending_files:
    stats = process_coreset_file(..., start_shard_idx=next_shard_idx)
    next_shard_idx += stats["num_shards"]              # advance after each batch
    progress["next_shard_idx"] = next_shard_idx
    progress["completed"].append({
        "uri": uri,
        "shard_start": next_shard_idx - stats["num_shards"],
        "shard_end":   next_shard_idx - 1
    })
    save_progress_state(...)
```

On resume, `next_shard_idx` is read from `progress_state.json` and passed to the
first pending batch. Clean and deterministic.

---

### 3.3 Parallel path (`--file-parallelism N`) — FUNDAMENTAL BLOCKER ⛔

This is the critical problem. **The flat structure is fundamentally incompatible with
`--file-parallelism N`.**

**Why it breaks:**
Each parallel worker calls `process_coreset_file()` concurrently. All workers write to
the same flat `shards/` directory. Workers do not know each other's shard counts in
advance — the shard count per batch depends on actual token count after filtering, which
is not known until tokenization completes.

| Worker | Batch | Shard count (unknown in advance) |
|--------|-------|----------------------------------|
| Worker 0 | batch_000 | 3 shards |
| Worker 1 | batch_001 | 5 shards |
| Worker 2 | batch_002 | 2 shards |

Without pre-agreed non-overlapping ranges, two workers could both try to write
`shard_004` simultaneously — **collision and corrupted output**.

**Possible mitigations and their problems:**

| Approach | Problem |
|----------|---------|
| Pre-allocate ranges (estimate max shards/batch upfront) | Creates gaps in numbering; fragile |
| File-system locking (atomically claim next shard number) | Complex; S3 has no native locks; per-shard latency bottleneck |
| Post-process flatten (workers use staging dirs; rename after) | Extra pass; rename can fail partway |
| Disable parallel when flat mode active | Loses production capability entirely |
| Different layout per parallelism mode | Inconsistent UX |

**The per-batch subdirectory exists precisely because it gives each parallel worker
an isolated output namespace with zero coordination needed.**

---

### 3.4 `progress_state.json` schema — Medium complexity

The current schema uses URI strings as keys. With flat numbering, cleanup of
interrupted batches requires knowing which shard numbers to delete.

**Current (with interrupt fix applied):**
```json
{
  "completed":   ["uri_A", "uri_B"],
  "interrupted": ["uri_C"],
  "failed":      []
}
```

**Required with flat numbering:**
```json
{
  "completed": [
    {"uri": "uri_A", "shard_start": 1,  "shard_end": 3},
    {"uri": "uri_B", "shard_start": 4,  "shard_end": 8}
  ],
  "interrupted": [
    {"uri": "uri_C", "shard_start": 9,  "shard_end": 10}
  ],
  "next_shard_idx": 9,
  "failed": []
}
```

All logic that reads/writes this file must be updated.

---

### 3.5 Interrupt + resume cleanup — Medium complexity

**With per-batch dirs (current):**
```python
shutil.rmtree(f"{dst_uri}/{batch_stem}/")   # entire dir gone; shard counter rolls back
```

**With flat numbering:**
```python
# Must delete specific shard dirs by number
for idx in range(interrupted_entry["shard_start"], interrupted_entry["shard_end"] + 1):
    shard_dir = f"{dst_uri}/shards/shard_{idx:03d}"
    shutil.rmtree(shard_dir, ignore_errors=True)
# Reset global counter
next_shard_idx = interrupted_entry["shard_start"]
```

Works correctly, but the `interrupted` section must carry `shard_start`/`shard_end`
(not just the URI).

---

### 3.6 `validate_shards.py` — Low (actually simpler)

Current `list_shard_dirs_local()` (lines 71–86) traverses two levels:
`<base>/<coreset>/shard_NNN/`

With flat structure, one level — simpler:
```python
def list_shard_dirs_local(base_dir: str) -> List[str]:
    shards_dir = os.path.join(base_dir, "shards")
    return sorted([
        os.path.join(shards_dir, d)
        for d in os.listdir(shards_dir)
        if d.startswith("shard_")
    ])
```

---

### 3.7 `metadata.json` per shard — Change required

Each shard carries `"shard_name": "shard_005"` (local batch-relative number).
After flattening, `batch_000001/shard_005/` becomes `shards/shard_042/`.
The `shard_name` field must be updated.

| Field | Action |
|-------|--------|
| `shard_name` | **Rewrite** to new global name |
| `source_file` | No change — already has coreset batch URI for traceability |
| All other fields | No change |

---

### 3.8 Manifest — Low complexity

Add `shard_start`/`shard_end` to each `processed_files` entry:
```json
{
  "processed_files": [
    {
      "coreset_file": "batch_000000.parquet",
      "num_shards": 3,
      "shard_start": 1,
      "shard_end": 3,
      "total_tokens": 12288
    }
  ]
}
```

---

### 3.9 Impact summary table

| Layer | Complexity | Compatible with parallel? |
|-------|-----------|--------------------------|
| `ShardWriter` init | Low | N/A |
| `process_coreset_file` | Low | ✅ (if ranges pre-assigned) |
| Sequential main loop | Medium | ✅ |
| Parallel main loop (`pool.map`) | **Blocking** | ❌ Race condition on shard numbers |
| `progress_state.json` schema | Medium | Schema change required |
| Interrupt cleanup | Medium | Works if shard ranges tracked |
| `validate_shards.py` | Low (simpler) | ✅ |
| `metadata.json` `shard_name` | Low | ✅ (rewrite at flatten time) |
| Manifest | Low | ✅ |

---

## 4. Three Paths Forward

### Path A — Sequential only (simplest)
Adopt flat layout but drop `--file-parallelism` support (restrict to 1).
Production run is sequential — viable at 133 batches but ~4× slower.

### Path B — Post-process flatten (clean output, parallel preserved) ✅ CHOSEN
Parallel workers keep per-batch dirs during execution (isolation guaranteed).
After `pool.map()` completes, a separate `flatten_shards.py` script renames all
shards to continuous global numbers.

```
tokenize_curriculum.py  →  tok_staging/     (per-batch dirs, parallel-safe)
flatten_shards.py       →  shards/          (flat global numbering, post-processing)
validate_shards.py      →  reads shards/    (simple flat discovery)
```

Interrupted runs retain per-batch layout until next clean completion.

### Path C — Keep current structure (no change)
The per-batch subdir is the correct structure for a multi-batch parallel pipeline.
A cosmetic simplification of names (e.g. `batch_000000/shard_001/`) is possible
without any parallel compatibility issues.

---

## 5. Path B — Detailed Q&A

### Q1: Does `metadata.json` need to be rewritten?

**Yes.** `"shard_name": "shard_005"` is explicitly stored (confirmed from live
`metadata.json`). After flattening, this field must be updated to reflect the new
global shard name (e.g. `"shard_042"`). All other fields are unchanged.

On S3: GET metadata.json → patch → PUT to new path (3 API calls per shard).
On local: read → patch in memory → write to new path → delete old.
For local filesystem, `os.rename()` on the directory is O(1) — essentially instant
regardless of file sizes.

---

### Q2: AWS cost of moving files within the same bucket

S3 has no native rename. The operation is `CopyObject` (server-side) + `DeleteObject`.

| Cost component | At 2B tokens (~15 shards) | At 240B tokens (~1,920 shards) |
|----------------|--------------------------|-------------------------------|
| Data transfer (intra-region) | **$0.00** | **$0.00** |
| `CopyObject` PUTs (3 files/shard) | ~$0.0001 | ~$0.03 |
| `DeleteObject` (old files) | Free | Free |
| Temporary double-storage during copy (~1 hr) | ~$0.001 | ~$0.03 |
| **Total flatten cost** | **<$0.001** | **~$0.06** |

**Verdict: Negligible.** `CopyObject` is a server-side operation — no data flows
through EC2, no egress is charged. The only cost is API request fees.

---

### Q3: Scale impact at 2B → 240B tokens

**Shard counts at 512 MB/shard:**

| Scale | Shards | `.bin` data | CopyObject calls | Flatten time (50 parallel) |
|-------|--------|-------------|------------------|---------------------------|
| 2B tokens | ~15 | 8 GB | 45 | <10 seconds |
| 20B tokens | ~150 | 80 GB | 450 | ~1 minute |
| 240B tokens | ~1,920 | 960 GB | 5,760 | ~5–10 minutes |

The bottleneck is API concurrency, not data size. S3 `CopyObject` for a 512 MB file
completes in ~1–3 seconds server-side. With 50 parallel workers:
1,920 shards / 50 × ~2 seconds = ~4 minutes at 240B scale.

**If shard size is increased (e.g. 2 GB or 5 GB):**

| Shard size | Shards at 240B | Flatten time |
|------------|---------------|--------------|
| 512 MB | ~1,920 | ~5–10 min |
| 2 GB | ~480 | ~1–2 min |
| 5 GB | ~192 | ~30 sec |

Increasing shard size reduces API call count proportionally with no cost or
integrity impact. Verify with training team that their data loader supports
larger shards before changing from 512 MB.

**Important caveat at 240B scale:**
During flatten, both staging dirs and flat dirs exist simultaneously (~1.92 TB
temporarily). To avoid doubling peak storage, run the flatten script in streaming
mode: copy shard N → delete staging shard N → advance to N+1.

---

## 6. `flatten_shards.py` — Design Specification

```
flatten_shards.py <src-staging-dir> <dst-shards-dir> [--delete-src] [--parallel N]
```

**Algorithm:**
1. Discover all `batch_NNN/shard_NNN/` dirs under `<src-staging-dir>`, sorted
   deterministically by (batch name, shard name).
2. Assign global `shard_index` starting from 1 (or from `max existing + 1` for
   idempotent re-run).
3. For each shard:
   - `CopyObject` / `shutil.copy2` all 3 files to `<dst-shards-dir>/shard_{N:03d}/`
   - Read `metadata.json`, update `shard_name`, write to destination
   - If `--delete-src`: `DeleteObject` / `os.remove()` originals
4. Write `flatten_manifest.json` recording batch → shard range mapping (audit trail).

**Idempotent:** if `shards/shard_042/metadata.json` already exists at destination, skip.

**For local filesystem:** use `os.rename()` on the shard directory (O(1), no data
movement). Only `metadata.json` needs an in-place patch for `shard_name`.
