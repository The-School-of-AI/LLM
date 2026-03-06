# Folder Structure Change Analysis

**Date:** 2026-03-05

---

## User Request

> lets rename tok_out to 'shards', under which lets not have folders like - selected_indices_small_batch000000, selected_indices_small_batch000001 ...
> lets have continous folders like shard_001, shard_002
>
> Thoroughly analyze this for impacts.
> Also impacts to parallel processing.
> Impacts to interruption and resume

---

## Impact Analysis: Flat `shards/` Layout with Continuous Global Numbering

### Current structure (baseline)

```
<dst-uri>/                                    ← e.g. tok_out/
  progress_state.json
  manifest.json
  selected_indices_small_batch000000/          ← per-batch subdir (coreset stem)
    shard_001/ → tokens.bin, tokens.idx, metadata.json
    shard_002/ → ...
  selected_indices_small_batch000001/
    shard_001/ → ...
    shard_002/ → ...
```

### Proposed structure

```
<dst-uri>/shards/                             ← renamed + flattened
  shard_001/ → tokens.bin, tokens.idx, metadata.json
  shard_002/ → ...
  shard_003/ → ...
  shard_004/ → ...   ← batch 2 starts here (no separator)
  ...
  shard_NNN/ → ...
```

---

### Impact 1 — ShardWriter (Low complexity)

Two changes:

- **start_shard_idx parameter:** Currently `self.shard_idx = 1` hardcoded. Needs `start_shard_idx: int = 1` parameter so each batch starts from the global counter, not from 1.
- **dst_uri construction (line 547):** Currently appends `/{coreset_name}` when creating the writer. Must be removed — all batches write into the same flat dir.

```python
# Current (line 547):
dst_uri=f"{dst_base_uri.rstrip('/')}/{coreset_name}"

# New:
dst_uri=f"{dst_base_uri.rstrip('/')}/shards"
```

---

### Impact 2 — Sequential main loop (Medium complexity)

Needs a global shard counter that accumulates across batches:

```python
next_shard_idx = progress.get("next_shard_idx", 1)   # from progress_state on resume

for uri in pending_files:
    stats = process_coreset_file(..., start_shard_idx=next_shard_idx)
    next_shard_idx += stats["num_shards"]              # advance after each batch
    progress["next_shard_idx"] = next_shard_idx
    progress["completed"].append({"uri": uri, "shard_start": ..., "shard_end": ...})
    save_progress_state(...)
```

On resume, `next_shard_idx` is read from `progress_state.json` and passed to the first pending batch. Clean, deterministic.

---

### Impact 3 — `progress_state.json` schema (Medium complexity)

The current schema must evolve significantly because URIs alone are no longer sufficient to identify which shard dirs to clean up on interrupt:

**Current (with interrupt fix):**

```json
{ "completed": ["uri_A", "uri_B"], "interrupted": ["uri_C"] }
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

### Impact 4 — Interrupt + resume cleanup (Medium complexity)

With per-batch dirs (current), cleanup is simple:

```python
shutil.rmtree(f"{dst_uri}/{batch_stem}/")   # entire dir gone, shard counter rolls back
```

With flat numbering, cleanup requires deleting specific shard dirs by number:

```python
for idx in range(interrupted_entry["shard_start"], interrupted_entry["shard_end"] + 1):
    shard_dir = f"{dst_uri}/shards/shard_{idx:03d}"
    shutil.rmtree(shard_dir, ignore_errors=True)
# Then reset next_shard_idx = interrupted_entry["shard_start"]
```

Works correctly, but the interrupted section must now carry `shard_start`/`shard_end` (not just the URI).

---

### Impact 5 — Parallel path: FUNDAMENTAL BLOCKER ⛔

This is the critical problem. The flat structure is fundamentally incompatible with `--file-parallelism N`.

**Why it breaks:** Each parallel worker calls `process_coreset_file()` concurrently. All workers write to the same flat `shards/` directory. Workers don't know each other's shard counts in advance (because shard count depends on actual token count after filtering, which isn't known until tokenization completes).

| Worker | Batch | Shard count (unknown in advance) |
| :--- | :--- | :--- |
| Worker 0 | batch_000 | 3 shards |
| Worker 1 | batch_001 | 5 shards |
| Worker 2 | batch_002 | 2 shards |

Without pre-agreed ranges, two workers could both try to write `shard_004` simultaneously — collision and corrupted output.

**Possible mitigations:**

| Approach | How | Problem |
| :--- | :--- | :--- |
| Pre-allocate ranges | Estimate max shards/batch upfront; assign non-overlapping ranges | Gaps in numbering (e.g. batch_000 allocated 1–50 but only used 1–3; batch_001 starts at 51) |
| File-system locking | Workers atomically claim next shard number via lock file or S3 conditional put | Complex, S3 has no native locks; adds per-shard latency |
| Post-processing flatten | Workers write to temp per-batch dirs during execution; after pool completes, rename to continuous numbers | Extra pass; rename can fail partway (partial state) |
| Disable parallel for flat mode | Only allow flat structure when `--file-parallelism 1` | Loses production capability entirely |
| Keep per-batch dirs during parallel; flatten only for sequential | Different layout per mode | Inconsistent UX |

None of these are clean. The per-batch subdirectory exists precisely because it gives each parallel worker an isolated output namespace with no coordination needed.

---

### Impact 6 — `validate_shards.py` (Low — actually simpler)

Current `list_shard_dirs_local()` (lines 71–86) traverses two levels: `<base>/<coreset>/shard_NNN/`.

With flat structure it becomes one level — simpler:

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

### Impact 7 — `metadata.json` per shard (No change needed)

Each shard already carries `"source_file": "<coreset_batch_uri>"`. Even with flat numbering, traceability back to which coreset batch produced which shard is preserved at the file level. No change needed.

---

### Impact 8 — Manifest (Low complexity)

Add `shard_start`/`shard_end` to each `processed_files` entry so the batch → shard range mapping is explicit in the global manifest.

---

### Summary Table

| Layer | Complexity | Compatible with parallel? |
| :--- | :--- | :--- |
| ShardWriter init | Low | N/A |
| process_coreset_file | Low | ✅ (if ranges pre-assigned) |
| Sequential main loop | Medium | ✅ |
| Parallel main loop (pool.map) | Blocking | ❌ Race condition on shard numbers |
| progress_state.json schema | Medium | Schema change needed |
| Interrupt cleanup | Medium | Works if shard ranges tracked |
| validate_shards.py | Low (simpler) | ✅ |
| Manifest | Low | ✅ |

---

### Recommendation

The flat layout is desirable for UX — clean, simple, contiguous. But parallel processing is a hard blocker unless you choose one of these paths:

**Path A — Sequential only (simplest)**
Adopt the flat layout but drop `--file-parallelism` support (or restrict it to 1). The production AWS run would be sequential, which is viable for the current 133-batch scale but ~4× slower.

**Path B — Post-process flatten (clean output, parallel preserved)**
Parallel workers keep per-batch dirs during execution (isolation guaranteed). After `pool.map()` completes successfully, a flattening step renames all shards to continuous global numbers. Interrupted runs retain per-batch layout until next clean completion.

**Path C — Keep current structure (no change)**
The per-batch subdir IS the right structure for a multi-batch parallel pipeline. The naming (`selected_indices_small_batch000000`) is long but meaningful. A cosmetic simplification of the names (e.g. `batch_000000/shard_001/`) is possible without any of the parallel compatibility issues.

Which path do you prefer?

---

## User Decision — Path B

> I like Path B — Post-process flatten (clean output, parallel preserved)
> So you are suggesting use the current scheme as staging area and then move them all to different folder with the expected structure?
>
> 1. The metadata.json needs to also rewrite all the shard_name
> 2. Will I have additional cost in AWS by doing the file or folder movement from one to another within the same bucket?
> 3. My initial run is with 2B tokens and 512 mb shard size (shard size can be increased if there are no impacts. This is flexible.), eventually i will be running it upto 240B tokens. What will be the impacts of rewrite at this massive scale?

---

## Follow-up Q&A

Good confirmation — `"shard_name": "shard_005"` is explicitly stored in `metadata.json` (line 13). It must be rewritten during the flatten step.

### Q1 — Does `metadata.json` need to be rewritten?

Yes. Every `metadata.json` carries `"shard_name": "shard_005"` (local batch-relative number). After flattening, `batch_000001/shard_005/` might become `shards/shard_042/`. The field must change from `"shard_005"` → `"shard_042"`.

**Fields that change:**

| Field | Action |
| :--- | :--- |
| shard_name | Rewrite to new global name |
| source_file | No change — already has the coreset batch URI for traceability |

Fields that stay the same: all others — `tokenizer_hash`, `band`, `domain`, `total_tokens`, `rows_input`, etc. Only `shard_name` changes.

On S3: Read → modify → re-upload (3 API calls per shard: GET + PUT + no delete needed for metadata since it's overwritten in place at the new path after CopyObject).

On local: Read → modify in memory → write to new path → delete old. Or: rename directory then patch the `metadata.json` in place (cheapest).

---

### Q2 — AWS cost of moving files within the same bucket

S3 has no native rename/move. The operation is CopyObject (server-side) + DeleteObject.

| Cost component | Per shard | At 2B tokens (~15 shards) | At 240B tokens (~1,920 shards) |
| :--- | :--- | :--- | :--- |
| Data transfer (intra-region) | $0.00 | $0.00 | $0.00 |
| CopyObject for tokens.bin (counts as PUT) | $0.005/1,000 | ~$0.0001 | ~$0.010 |
| CopyObject for tokens.idx + metadata.json | $0.005/1,000 | ~$0.0001 | ~$0.010 |
| DeleteObject (old files) | Free | $0.00 | $0.00 |
| Temporary double-storage during copy (1 hr) | ~$0.023/GB/month × 1hr | ~$0.001 | ~$0.03 |
| **Total flatten cost** | | **<$0.001** | **~$0.05** |

**Verdict:** Negligible cost at any scale. The only real spend is the tiny API request fees — less than a coffee at 240B tokens.

**Key reason:** CopyObject is a server-side operation. AWS copies data internally between storage nodes. No data leaves the region, no egress is charged, and no data flows through your EC2 instance at all.

---

### Q3 — Scale impact of the flatten rewrite

Numbers at each scale (512 MB shards):

| Scale | Shards | .bin data | CopyObject calls | Flatten time (50 parallel) |
| :--- | :--- | :--- | :--- | :--- |
| 2B tokens | ~15 | 8 GB | 45 | <10 seconds |
| 20B tokens (current S3 run) | ~150 | 80 GB | 450 | ~1 minute |
| 240B tokens | ~1,920 | 960 GB | 5,760 | ~5–10 minutes |

The bottleneck is not data size — it's API concurrency. AWS S3 CopyObject for a 512 MB file completes in ~1–3 seconds server-side. With 50 parallel workers, 1,920 shards × 3 files / 50 = ~115 batches × 2 seconds ≈ 4 minutes.

**If shard size is increased (e.g. 2 GB or 5 GB):**

- Fewer shards → proportionally fewer API calls → faster flatten
- At 2 GB: 240B tokens = ~480 shards → flatten in ~1 minute
- At 5 GB: 240B tokens = ~192 shards → flatten in ~30 seconds
- No impact on data integrity or cost — only API call count changes
- Verify with training team that their data loader supports the larger shard size before changing

**One important caveat at 240B scale:**

During the flatten, you're temporarily storing both the staging layout and the flat output simultaneously — ~960 GB × 2 = ~1.92 TB. If your S3 bucket runs close to a quota, this brief doubling matters. The solution is to run flatten in streaming fashion: copy shard N, delete staging shard N, then move to N+1 (never hold both at once).

---

## Revised Path B Design

Given all the above, the proposed design is:

```
tokenize_curriculum.py  →  tok_staging/           (per-batch dirs, parallel-safe)
flatten_shards.py       →  shards/                (flat global numbering, post-processing)
validate_shards.py      →  reads shards/          (simple flat discovery)
```

`flatten_shards.py` does:

1. Discover all `batch_NNN/shard_NNN/` dirs under `tok_staging/`, sorted deterministically
2. Assign global `shard_index` starting from 1 (or from where a previous flatten left off — idempotent)
3. For each shard: CopyObject all 3 files → patch `shard_name` in `metadata.json` → DeleteObject originals
4. Write a `flatten_manifest.json` recording the batch → shard range mapping (audit trail)
5. Resumable: if `shards/shard_042/metadata.json` already exists, skip

For local runs (Windows/Linux dev): directory rename is O(1) — the flatten step is effectively instant at any scale because `os.rename()` just updates inode metadata, no data movement.

Shall I implement both fixes together — the interrupt/resume bug fix in `tokenize_curriculum.py` AND the new `flatten_shards.py` script?
