# Tokenization Granularity & Sharding Strategy

## Question

> If we pre-tokenize, should we tokenize across parquet files or within the same parquet file? How do we decide?

This is about the **granularity boundary** of tokenization relative to your source files. There are three possible strategies, and the right choice depends on what happens *after* tokenization.

---

## The Three Strategies

### Strategy A: 1 Parquet → 1 Tokenized File (same boundary)

```
parquet-00000.parquet  →  tokenize  →  tokens-00000.npy
parquet-00001.parquet  →  tokenize  →  tokens-00001.npy
parquet-00002.parquet  →  tokenize  →  tokens-00002.npy
```

Each parquet file produces exactly one `.npy` file. The boundaries stay the same.

### Strategy B: N Parquets → 1 Token Stream → Re-shard into fixed-size outputs

```
parquet-00000.parquet ─┐
parquet-00001.parquet ─┼→ tokenize → continuous token stream → shard → shard-00000.npy (500M tokens)
parquet-00002.parquet ─┤                                              shard-00001.npy (500M tokens)
parquet-00003.parquet ─┘                                              shard-00002.npy (500M tokens)
```

All parquets are tokenized into one logical stream, then *re-sharded* into **uniform-sized** output files.

### Strategy C: 1 Parquet → tokenize in-place (add token column to parquet)

```
parquet-00000.parquet  →  tokenize  →  parquet-00000-tokenized.parquet
  (text column)                          (text column + token_ids column)
```

Keep the parquet format but add a `token_ids` column.

---

## Comparison — Pros and Cons

| Aspect | **A: 1:1 mapping** | **B: Stream + re-shard** | **C: In-place parquet** |
|--------|:---:|:---:|:---:|
| **Shard size uniformity** | ❌ Varies wildly (parquets are different sizes, Hindi vs English tokenize differently) | ✅ Every shard is exactly 500M tokens | ❌ Varies (same as source parquets) |
| **Parallelizable tokenization** | ✅ Trivially parallel (each parquet is independent) | ⚠️ Tokenization is parallel, but final sharding is sequential | ✅ Trivially parallel |
| **Training simplicity** | ⚠️ DataLoader must handle variable-size shards | ✅ Trivial — every shard looks identical | ❌ Must parse parquet at training time (slow) |
| **Memory-map (mmap) support** | ✅ .npy files can be memory-mapped | ✅ .npy files can be memory-mapped | ❌ Parquet cannot be memory-mapped efficiently |
| **Checkpoint/resume precision** | ⚠️ Works, but variable shard sizes make progress tracking harder | ✅ Uniform shards → simple (shard_idx, seq_offset) | ❌ Complex — must track row offsets in parquet |
| **Cross-source mixing** | ❌ Each file is still one source — mixing happens at training time | ✅ Sources are pre-mixed into unified shards | ❌ No mixing |
| **Implementation complexity** | ⭐ Simple | ⭐⭐ Medium | ⭐ Simple |
| **Disk space** | Normal | Normal | ~2× (keeps original text + tokens) |

---

## Why Uniform Shard Size Matters

This is the **key deciding factor**. Consider what happens with non-uniform shards:

```
Strategy A output (1:1 mapping):
  tokens-00000.npy  →  1.2 GB  (English web text, lots of tokens)
  tokens-00001.npy  →  0.3 GB  (Hindi text, fewer tokens for same byte size)
  tokens-00002.npy  →  4.1 GB  (code from The Stack, very dense)
  tokens-00003.npy  →  0.01 GB (NCERT chapter, tiny)

Problems:
  → GPU finishes shard-00003 in 2 seconds, shard-00002 takes 10 minutes
  → Progress tracking is inaccurate ("50% of shards done" ≠ 50% of tokens)
  → Checkpoint resume lands at different amounts of data depending on shard
  → S3 staging can't predict how long each shard takes to consume
```

vs.

```
Strategy B output (re-sharded):
  shard-00000.npy  →  2.0 GB  (500M tokens — mixed English + Hindi + code + NCERT)
  shard-00001.npy  →  2.0 GB  (500M tokens — mixed English + Hindi + code + NCERT)
  shard-00002.npy  →  2.0 GB  (500M tokens — mixed English + Hindi + code + NCERT)
  shard-00003.npy  →  2.0 GB  (500M tokens — mixed English + Hindi + code + NCERT)

Benefits:
  → Every shard takes the same time to train on
  → "50% of shards done" = exactly 50% of tokens done
  → Checkpoint resume is precise to the token
  → S3 staging can predict consumption rate perfectly
```

---

## The Best Approach: Parallel Tokenize → Sequential Re-shard

You get the best of both worlds by splitting it into two sub-steps:

```
PHASE 1: TOKENIZE (embarrassingly parallel — one job per parquet)
═══════════════════════════════════════════════════════════════════

  Worker 1: parquet-00000.parquet → tokenize → tokens-tmp-00000.npy
  Worker 2: parquet-00001.parquet → tokenize → tokens-tmp-00001.npy
  Worker 3: parquet-00002.parquet → tokenize → tokens-tmp-00002.npy
  ...
  Worker N: parquet-NNNNN.parquet → tokenize → tokens-tmp-NNNNN.npy

  → Runs on 96 CPU cores in parallel
  → Each worker is independent (no coordination needed)
  → Intermediate .npy files (temporary)


PHASE 2: SEQUENTIAL RE-SHARD (sequential stream, but fast — just copying integers)
═══════════════════════════════════════════════════════════════════

  Read tokens-tmp-00000.npy, tokens-tmp-00001.npy, ... in mixed order
       ↓
  Buffer 500M tokens → write shard-00000.npy
  Buffer 500M tokens → write shard-00001.npy
  Buffer 500M tokens → write shard-00002.npy
  ...

  → Sequential but extremely fast (just copying uint32 arrays, no parsing)
  → Interleaves sources at the desired mixing ratios
  → Produces uniform-size output shards
  → Deletes temporary files after completion
```

### Why this works well

| Sub-step | Speed bottleneck | Solution |
|----------|-----------|----------|
| **Phase 1 (tokenize)** | CPU-bound (BPE encoding is slow) | Parallelize across 96 cores — each parquet independently |
| **Phase 2 (re-shard)** | I/O-bound (read/write uint32 arrays) | Sequential but fast (~10 GB/s on NVMe) — finishes in minutes |

The expensive part (tokenization) is fully parallel. The cheap part (re-sharding) is sequential but blazingly fast because it's just copying `uint32` arrays — no text parsing, no encoding.

---

## Decision Framework

```
  Is your dataset multi-source and multi-lingual?
      │
      ├── NO (single source, single language)
      │     → Strategy A is fine (1:1 mapping)
      │
      └── YES (you have Dolma + Sangraha + NCERT + IndicNLP)
            │
            ├── Do you need uniform shard sizes?
            │     │
            │     ├── YES (for deterministic resume + uniform GPU load)
            │     │     → Strategy B: Parallel tokenize → Re-shard ✅
            │     │
            │     └── NO
            │           → Strategy A (simpler, works for experiments)
            │
            └── Your situation → Strategy B ✅
```

---

## The Correct Processing Order

Tokenize FIRST → Re-shard SECOND. This order is fixed and cannot be reversed.

```
Step 1  INGEST         Read raw files (Dolma JSONL, Sangraha Parquet, NCERT CSV, IndicNLP text)
  ↓                    Output: (text, language, source) stream per file
  
Step 2  FILTER         Quality filter, dedup, min-length check
  ↓                    Output: cleaned (text, language, source) stream

Step 3  TOKENIZE       ← PHASE 1 (parallel, 96 CPU cores)
  ↓                    text → token IDs + EOS token
  ↓                    Output: tmp-*.npy files (uneven sizes — that's OK)

Step 4  MIX + RESHARD  ← PHASE 2 (sequential, I/O-bound, fast)
  ↓                    Read tmp files in interleaved order (55% Dolma, 20% Sangraha-V, ...)
  ↓                    Concatenate into a buffer, write every 500M tokens as one shard
  ↓                    Output: shard-00000.npy, shard-00001.npy, ... (uniform 2 GB each)

Step 5  UPLOAD         Upload final shards to S3
  ↓                    Delete tmp-*.npy files

Step 6  TRAIN          S3Stager → NVMe → mmap → DataLoader → GPU
```

### Why tokenize before sharding (not the reverse)?

```
WRONG ORDER: Shard raw text → Tokenize
────────────────────────────────────────
  raw-shard-A (2 GB of Hindi text)  → tokenize → 200M tokens  ← SMALL
  raw-shard-B (2 GB of English web) → tokenize → 500M tokens  ← MEDIUM
  raw-shard-C (2 GB of Python code) → tokenize → 800M tokens  ← LARGE
  
  Result: Non-uniform token counts! Training is unbalanced.


CORRECT ORDER: Tokenize → Re-shard
────────────────────────────────────────
  All text → tokenize → continuous integer stream → split at every 500M tokens
  
  shard-00000.npy = exactly 500M tokens  ✅
  shard-00001.npy = exactly 500M tokens  ✅
  shard-00002.npy = exactly 500M tokens  ✅
  
  Result: Perfectly uniform. Every shard takes the same training time.
```

---

## Time Estimates

| Phase | Speed bottleneck | Time for 4 TB raw data | Parallelizable? |
|---|---|---|---|
| **Phase 1: Tokenize** | CPU-bound (BPE encoding) | ~6–10 hours on 96-core machine | ✅ Yes — embarrassingly parallel |
| **Phase 2: Re-shard** | I/O-bound (read/write uint32 arrays) | ~30–60 minutes on NVMe | ❌ Sequential, but very fast |
| **Total** | | **~7–11 hours** | |
| **Cost** (c7i.24xlarge Spot, 96 vCPUs) | | ~$1.22/hr × 12 hrs = **~$15 compute** | One-time cost |

---

## Summary

**For our 4 TB multi-source, multi-lingual dataset (Dolma + Sangraha + NCERT + IndicNLP):**

1. **Tokenize each parquet file independently** (parallel, one per CPU core)
2. **Re-shard the tokenized outputs** into uniform 500M-token `.npy` files, interleaving sources at the desired mixing ratios
3. The output shards are what goes to S3 and feeds the training pipeline

This gives you **parallel tokenization speed** + **uniform shards** + **pre-mixed sources** + **exact checkpoint/resume** — all four properties you need for robust large-scale training.
