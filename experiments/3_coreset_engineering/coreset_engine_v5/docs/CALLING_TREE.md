# Coreset Builder — Calling Tree

> How `shard.sh` invokes `coreset_builder.py` and the full execution flow
> of the `StreamingCoresetBuilder` (default, production path).

## Entry Point

```text
shard.sh
└── for SHARD_ID in 0..N-1 (parallel background processes)
    └── python coreset_builder.py --num-shards N --shard-id $SHARD_ID ...
```

## main() → StreamingCoresetBuilder

```text
main()                                                    # L1743
├── argparse (parse CLI args)
├── validate --config / --curriculum exist
│
├── StreamingCoresetBuilder.__init__()                     # L358
│   ├── CoresetBuilder.__init__()  (super)                 # L58
│   │   ├── PipelineConfig.load_from_file()
│   │   ├── CurriculumLoader()
│   │   │   ├── .load()
│   │   │   ├── .validate_curriculum_frozen()
│   │   │   └── .validate_deterministic_guarantees()
│   │   └── config.compute_hash()
│   ├── BatchProcessor()                                   # L465
│   ├── ErrorRecoveryManager()                             # L468
│   └── UsedChunksStore(sqlite per-shard)                  # L473
│
├── signal.signal()  (SIGINT / SIGTERM handlers)
│
├── builder.build_coresets()                               # L794
│   └── (see Build Coresets below)
│
├── streaming summary logging                              # L1993
│
└── builder.generate_reports(results)                      # L325
    ├── detect shard_id / num_shards from results
    ├── if multi-shard → "ablation_..._shard{id:03d}.md"
    └── AblationReporter.generate_report()
```

## build_coresets()

```text
build_coresets()                                           # L794
└── for stage_name in ["1B", "3B", "8B", "70B"]:
    └── _build_stage_coreset(stage_name, stage_config)     # L950
```

## _build_stage_coreset()

```text
_build_stage_coreset(stage_name, stage_config)             # L950
│
│ ── TARGET TOKEN SCALING ──
├── curriculum.get_stage_config()
├── stage_target_tokens /= num_shards                      # L978
├── shard_total_tokens_est /= num_shards                   # L982
│
│ ── CHECKPOINT RESUME ──
├── batch_processor.find_last_checkpoint()                 # L988
├── batch_processor.load_checkpoint()                      # L1094
│   └── validate num_shards / shard_id match               # L1100
│       └── engine.load_checkpoint_state()
│
├── BatchedSelectionEngine()                               # L993
├── build protected_slices (B4, B5, code, agentic, indic)  # L996
├── pre-compute language / band / domain gates             # L1066
│
│ ── BATCH PROCESSING LOOP ──
├── for batch_idx, batch in _iter_batches():               # L1165
│   │
│   ├── used_store.filter_unused(batch_ids)                # L1196
│   │   ├── (optional) _used_cache_get()                   # L599
│   │   └── (optional) _used_cache_put()                   # L610
│   │
│   ├── ROW PARSING → ChunkMetadata                        # L1200
│   │   ├── _extract_band_score(row, meta_dict)            # L674
│   │   ├── _extract_band_from_band_p(row, meta_dict)      # L747
│   │   └── _infer_band_from_score(score)                  # L624
│   │
│   ├── engine._process_batch(                             # L1382
│   │       stream, stage_name, protected_slices,
│   │       stage_target_tokens, ...)
│   │
│   ├── WRITE SELECTED INDICES (part files)                # L1394
│   │   ├── "selected_indices_part_shard{id}_batch{idx}.parquet"
│   │   └── used_store.add_many(selected_ids)              # L1483
│   │
│   └── _write_checkpoint() (every N batches)              # L1547
│       └── batch_processor.save_checkpoint()
│           └── state: shard_id, num_shards, engine_state
│
│ ── STAGE FINALIZATION ──
├── final _write_checkpoint() (if needed)                  # L1593
├── build BandDistribution / DomainDistribution / ...
├── build CoresetManifest (includes shard_id, num_shards)  # L1651
├── save manifest_shard{id:03d}.json                       # L1697
├── if num_shards == 1 → also save manifest.json           # L1703
└── return stats dict
```

## _iter_batches() — Input Sharding

```text
_iter_batches()                                            # L815
├── _should_enable_batch_prefetch()                        # L481
│   └── shard_cpu_ratio = num_shards / cpu_count
│
├── _base_iter_batches()                                   # L818
│   │
│   ├── [JSONL path]:
│   │   ├── batch_processor.list_input_files(path, "jsonl")
│   │   ├── if multiple files → FILE-LEVEL sharding:
│   │   │   └── batch_processor.shard_files(               # L833
│   │   │           files, shard_id, num_shards)
│   │   │       └── xxhash(path) % num_shards == shard_id
│   │   ├── if single file → ROW-LEVEL sharding:
│   │   │   └── batch_processor.batch_iterator(            # L840
│   │   │           shard_id, num_shards, shard_key="chunk_id")
│   │   │       └── hash(chunk_id) % num_shards == shard_id
│   │   └── yield batch_idx, batch
│   │
│   └── [Parquet path]:
│       ├── batch_processor.list_input_files(path, "parquet")
│       ├── batch_processor.shard_files(                   # L863
│       │       files, shard_id, num_shards)
│       ├── batch_processor.parquet_batch_iterator()
│       └── yield batch_idx, batch
│
└── (optional) _iter_with_prefetch()                       # L507
    └── background thread: prefetch next batch into queue
```

## How num_shards Controls Execution

| Aspect | Where | What it does |
|--------|-------|-------------|
| **Input splitting** | `shard_files()` | `xxhash(filepath) % N == shard_id` assigns files to workers |
| **Row-level fallback** | `batch_iterator()` | `hash(chunk_id) % N` when only 1 input file exists |
| **Token budget** | `_build_stage_coreset` L978 | `target /= num_shards` — each shard targets 1/N of total |
| **Prefetch tuning** | `_should_enable_batch_prefetch` | Disables prefetch if shards/CPUs ratio is too high |
| **Output naming** | Part files, manifests, reports | All stamped with `shard{id:03d}` to avoid collisions |
| **Checkpoint guard** | `load_checkpoint` L1100 | Rejects resume if num_shards changed between runs |
