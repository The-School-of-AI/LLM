# Upstream Data Contract (Input Chunk Pool)

Defines the **expected upstream chunk schema** consumed by the coreset pipeline (load → bucket by band/domain → score → stratified selection). Source of truth is the current loader/engine behavior in `src/io/loaders.py`, `src/selection/engine.py`, and `src/selection/engine_batched.py`.

## Supported input formats

- **JSONL**: one JSON object per line; may include nested `metadata` object.
- **Parquet**: columnar dataset; loader is stricter (rows with missing required columns are typically skipped).

## Field contract (one row = one chunk)

> Note: There are two common upstream shapes:
> 1) **Metadata-only chunk pool** (like `data/outputv2/b0_shard_0.jsonl`): IDs + band/domain/language + counts + band probabilities/scores.
> 2) **Text/tokens present**: includes `chunk_text` and/or `token_ids` to enable real dedup and richer diversity scoring.
>
> Band inference is controlled by the streaming entrypoint flags `--band-inference` and `--band-score-source` (sometimes described informally as “band inference” / “band source score”).

### Required fields

| Field | Type | JSONL aliases / sources | Why required | Behavior if missing |
|---|---:|---|---|---|
| `chunk_id` | string | `chunk_id` \| `uid` \| `guid` \| `id` | Determinism + unique tie-break + sharding | Missing/empty can break type assumptions or create collisions |
| `token_count_estimate` (or `token_count`) | int | top-level or `metadata.*` | Budgets/targets, rolling-window, selection accounting | Loader defaults to `0`; selection budgets/accounting degrade and stages may underfill |
| `domain` | string | top-level or `metadata.domain` | Bucketing + curriculum domain policy | JSONL defaults to `unknown`; disallowed domains typically get `target_tokens=0` for that stage |
| `band` | string enum `B0`–`B6` | top-level or `metadata.band` | Bucketing + curriculum band ratios | `ChunkLoader`: defaults to `B0`; invalid bands may cause row skip. Streaming builder: invalid/missing band can be inferred (if configured) or defaulted to `B0` |
| `language` | string | top-level or `metadata.language` | Language policy enforcement | JSONL defaults to `en`; policy treats it as English |

### Optional fields (and degradations)

| Field | Type | Used for | Behavior if absent |
|---|---:|---|---|
| `token_ids` | list[int] | Diversity scoring | Scoring uses proxy token list `list(range(min(100, token_count)))` |
| `band_score` | float | Selection ranking | Ranking falls back to diversity composite score; deterministic tie-break by `chunk_id` |
| `difficulty_score` | float | Band inference / band score derivation (streaming builder) | Only used when band inference is enabled and `--band-score-source` selects it (or `auto` falls back to it); otherwise ignored |
| `band_p_B0`…`band_p_B6` | float | Band inference / band score derivation (streaming builder) | Only used when `--band-score-source` selects a `band_p_*` strategy (`band_p_Bx`, `band_p_max`, `band_p_argmax`, or `auto` fallback); otherwise ignored |
| `chunk_text` | string | Dedup only | Dedup signatures not computed; in streaming batch dedup, dedup effectively reduces to duplicate `chunk_id` detection within a batch |
| `dataset_id` (or `source`) | string | Traceability/output | JSONL defaults to `"ds"` (aliases: `dataset_id` or `source` or `metadata.source`) |
| `byte_length` | int | Traceability/output | Defaults to `0` |
| `source_doc_id` | string | Traceability/output | Should be provided; otherwise empty/missing propagates |
| `source_url` | string | Traceability/output | Optional |
| `quality_flags` | list[str] | Output metadata | Defaults to `[]` |
| `sensitive_markers` | list[str] | Output metadata | Defaults to `[]` |
| `start_offset` | int | Output metadata | Defaults to `0` |

## Chunk file Schema:

This file is a **metadata-only** chunk pool: it does **not** include `chunk_text` or `token_ids`. All keys below appear on every row.

### Columns present (verbatim)

`agentic_score`, `band`, `band_p_B0`, `band_p_B1`, `band_p_B2`, `band_p_B3`, `band_p_B4`, `band_p_B5`, `band_score`, `byte_length`, `chunk_id`, `code_score`, `compression_ratio`, `cot_score`, `difficulty_score`, `domain`, `fertility_estimate`, `has_agentic`, `has_code`, `has_cot`, `has_reasoning`, `language`, `math_score`, `reasoning_score`, `source`, `source_doc_id`, `source_url`, `token_count_estimate`, `unique_token_ratio`, `word_count`.

### What the pipeline consumes from these columns

`ChunkLoader.load_chunks_from_jsonl()` maps:

- `chunk_id` → `ChunkMetadata.chunk_id`
- `source` → `ChunkMetadata.dataset_id` (loader prefers `dataset_id` if present, else falls back to `source`)
- `token_count_estimate` → `ChunkMetadata.token_count`
- `byte_length` → `ChunkMetadata.byte_length`
- `domain` → `ChunkMetadata.domain`
- `language` → `ChunkMetadata.language`
- `band` → `ChunkMetadata.band`
- `source_doc_id` → `ChunkMetadata.source_doc_id`
- `source_url` → `ChunkMetadata.source_url`
- `band_score` → attached dynamically as `metadata.band_score` (used for ranking when present)

When running the streaming entrypoint `coreset_builder.py` with `--band-inference` enabled (anything other than `none`), the builder may also read `difficulty_score` and/or `band_p_B0..band_p_B6` (per `--band-score-source`) to:

- infer/override the discrete `band`, and/or
- populate `metadata.band_score` for downstream ranking.

Other fields in this file (e.g., `has_code`, `*_score`, `word_count`, `unique_token_ratio`, `compression_ratio`, `fertility_estimate`) are still **not currently used by selection** as implemented in `src/selection/engine.py`.

## JSONL examples

Flat record (recommended):
```json
{"chunk_id":"ch_001","dataset_id":"books","token_count_estimate":2048,"byte_length":9876,"domain":"clean_web","language":"en","band":"B2","source_doc_id":"part-00000","source_url":"s3://...","token_ids":[1,2,3]}
```

Nested metadata (accepted):
```json
{"uid":"ch_001","token_count":2048,"metadata":{"source":"books","domain":"clean_web","language":"en","band":"B2","source_doc_id":"part-00000"}}
```

## Parquet: minimum viable columns

Required columns:
- `chunk_id`, `dataset_id`, `domain`, `language`, `band`, `byte_length`, `source_doc_id`, and one of `token_count`/`token_count_estimate`

Optional columns:
- `source_url`, `quality_flags`, `sensitive_markers`, `start_offset`, `token_ids`
