# Curriculum Dataset Processing Pipeline

This directory contains the data processing pipeline used to prepare a large Parquet dataset in S3 for curriculum-based training of LLMs.

The pipeline enriches raw documents with curriculum metadata, builds deterministic global indices, and produces stage-specific manifests used directly by training dataloaders.

The design prioritizes:

-   Deterministic sampling and batching
-   Minimal data duplication
-   Efficient S3-based access
-   Separation of data and indices

------------------------------------------------------------------------

## High-Level Flow

    Raw Parquet (S3)
       ↓
    Enriched Parquet (adds curriculum tags)
       ↓
    Global Index (id → file + row + band)
       ↓
    Deterministic Shuffle
       ↓
    Stage Manifests (band ratios per stage)
       ↓
    Training DataLoader (manifest-driven)

Only index files are shuffled or filtered. The underlying Parquet data
is never copied.

------------------------------------------------------------------------

## Pipeline Stages

### 1. Enrichment

Each raw Parquet file is processed and augmented with curriculum
metadata (stored under `curriculum_tags`):

- Band assignment (B0–B5)
- Difficulty metrics
- Modality signals
- Additional curriculum features

Output:

    s3://.../enriched/data/*.parquet

These files contain the full original records plus curriculum fields.

------------------------------------------------------------------------

### 2. Global Index Construction

A lightweight index is built across all enriched Parquet files.

Each row in the index represents a single document:

  |Field    | Description                        |
  |------------------------------------|------------------------------------|
  | `id`     | Document ID                        |
  |`band`   | Curriculum band (B0--B5)           |
  | `file`  | S3 path to enriched Parquet file   |
  |`row`    | Row index inside that Parquet file |


Output:

    global_index.parquet

This file is small compared to the dataset and is used for all
downstream sampling.

------------------------------------------------------------------------

### 3. Deterministic Global Shuffle

To ensure reproducible ordering, a stable hash is computed for each `id`
using a fixed seed:

    hash = xxhash64(id + seed)

The index is sorted by this hash, producing a globally shuffled but
deterministic ordering. The shuffled index is written under a canonical hierarchy:

    *_index/*   
      curriculum_<version>/    
        seed_<seed>/
          global_index_shuffled.parquet   

Example:   
`s3://<bucket>/enriched/_index/curriculum_0.3/seed_42/global_index_shuffled.parquet`    

Notes:
- The curriculum `version` comes from `curriculum.yaml`
- The `seed` is provided at runtime
- Both values are also embedded into Parquet metadata for provenance  

Changing the seed produces a new shuffle; keeping the seed guarantees
identical order across runs and machines.

------------------------------------------------------------------------

### 4. Stage Manifest Generation

Using `global_index_shuffled.parquet`, stage-specific manifests are 
created according to band ratios defined in `curriculum.yaml`.

Each stage manifest is a filtered view of the shuffled index:

    stage_1B_manifest.parquet
    stage_3B_manifest.parquet
    stage_8B_manifest.parquet
    stage_70B_manifest.parquet
    ...

Each manifest contains rows of:

    id | band | file | row

The order in these manifests defines training order.
No data is duplicated.
All manifests live alongside the shuffled index:

    *_index/*   
        curriculum_<version>/    
            seed_<seed>/
              global_index_shuffled.parquet 
              stage_1B_manifest.parquet
              stage_3B_manifest.parquet
              stage_8B_manifest.parquet
              stage_70B_manifest.parquet


------------------------------------------------------------------------

### 5. Training

Training reads only:

-   A stage manifest
-   The enriched Parquet files

The PyTorch Dataset:

1.  Reads entries from the manifest
2.  Uses `(file, row)` to fetch the exact record from S3 Parquet
3.  Returns samples in manifest order

Important:

-   `DataLoader(shuffle=False)`
-   Determinism is fully controlled by the manifest

This guarantees:

-   Reproducible batches
-   Stable resume from any batch index
-   Identical data ordering across reruns

------------------------------------------------------------------------

## Determinism Guarantees

Deterministic behavior is achieved by:

-   Hash-based global ordering with fixed seed
-   Manifest-driven sampling
-   Disabling DataLoader shuffle
-   Explicit curriculum versioning in output paths

As a result:

-   Batch N always contains the same samples
-   Rerunning stage generation produces identical manifests
-   Training can resume from arbitrary batch IDs safely
-   Multiple curricula and seeds can coexist side-by-side


------------------------------------------------------------------------

## Running the Main Postprocessing Script

The main entrypoint consumes:

- A global index Parquet
- A curriculum YAML
- An output prefix (S3)
- A shuffle seed

Example:

```bash
python main.py \
  --index s3://<bucket>/enriched/_index/global_index.parquet \
  --curriculum curriculum.yaml \
  --out-prefix s3://<bucket>/enriched/_index \
  --seed 42
```

This will produce:

    s3://<bucket>/enriched/_index/
      curriculum_0.3/
        seed_42/
          global_index_shuffled.parquet
          stage_1B_manifest.parquet
          stage_3B_manifest.parquet
          stage_8B_manifest.parquet
          stage_70B_manifest.parquet

You may safely rerun this command with the same inputs to reproduce identical outputs.

------------------------------------------------------------------------

## Artifacts

Typical outputs:

    enriched/data/*.parquet
    enriched/_index/
      curriculum_<version>/
        seed_<seed>/
          global_index.parquet
          global_index_shuffled.parquet
          stage*_manifest.parquet
    ...

Only the `enriched` Parquet files contain full data. All other files are
lightweight indices.

------------------------------------------------------------------------

## Summary

This pipeline separates:

-   **Data** (large, immutable Parquet files)
-   **Ordering + curriculum** (small, mutable manifests)

This enables fast iteration on curriculum strategies while keeping
storage and compute costs low, and provides strict reproducibility for
large-scale training.
