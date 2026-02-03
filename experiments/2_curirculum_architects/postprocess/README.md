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
metadata:

-   `curriculum_band` (B0--B5)
-   additional metrics as needed

Output:

    s3://.../enriched/*.parquet

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
deterministic ordering.

Output:

    global_index_shuffled.parquet

Changing the seed produces a new shuffle; keeping the seed guarantees
identical order across runs and machines.

------------------------------------------------------------------------

### 4. Stage Manifest Generation

Using `global_index_shuffled.parquet`, stage-specific manifests are
created according to curriculum ratios (from YAML or config).

Each stage manifest is a filtered view of the shuffled index:

    stage1_manifest.parquet
    stage2_manifest.parquet
    ...

Each manifest contains rows of:

    id | band | file | row

The order in these manifests defines training order.

No data is duplicated.

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

As a result:

-   Batch N always contains the same samples
-   Rerunning stage generation produces identical manifests
-   Training can resume from arbitrary batch IDs safely

------------------------------------------------------------------------

## Artifacts

Typical outputs:

    enriched/*.parquet
    global_index.parquet
    global_index_shuffled.parquet
    stage1_manifest.parquet
    stage2_manifest.parquet
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
