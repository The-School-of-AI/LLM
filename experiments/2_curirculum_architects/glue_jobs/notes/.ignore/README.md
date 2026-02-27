
# Glue Metric Calculator Job (Optimized)

This Glue job processes raw json.gz datasets to produce:
1.  **Team 1 Output**: annotated raw data (including rejection status) partitioned by `source`.
2.  **Team 2 Output**: metrics-only dataset partitioned by `domain` and `source`.

## Optimization Highlights (Global Parallel Architecture)
*   **Global Union**: Processes all 16 datasets in a single parallel job to maximize cluster utilization.
*   **Zero-Shuffle UDF**: Uses inline struct projection instead of Joins to eliminate expensive text shuffling.
*   **Strict Rejection**: fast filters (Length, HTML, Noise) run first; expensive Python metrics (`tiktoken`) are skpped for rejected rows.
*   **Compression**: **ZSTD** is used for all outputs to balance archival compression ratio and speed.
*   **AQE**: Adaptive Query Execution is enabled with 128MB target partition sizes.

## Prerequisites

The job requires the following **native Python libraries** (not standard in Glue) to be passed via `--additional-python-modules`:
-   `tiktoken` (for token counting)
-   `textstat` (for Flesch reading ease)

## Usage

### CLI Command (Example)
```bash
aws glue start-job-run \
    --job-name curriculum-metrics-job \
    --arguments '{"--additional-python-modules":"tiktoken,textstat"}'
```
*Note: Output paths are hardcoded in the script for safety, as per latest configuration.*

## Output Schema
### Team 1
`id`, `hash`, `dataset`, `domain`, `source`, `text`, `language`, `metadata`, `added`, `created`, `version`, `is_rejected`, `rejection_reason`.

### Team 2
Includes `id`, `file_path`, `is_rejected`, `rejection_reason` + all ~60 metric columns.
