# Curriculum Data PySpark Pipeline (AWS Glue Production-Ready)

This repository contains a high-scale PySpark implementation for processing and deduplicating LLM curriculum datasets. It is designed to be deployed as an **AWS Glue Job** (Spark 3.x/4.x).

---

## 🧠 Logic & Architecture (For Experts)

The pipeline employs a **Source-Centric Distributed Deduplication** strategy. Unlike traditional global dedupes that shuffle the entire corpus, this job isolates deduplication to each source but consolidates all its bands.

### 1. Data Flow Step-by-Step

1. **Dynamic Discovery**: Uses `boto3` to crawl the S3 hierarchy and find all `source=<NAME>` partitions.
2. **Unit Consolidation**: For each source, the job unions all target bands (B0-B5). This ensures cross-band consistency within a source.
3. **Deterministic Deduplication**: 
    - Uses a Spark **Window Function** (`Window.partitionBy("hash").orderBy("band")`).
    - This ensures that if a document exists in multiple bands, the one from the "lowest" band (e.g., B0 over B1) is deterministically kept.

    ```python
    window_spec = Window.partitionBy("hash").orderBy(F.col("band"))
    unique_df = transformed_df.withColumn("row_num", F.row_number().over(window_spec)) \
        .filter(F.col("row_num") == 1)
    ```

4. **Shuffle Management**: `spark.sql.shuffle.partitions` is dynamically tuned via `parallelism` config to avoid the "small files" or "skewed partitions" problems.
5. **Partitioned Write**: Uses `partitionBy("source")` in the final S3 write, allowing for efficient downstream usage in Athena or Glue Data Catalog.

### 2. Resilience & Checkpointing

- **Checkpoint Manager**: Tracks progress at the **Source level**.
- **Execution Flow**:
    1. **Check**: Before processing, the job checks for a `.done` marker in S3.
    2. **Process**: Consolidates and deduplicates all bands for the source.
    3. **Commit**: Only after successful S3 write is the `.done` marker created.
- **Auto-Resume**: If a job fails or is timed out by Glue, rerunning it will automatically check these markers and skip completed sources, saving cost and time.

---

## 📁 Project Structure

```text
curriculum_pyspark_glue/
├── glue_job.py           # Main Entry Point: Orchestrates the Glue Job, handles GlueContext & Job Init.
├── config/
│   └── config.yaml       # Configuration: Centralized S3 paths, schema maps, and DPU/Worker tuning.
├── src/
│   ├── processor.py      # Core Logic: Spark implementation of schema transforms & Window-based dedup.
│   └── utils.py          # Utilities: Boto3-based S3 discovery, logging setup, and CheckpointManager.
└── README.md             # Documentation: Professional guide for logic, deployment, and monitoring.
```

---

## 🛠️ AWS Glue Configuration Guide

To satisfy professional standards, configure your Glue job with these parameters:

### Worker Selection

- **Worker Type**: `G.1X` is usually sufficient for metadata-heavy tasks. Use `G.2X` if your source doc sizes are extremely large (>1GB per doc).
- **DPU Scaling**: Set `num_workers` based on the number of sources. A good rule of thumb is `1 worker per 5GB of source data`.

### Job Parameters

| Variable | Value | Description |
| :--- | :--- | :--- |
| `--config_path` | `s3://bucket/config.yaml` | Location of config (S3 supported) |
| `--additional-python-modules` | `pyyaml` | Required for config parsing |
| `--enable-metrics` | `true` | Enables CloudWatch Spark UI |
| `--enable-continuous-cloudwatch-log` | `true` | Real-time logging |

---

## 🔄 How to Run & Resume

1. **First Run**: The job will discover all sources and process them one by one. Logs will show `🚀 Processing Source Group: <NAME>`.
2. **Failure Scenario**: If the job fails (e.g., S3 timeout), check the CloudWatch logs to find the last successful source.
3. **Resuming**: Simply **Restart** the Glue job. The `CheckpointManager` will log `Skipping already processed source: <NAME>` for finished units and pick up exactly where it left off.

---

## 📈 Monitoring

Logs are categorized for easy filtering in CloudWatch:

- `INFO [glue_logger]`: High-level orchestration.
- `ERROR [glue_logger]`: Critical failures in S3 I/O or Spark logic.
