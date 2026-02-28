# Pipeline — EMR Serverless Band Assignment Jobs

Three independent PySpark jobs that assign a curriculum band (B0–B5) to every document in the training corpus. All three jobs share the same probabilistic banding algorithm and produce an identical output schema. They differ in what data they cover and how source-specific constraints are applied.

---

## Jobs

| Job | Script | Input Data | Band Range |
|-----|--------|-----------|------------|
| Main | `jobs/main_job.py` | Large-scale web/book/code (RedPajama, FineWeb, Dolma, Sangraha, arXiv, etc.) | B0–B5 (full) |
| Curated Datasets | `jobs/curated_datasets_job.py` | 17 HuggingFace SFT/math/code/preference datasets | Source-clamped per dataset |
| Student Data | `jobs/student_data_job.py` | ERAv4 Q&A drills + Samvaad conversation | B0–B2 only |

---

## Running on EMR Serverless

### Prerequisites

- EMR Serverless application provisioned (Spark runtime)
- S3 bucket with T1 output Parquet files
- IAM execution role with S3 read/write permissions

### Job Arguments

All three jobs accept the same arguments:

| Argument | Description | Example |
|----------|-------------|---------|
| `--INPUT_BASE` | S3 path to T1 normalized Parquet | `s3://your-bucket/t1-output/` |
| `--OUTPUT_BASE` | S3 path for band assignment output | `s3://your-bucket/t2-output/` |
| `--JOB_NAME` | Job run identifier (used in output paths) | `t2_main_2026_02_21` |

### Submitting a Job

```bash
aws emr-serverless start-job-run \
  --application-id <APPLICATION_ID> \
  --execution-role-arn <ROLE_ARN> \
  --job-driver '{
    "sparkSubmit": {
      "entryPoint": "s3://your-bucket/scripts/main_job.py",
      "entryPointArguments": [
        "--INPUT_BASE", "s3://your-bucket/t1-output/",
        "--OUTPUT_BASE", "s3://your-bucket/t2-output/",
        "--JOB_NAME", "t2_main_2026_02_21"
      ],
      "sparkSubmitParameters": "--conf spark.executor.cores=4 --conf spark.executor.memory=16g --conf spark.driver.memory=8g --conf spark.sql.shuffle.partitions=4000"
    }
  }' \
  --configuration-overrides '{
    "monitoringConfiguration": {
      "cloudWatchLoggingConfiguration": { "enabled": true }
    }
  }'
```

### Recommended Spark Config (4TB corpus)

```
spark.executor.cores=4
spark.executor.memory=16g
spark.driver.memory=8g
spark.sql.shuffle.partitions=4000
spark.sql.adaptive.enabled=true
spark.sql.adaptive.coalescePartitions.enabled=true
spark.sql.parquet.compression.codec=zstd
```

---

## Expected Output

```
s3://your-bucket/t2-output/
├── bands/
│   ├── assigned_band=B0/
│   ├── assigned_band=B1/
│   ├── assigned_band=B2/
│   ├── assigned_band=B3/
│   ├── assigned_band=B4/
│   └── assigned_band=B5/
└── rejections/
    ├── rejection_level=1/
    └── rejection_level=2/
```

Parquet files, zstd compressed, partitioned by `assigned_band`.

### Output Schema

| Column Group | Columns |
|-------------|---------|
| Identity | `uuid`, `id`, `file_path`, `source`, `domain`, `hash`, `language`, `metadata` |
| Band | `assigned_band`, `band`, `difficulty_score`, `band_p_B0`–`band_p_B5` |
| Modality flags | `has_code`, `has_cot`, `has_reasoning`, `has_agentic` |
| Scores | `code_score`, `math_score`, `reasoning_score`, `agentic_score`, `cot_score` |
| Size | `byte_length`, `word_count`, `unique_token_ratio`, `compression_ratio`, `token_count_estimate` |
| Rejection (rejected only) | `is_rejected`, `rejection_reason`, `rejection_level` |

---

## Expected Rejection Rates

- Stage 1 (physical corruption): 2–3%
- Stage 2 (noise/spam): 2–3%
- **Total rejected: 4–6%. Target pass-through: 95–98%.**

If rejection rate exceeds 10%, check:
1. Source encoding — non-UTF-8 bytes trigger Stage 1
2. `boilerplate_ratio` threshold — template-heavy sources may inflate this
3. `whitespace_ratio` threshold — structured/formatted sources (books, code) legitimately have higher ratios

---

## Methodology

See `../docs/band_assignment_methodology.md` for the full specification:
signal extraction, composite score formulas, difficulty calculation,
probabilistic banding algorithm, source clamping, and a worked example.

See `../docs/CHANGELOG.md` for the full version history (V2.1 → v7.1).
