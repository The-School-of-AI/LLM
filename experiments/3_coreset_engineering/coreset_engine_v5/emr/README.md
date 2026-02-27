# Curriculum Data Processing Pipeline (EMR Serverless)

This pipeline consolidates, transforms, and deduplicates curriculum data from multiple S3 sources. It is optimized for EMR Serverless and handles millions of records efficiently by pruning data before shuffling.

## Key Features

- **Resume from Checkpoint**: Uses S3-based `.done` files to skip already processed sources, allowing for robust retries.
- **Optimized Deduplication**: Performs "Exact Deduplication" on a minimized schema (hash-only) to significantly reduce Network I/O and Disk spill.
- **Advanced Statistics**: Reports detailed metrics including input/unique counts for documents, words, and tokens, along with reduction percentages.
- **Metadata Capture**: Automatically tracks the origin of each record (`source_doc_id` and `source_url`).

## Data Schema Mapping

| Source Column | Target Column | Description |
| :--- | :--- | :--- |
| `id` | `chunk_id` | Mapped from source `id` for unique identification. |
| `id`, `uuid` | *Dropped* | Internal IDs are removed to keep the output clean. |
| `text`, `metadata` | *Dropped* | Large fields are removed from the final Parquet output. |
| `hash` | *Dropped* | Used for deduplication and then removed. |
| `assigned_band` | `band` | The final band the record belongs to. |
| (New) | `band_score` | Probability score specifically for the assigned band. |
| (New) | `source_doc_id` | Filename of the source Parquet file. |
| (New) | `source_url` | Full S3 folder path of the source data. |

## System Architecture & Flow

```text
[  S3 Raw Data  ] --> [ Source Discovery ] --> [ Band Consolidation ]
(B0-B5 Parquet)       (Boto3 Listing)          (Union all Bands)
                                                      |
                                                      v
[ Final Cleanup ] <-- [ Stats & Checkpoint ] <-- [ Feature Scoring ]
(Drop Hash/Band)      (Docs/Words/Tokens)       (band_score lookup)
        ^                                             ^
        |                                             |
[ Deduplicated  ] <--- [ Hash-Only Dedup ] <--- [ Data Thinning ] 
(Unique Records)      (Spark Shuffle)         (Drop Text/Metadata)
```

## Processing Logic (Step-by-Step)

1.  **Discovery**: Scans S3 prefixes to find all `source=xxx` folders.
2.  **Consolidation**: For each source, it finds all available band folders (`band=B0` to `band=B5`) and performs a Spark `union` to create one master dataset for that source.
3.  **Metadata Capture**: 
    - `source_doc_id`: Captures the specific `.parquet` filename for every record using `input_file_name()`.
    - `source_url`: Captures the full s3 folder url path (e.g., `s3://.../band=B2/`).
4.  **Schema Preparation**: Renames the `id` column to `chunk_id` as the primary identifier.
5.  **Data Thinning (Performance Block)**: To prevent Out-of-Memory errors during shuffle, the script immediately drops heavy `text` and `metadata` columns. It only keeps `hash`, `chunk_id`, `assigned_band`, `source_doc_id`, `source_url` and the `band_p_*` scores.
6.  **Global Deduplication**: Executes `dropDuplicates(["hash"])`. Since the data is now "thin", Spark performs the shuffle and deduplication significantly faster and with less disk spill.
7.  **Post-Dedup Feature Engineering**:
    - **Band Alignment**: Derives the final `band` column from `assigned_band`.
    - **Score Mapping**: Selects the `band_score` by looking up the specific probability column (e.g., if band is `B3`, it matches the value in `band_p_B3`).
8.  **Statistics Collection**: Aggregates word and token counts for both the input set and the unique set, calculating the reduction percentages.
9.  **S3 Persistence**: Saves the unique records to S3 as Parquet, optimized for downstream analytical tasks.
10. **Checkpointing**: Writes a `<source>.done` file to S3 marking completion. This enables the script to "Resume" if interrupted.

## Sample Record (Schema Illustration)

While the final output is in **Parquet** format, each record contains the following schema and data structure:

```json
{
  "chunk_id": "c9cc4f99-2006-479e-95b6-b84a5094ca93",
  "source_doc_id": "part-00019-e5d823ab-2176-4666-90e1-47d89b6983e8.c000.zstd.parquet",
  "source_url": "s3://t2-datacurriculum-353/processed_dataset/curriculum_data/source=ncert/bands/band=B2/",
  "source": "ncert",
  "band": "B2",
  "band_score": 0.7746882097457848,
  "domain": "education",
  "language": "en",
  "difficulty_score": 0.37808529263565893,
  "has_code": false,
  "has_cot": false,
  "has_reasoning": false,
  "has_agentic": false,
  "agentic_score": 0,
  "cot_score": 2,
  "reasoning_score": 0,
  "code_score": 0,
  "math_score": 0,
  "byte_length": 1103,
  "word_count": 172,
  "unique_token_ratio": 0.5523255813953488,
  "compression_ratio": 1.0,
  "token_count_estimate": 223,
  "fertility_estimate": 4.946188340807175,
  "band_p_B0": 0.0,
  "band_p_B1": 0.0,
  "band_p_B2": 0.7746882097457848,
  "band_p_B3": 0.1,
  "band_p_B4": 0.1253117902542152,
  "band_p_B5": 0.0
}
```

## Statistics Tracking

The script generates a per-source summary in the logs:
- **Input Docs/Words/Tokens**: Full volume loaded from S3.
- **Unique Docs/Words/Tokens**: Volume remaining after global deduplication.
- **Dropped (Duplicates)**: Absolute reduction achieved.
- **Token Reduction**: Percentage decrease in token count.

Aggregated distribution stats are also written to S3 in CSV format at:
`s3://<BUCKET>/<OUTPUT_PREFIX>/stats/<SOURCE>/`

## Usage

### Run on EMR Serverless

Use the following AWS CLI command to submit the job. This command includes the monitoring configuration to save your `INFO` logs to S3.

```bash
aws emr-serverless start-job-run \
  --application-id <APP_ID> \
  --execution-role-arn <ROLE_ARN> \
  --job-driver '{
    "sparkSubmit": {
      "entryPoint": "s3://t2-datacurriculum-353/scripts/T3_final_emr_serverless_stats.py",
      "entryPointArguments": [
        "--BUCKET", "t2-datacurriculum-353",
        "--BASE_PREFIX", "processed_dataset/curriculum_data",
        "--OUTPUT_PREFIX", "processed_dataset/curriculum_pyspark_output"
      ],
      "sparkSubmitParameters": "--conf spark.executor.cores=4 --conf spark.executor.memory=16g --conf spark.driver.cores=4 --conf spark.driver.memory=16g --conf spark.sql.shuffle.partitions=200"
    }
  }' \
  --configuration-overrides '{
    "monitoringConfiguration": {
      "s3MonitoringConfiguration": {
        "logUri": "s3://t2-datacurriculum-353/processed_dataset/curriculum_pyspark_output/logs/"
      }
    }
  }'
```

---

## Output Locations

All results are centralized under the `OUTPUT_PREFIX`:

1.  **Processed Data**: `s3://<BUCKET>/processed_dataset/curriculum_pyspark_output/source=<SOURCE>/` (Parquet format)
2.  **Distribution Stats**: `s3://<BUCKET>/processed_dataset/curriculum_pyspark_output/stats/<SOURCE>/` (CSV format)
3.  **Job Logs**: `s3://<BUCKET>/processed_dataset/curriculum_pyspark_output/logs/` (Spark stdout/stderr)

### Script Arguments
- `--BUCKET`: The S3 bucket name.
- `--BASE_PREFIX`: Input data prefix (where `source=...` folders are located).
- `--OUTPUT_PREFIX`: Destination for processed results and stats.
- `--SOURCE`: (Optional) Process a specific source only.
