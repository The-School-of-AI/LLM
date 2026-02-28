# CurriculumTagger S3 Enrichment Pipeline

This script processes Dolma (or other) Parquet datasets on S3, adds curriculum metadata to each record using `CurriculumTagger`, and writes enriched Parquet files back to S3. It supports **parallel processing, atomic writes, resume capability**, and **manifest tracking** for production workflows.

---

## Features

- Process **one input file → one output file** (deterministic)
- Direct **S3 streaming** (no local download required)
- **Batch processing** for memory efficiency
- **Ray parallelism** (configurable number of files processed concurrently)
- **Resume and retry** using a manifest file
- **Atomic writes** to avoid partial/corrupt outputs
- Tracks success/failure, number of rows, and errors in a local manifest

---

## Configuration

Edit the configuration section of the script:

```python
INPUT_S3_PREFIX = "s3://my-bucket/dolma_parquet/"  # S3 folder with input Parquet files
OUTPUT_S3_PREFIX = "s3://my-bucket/dolma_enriched/"  # S3 folder for enriched files
CURRICULUM_YAML = "/home/ubuntu/curriculum.yaml"  # Path to your curriculum config
BATCH_SIZE = 10000       # Number of rows processed at a time
NUM_CPUS = 8             # Number of files processed in parallel
RESUME = True            # Skip already processed files
MANIFEST_FILE = "manifest.json"  # Local JSON manifest file
```

---

## Run

- Progress is printed to the console  
- `manifest.json` is updated **after each file**  
- Example entry in `manifest.json`:

```json
{
  "v1_5r2_sample-0000.parquet": {
    "input_file": "s3://my-bucket/dolma_parquet/v1_5r2_sample-0000.parquet",
    "output_file": "s3://my-bucket/dolma_enriched/v1_5r2_sample-0000.parquet",
    "status": "success",
    "total_rows": 12345,
    "error": null
  }
}
```

- If a file fails or the pipeline is interrupted, simply rerun the script.  
- Files with `"status":"success"` in `manifest.json` will be skipped.  
- Failed files will be retried on the next run.

---

## Notes / Best Practices

- **Batch size** can be tuned based on your EC2 memory size.  
- **NUM_CPUS** controls parallelism; avoid exceeding instance cores to prevent overloading.  
- **Atomic writes** prevent corrupt output in case of crashes or S3 network errors.  
- Manifest ensures **reproducible, resumable processing**.

---

For production EC2 deployment, ensure S3 credentials are available via environment variables or IAM roles, and adjust `NUM_CPUS` / `BATCH_SIZE` based on instance size.

