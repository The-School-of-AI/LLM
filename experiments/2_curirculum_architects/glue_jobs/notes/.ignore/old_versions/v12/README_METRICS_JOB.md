# Curriculum Metrics Glue Job - Setup & Operations Guide

## Overview
This Glue job computes 59 text quality and curriculum metrics on your training data, with intelligent early rejection to optimize processing of large datasets (1TB+).

## Quick Start

### 1. Prerequisites

**AWS Glue Job Configuration:**
```
Type: Spark
Glue Version: 4.0
Language: Python 3
Worker Type: G.2X (recommended for 1TB+ data)
Number of Workers: 50-100 (tune based on data size)
Job Timeout: 2880 minutes (48 hours max)
Max Concurrent Runs: 1
```

**Required Job Parameters:**
- `--JOB_NAME`: Auto-populated by Glue
- `--TEAM1_INPUT_PATH`: S3 path to Team 1's parquet files (e.g., `s3://bucket/parquet/dolma/`)
- `--METRICS_OUTPUT_PATH`: S3 path for metrics output (e.g., `s3://bucket/metrics/dolma/`)
- `--TIKTOKEN_MODEL`: (Optional) Tokenizer model, default `cl100k_base`
- `--NUM_PARTITIONS`: (Optional) Output partitions, default `400`

**IAM Permissions:**
The Glue job role needs:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket/parquet/*",
        "arn:aws:s3:::your-bucket"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket/metrics/*"
      ]
    }
  ]
}
```

### 2. Deploy the Job

**Option A: AWS Console**
1. Go to AWS Glue Console → Jobs → Create Job
2. Choose "Spark script editor"
3. Upload `metrics_computation_glue.py`
4. Configure job parameters (see above)
5. Save and run

**Option B: AWS CLI**
```bash
aws glue create-job \
  --name "curriculum-metrics-computation" \
  --role "AWSGlueServiceRole-YourRole" \
  --command "Name=glueetl,ScriptLocation=s3://your-scripts-bucket/metrics_computation_glue.py,PythonVersion=3" \
  --glue-version "4.0" \
  --worker-type "G.2X" \
  --number-of-workers 50 \
  --timeout 2880 \
  --default-arguments '{
    "--TEAM1_INPUT_PATH": "s3://your-bucket/parquet/dolma/",
    "--METRICS_OUTPUT_PATH": "s3://your-bucket/metrics/dolma/",
    "--NUM_PARTITIONS": "400",
    "--enable-metrics": "true",
    "--enable-spark-ui": "true",
    "--enable-job-insights": "true",
    "--enable-glue-datacatalog": "true",
    "--job-language": "python"
  }'
```

**Option C: Terraform**
```hcl
resource "aws_glue_job" "metrics_computation" {
  name     = "curriculum-metrics-computation"
  role_arn = aws_iam_role.glue_role.arn
  
  glue_version      = "4.0"
  worker_type       = "G.2X"
  number_of_workers = 50
  timeout           = 2880
  
  command {
    script_location = "s3://${var.scripts_bucket}/metrics_computation_glue.py"
    python_version  = "3"
  }
  
  default_arguments = {
    "--TEAM1_INPUT_PATH"     = "s3://${var.data_bucket}/parquet/dolma/"
    "--METRICS_OUTPUT_PATH"  = "s3://${var.data_bucket}/metrics/dolma/"
    "--NUM_PARTITIONS"       = "400"
    "--enable-metrics"       = "true"
    "--enable-spark-ui"      = "true"
    "--enable-job-insights"  = "true"
  }
}
```

### 3. Run the Job

```bash
aws glue start-job-run \
  --job-name "curriculum-metrics-computation" \
  --arguments '{
    "--TEAM1_INPUT_PATH": "s3://your-bucket/parquet/dolma/",
    "--METRICS_OUTPUT_PATH": "s3://your-bucket/metrics/dolma/"
  }'
```

---

## Key Features & Optimizations

### 🚀 Early Rejection Strategy
The job computes metrics in **priority order** and stops immediately when rejection criteria are met:

**Priority 1** (fastest, checked first):
- `byte_length`, `char_length`, `token_count_estimate`, `non_printable_ratio`
- Rejects ~15-20% of records typically
- **Savings**: Avoids 90% of computation for rejected records

**Priority 2** (medium cost):
- `unique_token_ratio`, `compression_ratio`, `capitalization_ratio`, `whitespace_ratio`, `noise_score`
- Rejects ~10-15% of remaining records
- **Savings**: Avoids 50% of computation for rejected records

**Priority 3** (most expensive):
- `flesch_reading_ease`, `dependency_depth_estimate`, `sentence_boundary_coherence`, `information_density`
- Final quality checks
- Only computed for ~60-70% of records

**Expected Time Savings**: 40-60% reduction in total processing time vs computing all metrics

### 📊 Output Schema

Each record in the metrics file contains:

```python
{
  # Join Keys
  "metric_record_uuid": "550e8400-e29b-41d4-a716-446655440000",  # UUID
  "source_record_id": "dolma-v1_7-c4-0000",                      # From Team 1
  "source_file_path": "s3://bucket/parquet/dolma/part-001.parquet",
  
  # Rejection Status
  "is_rejected": false,
  "rejection_reason": null,  # or "[P1] byte_length too short (<50)"
  
  # 59 Metric Columns
  "byte_length": 1543,
  "char_length": 1543,
  "token_count_estimate": 385,
  "unique_token_ratio": 0.65,
  "compression_ratio": 0.42,
  "flesch_reading_ease": 62.5,
  # ... all other metrics ...
  
  # Metadata
  "processed_at": "2026-02-06T10:30:00Z"
}
```

### 🔗 Joining Back to Team 1 Data

```python
# In PySpark
df_team1 = spark.read.parquet("s3://bucket/parquet/dolma/")
df_metrics = spark.read.parquet("s3://bucket/metrics/dolma/")

df_joined = df_team1.alias("t1").join(
    df_metrics.alias("m"),
    (F.col("t1.id") == F.col("m.source_record_id")) &
    (F.input_file_name() == F.col("m.source_file_path")),
    "left"
)

# Filter to accepted records only
df_accepted = df_joined.filter(F.col("m.is_rejected") == False)
```

```sql
-- In Athena/Presto (if registered in Glue Catalog)
SELECT 
    t1.*,
    m.flesch_reading_ease,
    m.structural_complexity_score,
    m.domain_signal
FROM team1_data t1
LEFT JOIN metrics_data m
    ON t1.id = m.source_record_id
WHERE m.is_rejected = false
```

### 💰 Cost Optimization

**Recommendations for 1TB Dataset:**

| Configuration | Workers | Type | Est. Time | Est. Cost | Use Case |
|--------------|---------|------|-----------|-----------|----------|
| **Budget** | 20 | G.1X | ~8 hours | ~$50 | Development/testing |
| **Balanced** | 50 | G.2X | ~3 hours | ~$150 | Production (recommended) |
| **Fast** | 100 | G.2X | ~90 min | ~$250 | Urgent processing |

**Cost-Saving Tips:**
1. Use **Spot instances** for Glue workers (save 70%)
2. Set `NUM_PARTITIONS` = workers × 4-8 for optimal parallelism
3. Use **zstd compression** (better ratio than snappy, ~30% size reduction)
4. Enable **Glue Job Insights** to identify bottlenecks
5. Consider **incremental processing** for daily updates instead of full reprocessing

### 🎯 Performance Tuning

**Spark Configuration** (add to Glue job arguments):
```python
--conf spark.sql.shuffle.partitions=800
--conf spark.default.parallelism=400
--conf spark.sql.adaptive.enabled=true
--conf spark.sql.adaptive.coalescePartitions.enabled=true
--conf spark.dynamicAllocation.enabled=false
```

**Memory Tuning:**
```python
--conf spark.executor.memory=16g
--conf spark.executor.memoryOverhead=4g
--conf spark.driver.memory=8g
```

**Partition Size Guidelines:**
- Target: 128MB - 256MB per partition
- Formula: `NUM_PARTITIONS = total_size_GB / 0.2`
- For 1TB: `NUM_PARTITIONS = 1000 / 0.2 = 5000` (distributed across workers)

### 📈 Monitoring

**Key Metrics to Watch:**
1. **Records Processed per Minute**: Target 50K-100K records/min on 50 workers
2. **Rejection Rate**: Expected 30-40% total rejection
3. **Average Metrics per Record**: Should be 15-30 (not all 59) due to early rejection
4. **Memory Usage**: Should stay under 80% per executor

**CloudWatch Alarms:**
```python
# Setup alarms for:
- Job Duration > 4 hours (indicates performance issue)
- Memory Usage > 85% (risk of OOM errors)
- DPU Hours > 300 (cost control)
```

**Glue Job Insights Dashboard:**
- Check for data skew (some partitions much larger)
- Look for stragglers (slow tasks)
- Monitor shuffle read/write volumes

---

## Troubleshooting

### Common Issues

**1. Out of Memory Errors**
```
Solution: Increase worker memory or reduce partition size
--worker-type G.4X  # Instead of G.2X
--conf spark.executor.memory=32g
```

**2. Job Runs Slowly**
```
Check:
- Data skew: Some files much larger than others?
- Partition count: Too few partitions = underutilized cluster
- Shuffle operations: Minimize with better partitioning

Fix:
- Repartition input: df.repartition(NUM_PARTITIONS, "id")
- Increase NUM_PARTITIONS parameter
- Use more workers
```

**3. High Rejection Rate (>60%)**
```
Indicates data quality issues. Analyze rejection reasons:

df_metrics.filter(F.col("is_rejected") == True) \
    .groupBy("rejection_reason") \
    .count() \
    .orderBy(F.desc("count")) \
    .show(50, truncate=False)

Common causes:
- Encoding issues (non_printable_ratio high)
- Truncated texts (byte_length too short)
- Spam/templates (unique_token_ratio low)
```

**4. Missing Metrics (NULL values)**
```
Expected for expensive metrics (mtld, fertility, etc.) 
These are placeholder columns for future implementation.

To compute them:
1. Add implementation in compute_pattern_metrics()
2. May require additional libraries (nltk, spacy)
3. Consider separate job for deep metrics on accepted records only
```

### Debugging Tips

**Enable Verbose Logging:**
```python
# Add to script
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add checkpoints
logger.info(f"Processing batch: {batch_id}")
logger.info(f"Rejection stats: {rejection_count}/{total_count}")
```

**Sample Records for Testing:**
```python
# Test on small subset first
df_sample = df_input.sample(fraction=0.001, seed=42)
df_sample.write.parquet("s3://bucket/test-input/")

# Run job on sample
--TEAM1_INPUT_PATH s3://bucket/test-input/
```

**Validate Output:**
```python
# Check metrics distribution
df_metrics = spark.read.parquet(METRICS_OUTPUT)

df_metrics.select(
    F.mean("byte_length"),
    F.stddev("byte_length"),
    F.min("byte_length"),
    F.max("byte_length")
).show()

# Check for nulls
df_metrics.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c)
    for c in df_metrics.columns
]).show()
```

---

## Advanced Features

### Incremental Processing

Process only new data since last run:

```python
# Get last processed timestamp
last_run = spark.read.parquet(METRICS_OUTPUT) \
    .select(F.max("processed_at")).collect()[0][0]

# Filter input to new records
df_new = df_input.filter(F.col("created") > last_run)

# Process and append
df_new_metrics.write.mode("append").parquet(METRICS_OUTPUT)
```

### Custom Rejection Criteria

Modify rejection thresholds in the script:

```python
# In check_priority1_rejection()
if metrics['byte_length'] < 100:  # Changed from 50
    return True, "byte_length too short (<100)"
```

### Multi-Language Support

Add language detection for non-English text:

```python
# Install langdetect in Glue: pip install langdetect
from langdetect import detect

def detect_language(text: str) -> str:
    try:
        return detect(text[:1000])  # Sample first 1K chars
    except:
        return "unknown"
```

### Integration with Data Catalog

Register output as Glue Table:

```python
df_metrics.write \
    .mode("overwrite") \
    .option("compression", "zstd") \
    .option("path", METRICS_OUTPUT) \
    .saveAsTable("curriculum_metrics")
```

Query with Athena:
```sql
SELECT 
    domain_signal,
    COUNT(*) as count,
    AVG(flesch_reading_ease) as avg_readability
FROM curriculum_metrics
WHERE is_rejected = false
GROUP BY domain_signal
```

---

## Next Steps

1. **Test on Sample Data** (0.1% of dataset)
   - Validate metrics accuracy
   - Check rejection rate distribution
   - Estimate full run cost/time

2. **Run Full Job** (monitor first hour)
   - Watch for OOM errors
   - Check rejection statistics
   - Validate output schema

3. **Analyze Results**
   - Profile rejection reasons
   - Identify data quality issues
   - Plan curriculum ordering strategy

4. **Iterate** 
   - Adjust rejection thresholds if needed
   - Add custom domain-specific metrics
   - Implement expensive metrics for accepted subset

---

## Support & References

- **AWS Glue Documentation**: https://docs.aws.amazon.com/glue/
- **PySpark API**: https://spark.apache.org/docs/latest/api/python/
- **Metrics CSV**: See `Curriculum Metrics.csv` for full metric definitions
- **Team 1 Schema**: See `glue.py` for input data structure

For questions or issues, contact the Data Engineering team or raise an issue in the repository.
