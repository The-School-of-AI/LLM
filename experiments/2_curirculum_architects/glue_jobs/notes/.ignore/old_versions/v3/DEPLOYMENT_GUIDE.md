# Deployment Guide - Optimized Glue Jobs

## Quick Start

### 1. Upload Script to S3
```bash
# Upload the optimized Glue script
aws s3 cp combined_t123_optimized.py s3://t1-dataacquisition-datasets/scripts/

# Verify upload
aws s3 ls s3://t1-dataacquisition-datasets/scripts/combined_t123_optimized.py
```

### 2. Create Glue Job
```bash
# Option A: Use orchestrator script (recommended)
python process_all_datasets.py --create-job --script-location s3://t1-dataacquisition-datasets/scripts/combined_t123_optimized.py

# Option B: Use AWS CLI
aws glue create-job \
  --name combined-t123-optimized \
  --role AWSGlueServiceRole-DataProcessing \
  --command "Name=glueetl,ScriptLocation=s3://t1-dataacquisition-datasets/scripts/combined_t123_optimized.py,PythonVersion=3" \
  --glue-version 5.0 \
  --worker-type G.2X \
  --number-of-workers 10 \
  --max-retries 1 \
  --timeout 2880 \
  --default-arguments '{
    "--enable-spark-ui":"true",
    "--enable-metrics":"true",
    "--enable-glue-datacatalog":"true",
    "--enable-continuous-cloudwatch-log":"true",
    "--conf":"spark.sql.adaptive.enabled=true",
    "--conf":"spark.sql.adaptive.coalescePartitions.enabled=true",
    "--conf":"spark.sql.adaptive.skewJoin.enabled=true",
    "--job-language":"python"
  }'
```

### 3. Test on Small Dataset
```bash
# Dry run (validate configuration)
python process_all_datasets.py --dataset dolmas_books_v1_7 --workers 5 --dry-run

# Process small dataset (3 files, ~10GB)
python process_all_datasets.py --dataset dolmas_books_v1_7 --workers 5
```

Expected output:
```
[2026-02-07 10:15:23] Processing dataset 1/1: dolmas_books_v1_7
[2026-02-07 10:15:24] ✓ Job started: jr_abc123
[2026-02-07 10:15:24]   Monitor: https://console.aws.amazon.com/glue/...
[2026-02-07 10:18:45]   Status: RUNNING (elapsed: 3.4 min)
[2026-02-07 10:19:12]   Final Status: SUCCEEDED (total: 3.8 min)
[2026-02-07 10:19:12] ✅ Dataset processed successfully: dolmas_books_v1_7
```

### 4. Verify Output
```bash
# Check processing status
python check_processing_status.py --dataset dolmas_books_v1_7 --detailed

# Check S3 output
aws s3 ls s3://t1-dataacquisition-datasets/processed/dolmas_books_v1_7/
aws s3 ls s3://t1-dataacquisition-datasets/metrics/dolmas_books_v1_7/

# Get rejection statistics
python check_processing_status.py --dataset dolmas_books_v1_7 --rejection-stats
```

---

## Processing All Datasets

### Option 1: Process by Size Category (Recommended)

**Step 1: Small datasets** (validation)
```bash
# 3-10 files, ~10-35GB each
python process_all_datasets.py --size-range small --workers 5
```

**Step 2: Medium datasets** (parallel)
```bash
# 25-100 files, ~50-350GB each
python process_all_datasets.py --size-range medium --workers 15
```

**Step 3: Large datasets** (overnight)
```bash
# 100-500 files, ~200-850GB each
python process_all_datasets.py --size-range large --workers 50
```

### Option 2: Process All at Once
```bash
# Process all 16 datasets sequentially
# Total: ~3.5TB, estimated 48-72 hours
python process_all_datasets.py --workers 25
```

### Option 3: Manual Parallel Execution

For maximum throughput, run multiple datasets in parallel:

```bash
# Terminal 1: Small datasets
python process_all_datasets.py --size-range small --workers 5 &

# Terminal 2: Medium datasets
python process_all_datasets.py --size-range medium --workers 15 &

# Terminal 3: Single large dataset
python process_all_datasets.py --dataset dolma_RefineWeb_v1_7 --workers 50 &

# Monitor all jobs
watch -n 30 'python check_processing_status.py'
```

---

## Monitoring During Execution

### Real-Time Monitoring
```bash
# Terminal 1: Status updates every 30 seconds
watch -n 30 'python check_processing_status.py'

# Terminal 2: View logs
tail -f processing_log.txt
```

### Spark UI (detailed metrics)
```bash
# Get Spark UI URL from job start output
# https://console.aws.amazon.com/glue/...

# Or find in CloudWatch logs
aws logs filter-log-events \
  --log-group-name /aws-glue/jobs/logs-v2 \
  --log-stream-name-prefix combined-t123-optimized \
  --filter-pattern "Spark UI" \
  --max-items 1
```

### Key Metrics to Watch

**1. Task Duration** (Spark UI → Stages → Task Metrics)
- ✅ Good: Max task time ≤ 2x median
- ⚠️ Warning: Max task time = 2-5x median (minor skew)
- ❌ Bad: Max task time > 5x median (major skew, adjust partitioning)

**2. Memory Usage** (CloudWatch → Glue Metrics)
- ✅ Good: Memory usage 50-80%
- ⚠️ Warning: Memory usage 80-95% (may need more workers)
- ❌ Bad: Memory usage >95% or OOM errors (reduce workers or increase worker type)

**3. Progress Rate**
```bash
# Check records processed
aws s3 ls s3://t1-dataacquisition-datasets/metrics/dolma_RefineWeb_v1_7/ --recursive --human-readable | tail -10

# Expected throughput: 20GB/hour per worker
# Example: 50 workers × 20GB/hour = 1TB/hour theoretical max (actual ~600GB/hour)
```

---

## Cost Management

### Estimated Costs (G.2X @ $0.44/DPU-hour)

| Dataset Category | Workers | Est. Time | Est. Cost |
|------------------|---------|-----------|-----------|
| Small (3-10 files) | 5 | 10-20 min | $0.50-1.00 |
| Medium (25-100 files) | 15 | 1-2 hours | $7-15 |
| Large (100-500 files) | 50 | 3-8 hours | $70-175 |

**Total for all 16 datasets**: ~$600-800 (vs $1,500-2,000 with old UDF code)

### Cost Optimization Tips

**1. Use Spot Instances** (70% savings)
```python
# Update Glue job configuration
ExecutionClass='FLEX'  # Spot pricing, up to 10min startup delay
```

**2. Process During Off-Peak Hours**
- AWS pricing is same 24/7, but S3 request rates may be better at night
- Recommended: Start large jobs at 6 PM EST

**3. Enable Job Bookmarks** (for incremental processing)
```python
# Only process new data
"--job-bookmark-option": "job-bookmark-enable"
```

---

## Troubleshooting

### Issue: Job Fails with "Executor Lost"

**Symptoms**: 
```
ExecutorLostFailure (executor 5 exited caused by one of the running tasks)
```

**Causes**:
1. Memory overflow (cache() on large dataset)
2. Worker instance terminated (spot interruption)
3. OOM from large partitions

**Solutions**:
```bash
# Solution 1: Verify no cache() calls in code
grep -n "\.cache()" combined_t123_optimized.py
# Should return nothing

# Solution 2: Increase partition size (reduces memory per task)
--TARGET_PARTITION_SIZE_MB 256  # (default: 192)

# Solution 3: Reduce workers (more memory per worker)
--workers 30  # (instead of 50)

# Solution 4: Use STANDARD instead of FLEX
ExecutionClass='STANDARD'  # More stable, no spot interruptions
```

### Issue: Task Skew (Long Tail)

**Symptoms**: Most tasks finish quickly, but 1-2 tasks take 10x longer

**Diagnosis**:
```bash
# Check Spark UI → Stages → Task Metrics
# Look for: Max Duration > 5x Median Duration
```

**Causes**:
1. One input file much larger than others
2. One partition has more complex data (longer text)

**Solutions**:
```python
# Enable skew join optimization (already enabled in script)
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

# Increase target partition size (fewer, larger partitions)
--TARGET_PARTITION_SIZE_MB 256

# Repartition by hash of text length (distributes complexity)
df = df.repartition(num_partitions, F.hash(F.length(F.col("text"))))
```

### Issue: S3 Throttling (503 SlowDown)

**Symptoms**:
```
Caused by: org.apache.hadoop.fs.s3a.AWSServiceThrottledException: 
  Slow Down (Service: Amazon S3; Status Code: 503)
```

**Causes**:
1. Too many small files (high request rate)
2. Hot partition (many requests to same S3 prefix)

**Solutions**:
```bash
# Solution 1: Increase partition size (fewer files)
--TARGET_PARTITION_SIZE_MB 256

# Solution 2: Add random prefix to output paths
# Already handled by Spark's hash partitioner

# Solution 3: Request S3 request rate increase
# aws support create-case --service-code s3 --category-code performance
```

### Issue: Regex Timeout

**Symptoms**: Tasks hang for hours without progress

**Diagnosis**:
```bash
# Check Spark UI → Stages → Tasks
# Look for: Tasks running >1 hour without shuffling data
```

**Causes**:
- Catastrophic backtracking in regex patterns

**Solutions**:
```python
# Solution: Already optimized in code with:
# 1. Non-capturing groups (?:...)
# 2. Limited input length for complex patterns
# 3. Pre-compiled patterns in Spark SQL

# If still occurring, identify problematic pattern:
# Add .withColumn("_debug_text_length", F.length(F.col("text")))
# Check which records have extreme length (>1MB)
```

---

## Checkpoint and Resume

### Resume After Failure
```bash
# Check checkpoint status
cat processing_checkpoint.json

# Resume processing (skips completed datasets)
python process_all_datasets.py --workers 25

# Manual checkpoint edit (if needed)
# Edit processing_checkpoint.json to mark dataset as completed/failed
```

### Clear Checkpoint (Start Fresh)
```bash
rm processing_checkpoint.json
rm processing_log.txt
python process_all_datasets.py --workers 25
```

---

## Validation

### After Processing Completes

**1. Check all datasets processed**
```bash
python check_processing_status.py
# Should show all datasets as "✅ completed"
```

**2. Verify output quality**
```bash
# Check acceptance rates
python check_processing_status.py --rejection-stats

# Expected: 60-70% acceptance rate
# If <50%: Thresholds may be too strict
# If >90%: Thresholds may be too loose
```

**3. Sample data quality**
```bash
# Read sample records from each dataset
python sample_quality_check.py --datasets all --sample-size 100
```

**4. Verify metrics completeness**
```python
# Check for NULL metrics (should be minimal except placeholders)
SELECT 
    COUNT(*) as total,
    SUM(CASE WHEN byte_length IS NULL THEN 1 ELSE 0 END) as null_byte_length,
    SUM(CASE WHEN unique_token_ratio IS NULL THEN 1 ELSE 0 END) as null_unique_token
FROM metrics.dolma_all
-- null_byte_length and null_unique_token should be 0
```

---

## Post-Processing (Optional)

### 1. Compute Advanced Metrics

Placeholders like `mtld`, `fertility`, `rare_word_ratio` require external libraries:

```bash
# Create separate Python environment job
python create_advanced_metrics_job.py --input s3://metrics/ --output s3://metrics_enriched/
```

### 2. Create Athena Tables

```sql
-- Team 1: Transformed data
CREATE EXTERNAL TABLE dolma_transformed (
    id STRING,
    hash STRING,
    dataset STRING,
    domain STRING,
    source STRING,
    text STRING,
    language STRING,
    metadata STRING,
    added TIMESTAMP,
    created TIMESTAMP,
    version STRING
)
STORED AS PARQUET
LOCATION 's3://t1-dataacquisition-datasets/processed/';

-- Team 2: Metrics
CREATE EXTERNAL TABLE dolma_metrics (
    metric_record_uuid STRING,
    source_record_id STRING,
    is_rejected BOOLEAN,
    rejection_reason STRING,
    byte_length INT,
    char_length INT,
    token_count_estimate INT,
    -- ... all other metrics
)
STORED AS PARQUET
LOCATION 's3://t1-dataacquisition-datasets/metrics/';

-- Add partitions
MSCK REPAIR TABLE dolma_transformed;
MSCK REPAIR TABLE dolma_metrics;
```

### 3. Create Curriculum Sampling Query

```sql
-- Sample balanced curriculum (70B model training)
WITH difficulty_scored AS (
    SELECT 
        source_record_id,
        domain_signal,
        (structural_complexity_score * 0.4 +
         (100 - flesch_reading_ease) / 100 * 0.3 +
         dependency_depth_estimate / 20 * 0.3) AS difficulty_score,
        (1.0 - noise_score) * unique_token_ratio * 
         sentence_boundary_coherence AS quality_weight
    FROM dolma_metrics
    WHERE is_rejected = false
),
stratified AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (
            PARTITION BY domain_signal, 
            NTILE(10) OVER (ORDER BY difficulty_score)  -- 10 difficulty buckets
            ORDER BY quality_weight DESC
        ) as rn
    FROM difficulty_scored
)
SELECT source_record_id
FROM stratified
WHERE rn <= 1000  -- Top 1000 per domain-difficulty bucket
ORDER BY domain_signal, difficulty_score;
```

---

## Support

**Questions?** Contact:
- Team Slack: #team2-curriculum-architects
- Email: curriculum-architects@company.com

**Report Issues**:
- GitHub: Create issue in LLM repo
- Include: Dataset name, job run ID, error logs
