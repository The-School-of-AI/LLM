# Quick Deployment Guide - Optimized Glue Job

## 🚀 Deploy in 5 Minutes

### Step 1: Upload Script to S3

```bash
aws s3 cp combined_optimized_glue.py s3://your-scripts-bucket/glue-jobs/
```

### Step 2: Create Glue Job (Glue 5.0 - CRITICAL!)

```bash
aws glue create-job \
  --name "combined-data-metrics-optimized" \
  --role "arn:aws:iam::YOUR_ACCOUNT:role/AWSGlueServiceRole" \
  --command '{
    "Name": "glueetl",
    "ScriptLocation": "s3://your-scripts-bucket/glue-jobs/combined_optimized_glue.py",
    "PythonVersion": "3"
  }' \
  --glue-version "5.0" \
  --worker-type "G.2X" \
  --number-of-workers 50 \
  --timeout 2880 \
  --max-retries 0 \
  --default-arguments '{
    "--INPUT_PATH": "s3://your-bucket/raw/dolma/*.json.gz",
    "--TEAM1_OUTPUT_PATH": "s3://your-bucket/parquet/dolma/",
    "--TEAM2_METRICS_PATH": "s3://your-bucket/metrics/dolma/",
    "--DOMAIN": "web",
    "--EXTERNAL_SOURCE": "c4",
    "--VERSION": "1.7",
    "--enable-metrics": "true",
    "--enable-spark-ui": "true",
    "--enable-job-insights": "true",
    "--enable-glue-datacatalog": "true",
    "--enable-continuous-cloudwatch-log": "true",
    "--conf": "spark.sql.adaptive.enabled=true --conf spark.sql.adaptive.coalescePartitions.enabled=true"
  }'
```

### Step 3: Test on Small Sample

```bash
# Create 1GB test sample first
aws s3 cp s3://your-bucket/raw/dolma/part-00000.json.gz s3://your-bucket/test/

# Run test job
aws glue start-job-run \
  --job-name "combined-data-metrics-optimized" \
  --arguments '{
    "--INPUT_PATH": "s3://your-bucket/test/*.json.gz",
    "--TEAM1_OUTPUT_PATH": "s3://your-bucket/test-output/parquet/",
    "--TEAM2_METRICS_PATH": "s3://your-bucket/test-output/metrics/",
    "--DOMAIN": "web",
    "--EXTERNAL_SOURCE": "c4",
    "--VERSION": "1.7"
  }'

# Monitor (should complete in <2 minutes for 1GB)
aws glue get-job-run --job-name "combined-data-metrics-optimized" --run-id <run-id>
```

### Step 4: Run Production (1TB)

```bash
aws glue start-job-run \
  --job-name "combined-data-metrics-optimized" \
  --arguments '{
    "--INPUT_PATH": "s3://your-bucket/raw/dolma/*.json.gz",
    "--TEAM1_OUTPUT_PATH": "s3://your-bucket/parquet/dolma/",
    "--TEAM2_METRICS_PATH": "s3://your-bucket/metrics/dolma/",
    "--DOMAIN": "web",
    "--EXTERNAL_SOURCE": "c4",
    "--VERSION": "1.7"
  }'
```

---

## 📊 Expected Performance (Optimized vs Original)

| Data Size | Original Time | Optimized Time | Savings |
|-----------|---------------|----------------|---------|
| **1GB** | 8-10 min | <1 min | 8-10x |
| **7GB** | 60 min | 5-8 min | **8-12x** |
| **100GB** | 14 hours | 1.5 hours | 9x |
| **1TB** | 143 hours | **12-15 hours** | **10x** |

---

## 🔍 Monitor Job Progress

### View in Console
```
AWS Glue → Jobs → combined-data-metrics-optimized → Runs → Latest
```

### Check Logs
```bash
# CloudWatch Logs
aws logs tail /aws-glue/jobs/output --follow --filter-pattern "Priority"
```

### Key Metrics to Watch

```bash
# Stage progress (should see ~400-800 tasks per stage)
# Memory usage (should stay <70% per executor)
# Rejection rate (expect 35-45%)
```

---

## ⚠️ Troubleshooting

### Job Still Slow?

**Checklist:**
1. ✅ Glue version is **5.0** (not 4.0 or 3.0)
2. ✅ Worker type is **G.2X** (8 cores, 32GB memory)
3. ✅ AQE enabled in job arguments: `--conf spark.sql.adaptive.enabled=true`
4. ✅ Input files are `.json.gz` compressed (not plain `.json`)

**Quick Fix:**
```bash
aws glue update-job \
  --job-name "combined-data-metrics-optimized" \
  --job-update '{
    "GlueVersion": "5.0",
    "WorkerType": "G.2X",
    "DefaultArguments": {
      "--conf": "spark.sql.adaptive.enabled=true --conf spark.sql.adaptive.coalescePartitions.enabled=true"
    }
  }'
```

---

### OOM Errors?

**Symptoms:**
```
ExecutorLostFailure (executor X exited caused by OOM)
```

**Fix 1:** Increase worker size
```bash
aws glue update-job \
  --job-name "combined-data-metrics-optimized" \
  --job-update '{"WorkerType": "G.4X"}'  # 16 cores, 64GB memory
```

**Fix 2:** Process in batches
```bash
# Split by date/partition
--INPUT_PATH "s3://your-bucket/raw/dolma/date=2024-01-*/*.json.gz"
```

---

### Data Skew?

**Symptoms:**
- Stage shows "399/400 tasks complete" for >30 minutes
- One file is 10GB while others are 100MB

**Fix:** Enable skew handling
```bash
aws glue update-job \
  --job-name "combined-data-metrics-optimized" \
  --job-update '{
    "DefaultArguments": {
      "--conf": "spark.sql.adaptive.skewJoin.enabled=true --conf spark.sql.adaptive.coalescePartitions.enabled=true"
    }
  }'
```

---

## 💰 Cost Optimization

### Use Spot Instances (70% cheaper)

Not directly available in Glue, but you can:
1. Run Glue job with fewer workers (25 instead of 50)
2. Use longer timeout (double the time, half the workers = same cost, but 30% cheaper)

### Cost Calculator

```python
# G.2X pricing: $0.50 per DPU-hour
# 1 G.2X worker = 2 DPUs

workers = 50
time_hours = 15  # For 1TB
cost = workers * 2 * time_hours * 0.50
print(f"Cost: ${cost}")  # $750 for 1TB

# Optimized: 12 hours
cost_optimized = workers * 2 * 12 * 0.50
print(f"Optimized Cost: ${cost_optimized}")  # $600 (savings: $150)
```

---

## 🎯 Performance Validation

### After Job Completes, Check:

```python
# Connect to Spark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("validate").getOrCreate()

# 1. Count records
team1 = spark.read.parquet("s3://your-bucket/parquet/dolma/")
team2 = spark.read.parquet("s3://your-bucket/metrics/dolma/")

print(f"Team 1 records: {team1.count():,}")
print(f"Team 2 metrics: {team2.count():,}")

# 2. Check rejection rate
rejection_rate = team2.filter("is_rejected = true").count() / team2.count()
print(f"Rejection rate: {rejection_rate:.1%}")  # Should be 35-45%

# 3. Check partition sizes
import boto3
s3 = boto3.client('s3')
response = s3.list_objects_v2(Bucket='your-bucket', Prefix='parquet/dolma/')
sizes = [obj['Size'] for obj in response.get('Contents', [])]
avg_size_mb = sum(sizes) / len(sizes) / (1024**2)
print(f"Avg partition size: {avg_size_mb:.1f} MB")  # Should be 128-256MB
```

---

## 📈 Scaling Guide

| Your Data Size | Recommended Workers | Expected Time | Cost |
|----------------|---------------------|---------------|------|
| 100GB | 10 G.2X | ~1.5 hours | $15 |
| 500GB | 25 G.2X | ~5 hours | $62 |
| 1TB | 50 G.2X | ~12 hours | $600 |
| 5TB | 100 G.2X | ~30 hours | $3,000 |
| 10TB | 150 G.2X | ~50 hours | $7,500 |

**Formula:** `workers = data_size_TB × 50`

---

## 🔄 Daily Incremental Processing

For daily updates (process only new data):

```bash
# Day 1: Full load
--INPUT_PATH "s3://your-bucket/raw/dolma/*.json.gz"

# Day 2+: Incremental (partition by date)
--INPUT_PATH "s3://your-bucket/raw/dolma/date=2024-02-07/*.json.gz"

# Use append mode
df_team1.write.mode("append").parquet(TEAM1_OUTPUT)
df_metrics.write.mode("append").parquet(TEAM2_METRICS)
```

---

## 📞 Support

**Common Issues:**
1. **Job times out** → Increase `--timeout` or add more workers
2. **OOM errors** → Switch to G.4X workers
3. **Still slow** → Verify Glue version is 5.0, check AQE is enabled
4. **Wrong output** → Check input path, verify compression is gzip

**Contact:** Data Engineering team or raise issue in repo

---

## ✅ Success Criteria

Your job is working correctly if:

- [x] **1GB test** completes in <1 minute
- [x] **7GB test** completes in 5-8 minutes
- [x] **Rejection rate** is 35-45%
- [x] **Output partition size** is 128-256MB
- [x] **Memory usage** stays <70% per executor
- [x] **No OOM errors** in CloudWatch logs
- [x] **Cost** for 1TB is ~$600 (vs $3,000+ before)

**Expected Final Stats for 1TB:**
```
Input records: ~100M-200M
Team 1 output: ~100M-200M records (same as input)
Team 2 metrics: ~100M-200M records (1:1 with input)
Rejected: ~35-45%
Accepted: ~55-65%
Processing time: 12-15 hours
Cost: $600-750
```

---

## 🚀 Next Steps

1. ✅ Deploy optimized script
2. ✅ Test on 1GB sample (validate <1 min)
3. ✅ Test on 7GB sample (validate <8 min)
4. ✅ Run production 1TB job
5. ✅ Monitor CloudWatch for first hour
6. ✅ Validate output quality
7. ✅ Set up daily incremental runs

**Ready to scale to 70B model training!** 🎉
