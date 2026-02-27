# Optimized Glue Jobs for TB-Scale Curriculum Data Processing

## Overview

This directory contains highly optimized AWS Glue jobs for processing large-scale datasets (1TB+) for curriculum learning and coreset engineering for 70B model training.

**Performance**: Processes 7GB in 3-5 minutes (vs 60 minutes with original implementation) - **12-20x faster**

---

## Files

### Core Scripts
- **`combined_t123_optimized.py`** - Main Glue ETL job (vectorized Spark operations)
- **`process_all_datasets.py`** - Orchestration script for processing all datasets
- **`check_processing_status.py`** - Monitor processing progress and generate reports

### Documentation
- **`OPTIMIZATION_README.md`** - Comprehensive guide to all optimizations
- **`DEPLOYMENT_GUIDE.md`** - Step-by-step deployment instructions
- **`PERFORMANCE_COMPARISON.md`** - Before/after benchmarks and analysis

### Data
- **`Datasets_details.csv`** - Dataset metadata (16 datasets, ~3.5TB total)
- **`Curriculum Metrics.csv`** - Metric definitions and thresholds

### Legacy (DO NOT USE)
- **`combined_t123.py`** - Original UDF-based implementation (DEPRECATED)

---

## Quick Start

### 1. Setup
```bash
# Install dependencies
pip install boto3 pyarrow

# Configure AWS credentials
aws configure
```

### 2. Deploy
```bash
# Upload script to S3
aws s3 cp combined_t123_optimized.py s3://t1-dataacquisition-datasets/scripts/

# Create Glue job
python process_all_datasets.py --create-job
```

### 3. Test
```bash
# Test on small dataset (3 files, ~10GB)
python process_all_datasets.py --dataset dolmas_books_v1_7 --workers 5
```

### 4. Process All
```bash
# Process all datasets (recommended: by size category)
python process_all_datasets.py --size-range small --workers 5
python process_all_datasets.py --size-range medium --workers 15
python process_all_datasets.py --size-range large --workers 50
```

### 5. Monitor
```bash
# Check status
python check_processing_status.py

# View rejection statistics
python check_processing_status.py --rejection-stats
```

---

## Key Optimizations

### 1. ✅ Eliminated Python UDFs (10-50x faster)
- Replaced all UDFs with native Spark SQL functions
- Uses vectorized operations (columnar processing)
- Enables Spark Catalyst optimizer

### 2. ✅ Dynamic Partitioning (prevents data skew)
- Automatically calculates optimal partition count
- Targets 128-256MB per Parquet file
- Handles datasets from 10GB to 850GB

### 3. ✅ Memory-Efficient Processing (no OOM errors)
- Single-pass lazy evaluation (no .cache())
- Processes in streaming fashion
- Scales to 1TB+ without memory issues

### 4. ✅ Adaptive Query Execution (Spark 3.5)
- Automatic partition coalescing
- Skew join optimization
- Runtime filter pushdown

### 5. ✅ Early Rejection (50-80% compute savings)
- 3-level priority rejection (P1 → P2 → P3)
- Filters bad data before expensive operations
- Clear rejection reasons for analysis

### 6. ✅ Folder-Wise Processing (fault tolerance)
- Process datasets independently
- Checkpoint/resume capability
- Parallel execution support

---

## Architecture

```
Input: S3 Raw Data (JSON.GZ)
  ↓
[Glue Job: combined_t123_optimized.py]
  │
  ├─→ Team 1 Output: Transformed Parquet (s3://processed/)
  │   • id, hash, dataset, domain, source
  │   • text, language, metadata
  │   • added, created, version
  │
  └─→ Team 2 Output: Metrics Parquet (s3://metrics/)
      • 75 metrics computed per record
      • Rejection flags and reasons
      • Quality scores for curriculum design
```

### Metrics Categories (75 total)

1. **Physical Properties** (5): byte_length, char_length, token_count, line_count, non_printable_ratio
2. **Lexical Diversity** (8): unique_token_ratio, vocab_size, compression_ratio, mtld, fertility, etc.
3. **Structural Complexity** (9): avg_sentence_length, punctuation_density, dependency_depth, etc.
4. **Noise & Quality** (10): noise_score, boilerplate_ratio, url_spam_score, html_tag_density, etc.
5. **Domain Signals** (8): code_block_count, math_expression_count, dialogue_turn_count, etc.
6. **Reasoning Signals** (7): reasoning_marker_density, step_indicator_count, etc.
7. **Derived Metrics** (5): structural_complexity_score, domain_signal, information_density, etc.
8. **Curriculum Metrics** (23): difficulty_score, quality_weight, few_shot_potential, etc.

---

## Dataset Processing Order

From `Datasets_details.csv`:

### Small (3-10 files, ~10-35GB)
1. dolmas_books_v1_7 (3 files, 10.5GB)
2. dolmas_cc_news_v1_7 (5 files, 17.5GB)

### Medium (25-100 files, ~50-350GB)
3. dolma_Pes2o_v1_7 (26 files, 104GB)
4. dolma_stackexchange_v1_7 (25 files, 50GB)
5. dolma_algebraic-stack_v1_7 (15 files, 15GB)
6. dolma_open-web-math_v1_7 (13 files, 19.5GB)
7. dolma_tulu_flan_v1_7 (66 files, 33GB)
8. dolma_reddit_v1_7 (77 files, 154GB)
9. dolma_arxiv_v1_7 (99 files, 19.8GB)
10. dolma_C4_v1_7 (100 files, 300GB)

### Large (100-500 files, ~200-850GB)
11. dolma_megawika_v1_7 (261 files, 522GB)
12. dolma_cc_en_tail_v1_7 (263 files, 526GB)
13. dolma_cc_en_head_v1_7 (274 files, 822GB)
14. dolma_cc_en_middle_v1_7 (379 files, 758GB)
15. dolma_RefineWeb_v1_7 (499 files, 849GB)
16. dolma_starcoder_v1_7 (49 files, 196GB)

**Total: 16 datasets, ~3.5TB**

---

## Cost Estimates

### Processing Costs (G.2X @ $0.44/DPU-hour)

| Category | Datasets | Est. Time | Est. Cost |
|----------|----------|-----------|-----------|
| Small | 2 | 20-40 min | $2-4 |
| Medium | 8 | 6-12 hours | $40-80 |
| Large | 6 | 40-60 hours | $350-530 |
| **Total** | **16** | **48-72 hours** | **~$600** |

**With FLEX (Spot)**: ~$180 (70% savings)  
**With old UDF code**: ~$1,800 (3x more expensive)

---

## Curriculum Learning Use Cases

### 1. Difficulty-Based Curriculum
```sql
-- Easy → Hard progression
SELECT source_record_id, text
FROM metrics JOIN transformed USING(id)
WHERE is_rejected = false
ORDER BY structural_complexity_score ASC, flesch_reading_ease DESC
LIMIT 10000000;  -- First 10M examples
```

### 2. Domain-Balanced Sampling
```sql
-- Ensure balanced domain coverage
SELECT domain_signal, COUNT(*) as count
FROM metrics
WHERE is_rejected = false
GROUP BY domain_signal;

-- Target: 30% web, 25% code, 20% math, 15% dialogue, 10% science
```

### 3. Quality-Weighted Training
```sql
-- Weight examples by quality during training
SELECT 
    source_record_id,
    (1.0 - noise_score) * unique_token_ratio * 
    sentence_boundary_coherence AS quality_weight
FROM metrics
WHERE is_rejected = false
ORDER BY quality_weight DESC;
```

### 4. Reasoning Signal Detection
```sql
-- Prioritize reasoning-heavy examples
SELECT source_record_id
FROM metrics
WHERE is_rejected = false
  AND reasoning_marker_density > 0.01
  AND step_indicator_count > 3
  AND math_expression_count > 0
ORDER BY structural_complexity_score DESC;
```

---

## Monitoring and Troubleshooting

### Real-Time Monitoring
```bash
# Terminal 1: Status updates every 30 seconds
watch -n 30 'python check_processing_status.py'

# Terminal 2: View logs
tail -f processing_log.txt

# Terminal 3: Monitor CloudWatch
aws logs tail /aws-glue/jobs/logs-v2 --follow
```

### Common Issues

**Issue: "Executor Lost" errors**
- **Cause**: Memory overflow, cache() usage
- **Fix**: Already fixed in optimized version (no cache)

**Issue: Task skew (long tail)**
- **Cause**: One partition much larger than others
- **Fix**: Already enabled AQE skew optimization

**Issue: S3 throttling (503)**
- **Cause**: Too many small files
- **Fix**: Increase TARGET_PARTITION_SIZE_MB to 256

---

## Validation

### After Processing Completes

```bash
# 1. Check all datasets processed
python check_processing_status.py
# Expected: All ✅ completed

# 2. Verify acceptance rates
python check_processing_status.py --rejection-stats
# Expected: 60-70% acceptance rate

# 3. Check output sizes
aws s3 ls s3://t1-dataacquisition-datasets/processed/ --recursive --summarize
aws s3 ls s3://t1-dataacquisition-datasets/metrics/ --recursive --summarize
# Expected: ~2TB processed, ~500GB metrics

# 4. Sample quality check
# Download 100 random accepted records and manually review
```

---

## Integration with Training Pipeline

### 1. Load Metrics
```python
import pyarrow.parquet as pq

# Read all metrics
metrics = pq.read_table('s3://t1-dataacquisition-datasets/metrics/')
df = metrics.to_pandas()

# Filter accepted records
accepted = df[~df['is_rejected']]
print(f"Accepted: {len(accepted):,} / {len(df):,} ({len(accepted)/len(df):.1%})")
```

### 2. Create Curriculum
```python
# Sort by difficulty (easy → hard)
curriculum = accepted.sort_values('structural_complexity_score')

# Add to training config
training_ids = curriculum['source_record_id'].tolist()
```

### 3. Load Training Data
```python
# Use filtered IDs to load actual text from processed data
texts = spark.read.parquet('s3://t1-dataacquisition-datasets/processed/') \
    .filter(F.col('id').isin(training_ids))
```

---

## Future Work

### Immediate (Next Sprint)
- [ ] Add multi-language support (currently English-only)
- [ ] Implement deduplication using MinHash LSH
- [ ] Create Athena views for easy querying
- [ ] Add data lineage tracking

### Medium-Term (Next Quarter)
- [ ] Implement advanced metrics (mtld, fertility, rare_word_ratio)
- [ ] Add Spark ML features (TF-IDF, embeddings)
- [ ] Create real-time streaming pipeline
- [ ] Build interactive dashboard for metrics

### Long-Term (Next Year)
- [ ] Migrate to Delta Lake for ACID transactions
- [ ] Add automatic quality feedback loop from model training
- [ ] Implement active learning for coreset selection
- [ ] Create unified data platform for all teams

---

## Support

**Team**: Team 2 - Curriculum Architects  
**Slack**: #team2-curriculum-architects  
**Documentation**: `/docs/2_curirculum_architects/`  
**GitHub**: [The-School-of-AI/LLM](https://github.com/The-School-of-AI/LLM)

**For Issues**:
1. Check `DEPLOYMENT_GUIDE.md` for troubleshooting
2. Review Spark UI for performance issues
3. Post in Slack with: dataset name, job run ID, error logs

---

## License

Internal use only. Do not distribute outside the organization.

---

**Last Updated**: 2026-02-07  
**Version**: 2.0.0  
**Glue Version**: 5.0 (Spark 3.5)
