# Metrics Computation Glue Job - Summary

## What You Want

### Overview
Compute text quality and curriculum metrics on ~1TB of data processed by Team 1, creating a **separate metrics parquet file** without disrupting their workflow.

### Key Requirements

1. **Input Source**: Read from Team 1's parquet files (from their glue job output)

2. **Output Structure**: Each record in the metrics file contains:
   - `metric_record_uuid`: Unique identifier for this metrics record
   - `source_record_id`: The `id` from Team 1's file
   - `source_file_path`: S3 path to the Team 1 parquet file (for joining later)
   - `[metric columns]`: 59 computed metrics as individual columns
   - `is_rejected`: Boolean flag indicating if record failed quality checks
   - `rejection_reason`: String explaining why record was rejected

3. **Early Rejection Strategy**:
   - Compute metrics in **rejection priority order** (priority 1 → 2 → 3)
   - If a metric causes rejection, **stop computing remaining metrics** for that record
   - Immediately set `is_rejected=True` and `rejection_reason` explaining which metric failed
   - Move to next record (saves computation time)

4. **Metric Priorities** (from CSV):
   - **Priority 1**: Fast, fundamental checks (byte_length, char_length, token_count_estimate, non_printable_ratio)
   - **Priority 2**: Lexical diversity and noise checks (unique_token_ratio, compression_ratio, capitalization_ratio, whitespace_ratio, truncation_indicators, noise_score)
   - **Priority 3**: Complex structural checks (avg_sentence_length, flesch_reading_ease, dependency_depth_estimate, url_count, sentence_boundary_coherence, information_density)

5. **Output File**:
   - Write to **separate S3 path** from Team 1
   - Parquet format with zstd compression
   - Includes `source_file_path` column for later joins
   - Can be joined to Team 1 data using `source_record_id` + `source_file_path`

## Implementation Strategy

### Rejection Metrics (22 total)
From the CSV, these metrics have rejection criteria:

**Priority 1 (Reject Early):**
- `byte_length`: <50 OR >1,000,000
- `char_length`: <20 OR >500,000
- `token_count_estimate`: <10 OR >128,000
- `non_printable_ratio`: >0.01

**Priority 2 (Medium Priority):**
- `unique_token_ratio`: <0.1
- `compression_ratio`: >0.95
- `sentence_count_estimate`: <2 AND token_count>100 (compound)
- `capitalization_ratio`: >0.5
- `whitespace_ratio`: >0.6
- `truncation_indicators`: >2
- `noise_score`: >0.6

**Priority 3 (Final Checks):**
- `avg_sentence_length`: >500
- `flesch_reading_ease`: <0 OR >120
- `dependency_depth_estimate`: >20
- `url_count`: url_ratio>0.3 (requires calculation)
- `sentence_boundary_coherence`: <0.5
- `information_density`: <0.2

### Performance Optimizations
1. **Lazy computation**: Only compute metrics needed up to rejection point
2. **Vectorization**: Use Spark UDFs with pandas_udf for batch processing
3. **Caching**: Tokenizer instances, regex patterns compiled once
4. **Partitioning**: Distribute work across Spark executors efficiently
5. **Early termination**: Skip expensive metrics for rejected records

## Benefits
- ✅ No disruption to Team 1's pipeline
- ✅ Separate metrics file for analysis
- ✅ Can join back to source data anytime
- ✅ Efficient processing with early rejection
- ✅ Clear audit trail of rejections
- ✅ Reusable metrics for curriculum learning
