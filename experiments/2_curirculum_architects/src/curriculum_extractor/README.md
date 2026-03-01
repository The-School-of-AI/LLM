# curriculum_extractor

Scalable metadata extraction pipeline for processing large parquet datasets.

## Overview

`curriculum_extractor` extracts curriculum metadata from text data for LLM training. It's designed for:

- **~1TB+ parquet data** processing from S3 or local storage
- **Incremental processing** with fault-tolerant state management
- **Read-only record handling** - source data is never modified
- **Early rejection** - stops processing on first quality failure
- **Distributed processing** with Ray support

## Key Design Principles

1. **Records are READ-ONLY**: Plugins receive immutable record wrappers
2. **No plugin chaining**: Each metric operates independently on original data
3. **Early rejection**: If any metric rejects a record, remaining metrics are skipped
4. **Level-based execution**: Metrics at the same level can potentially run in parallel
5. **Band assignment is separate**: Happens in post-processing, not during extraction

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Single Record Extraction

```python
from curriculum_extractor import CurriculumExtractor

extractor = CurriculumExtractor("curriculum.yaml")

record = {"id": "1", "text": "Hello world!"}
metadata, rejection = extractor.extract_record(record)

if rejection:
    print(f"Rejected: {rejection.rejected_reason}")
else:
    print(f"Metadata: {metadata}")
```

### Batch Processing

```python
records = [
    {"id": "1", "text": "Sample 1"},
    {"id": "2", "text": "Sample 2"},
]

processed, rejected = extractor.process_batch(records)
print(f"Processed: {len(processed)}, Rejected: {len(rejected)}")
```

### Parquet Processing with State Management

```python
from curriculum_extractor import CurriculumExtractor
from curriculum_extractor.core.state_manager import StateManager

state_manager = StateManager("./state")

extractor = CurriculumExtractor(
    "curriculum.yaml",
    state_manager=state_manager,
    metadata_output_path="./metadata",
    rejection_output_path="./rejections",
)

result = extractor.process_parquet("data.parquet")
print(result)  # {"status": "completed", "processed_rows": 100, ...}

# Resume if interrupted - already processed files are skipped
result = extractor.process_parquet("data.parquet")  # status: "skipped"
```

### S3 Processing

```python
import s3fs
from curriculum_extractor import CurriculumExtractor

fs = s3fs.S3FileSystem()

extractor = CurriculumExtractor(
    "curriculum.yaml",
    filesystem=fs,
    metadata_output_path="s3://bucket/metadata/",
    rejection_output_path="s3://bucket/rejections/",
)

result = extractor.process_parquet_s3("s3://bucket/data/file.parquet")
```

## CLI Usage

### Extract Metadata

```bash
python -m curriculum_extractor.scripts.run_pipeline \
    --config curriculum.yaml \
    --input-path ./data/ \
    --metadata-output ./metadata/ \
    --rejection-output ./rejections/ \
    --batch-size 10000
```

### Assign Bands (Post-processing)

```bash
python -m curriculum_extractor.scripts.assign_bands \
    --curriculum curriculum.yaml \
    --input ./metadata/ \
    --output ./metadata_with_bands/
```

### Benchmark

```bash
python -m curriculum_extractor.scripts.benchmark \
    --curriculum curriculum.yaml \
    --input ./data/ \
    --batch-size 10000 \
    --output benchmark_results.json
```

## Metrics Configuration

Create `metrics_config.yaml`:

```yaml
metrics:
  - class: DifficultyMetric
    enabled: true
    level: 0  # Lower levels run first
    
  - class: ReadabilityMetric
    enabled: true
    level: 0  # Same level can run in parallel
    
  - class: ModalityMetric
    enabled: true
    level: 1  # Runs after level 0
    
  - class: EntropyMetric
    enabled: true
    level: 1
```

## Output Schema

### Metadata Layer

```
file_name=xxx/metadata.parquet
├── uuid (string)          # Unique identifier
├── id (string)            # Original record ID
├── file_path (string)     # Source file path
├── difficulty_score (float)
├── difficulty_level (string)
├── readability_score (float)
├── modality_primary (string)
├── ... (all metrics flattened)
├── curriculum_version (string)
└── opt_metric_1-5 (any)   # Reserved for extensions
```

### Rejection Layer

```
file_name=xxx/rejections.parquet
├── uuid (string)
├── id (string)
├── file_path (string)
├── rejected_reason (string)
└── rejected_at (string)   # Metric that rejected
```

## Custom Metrics

```python
from curriculum_extractor.core.plugin import MetricPlugin, ExtractionResult, ReadOnlyRecord

class QualityFilter(MetricPlugin):
    name = "quality_filter"
    level = 0  # Run early
    
    def compute(self, record: ReadOnlyRecord) -> dict:
        text = record.get("text", "")
        return {"length": len(text), "checked": True}
    
    def extract(self, record: ReadOnlyRecord) -> ExtractionResult:
        text = record.get("text", "")
        
        if len(text) < 100:
            return ExtractionResult(
                metrics={},
                rejected=True,
                rejection_reason="Text too short (< 100 chars)",
            )
        
        return ExtractionResult(metrics=self.compute(record))
```

## Performance

With timing enabled:

```python
extractor = CurriculumExtractor(
    "curriculum.yaml",
    track_timing=True,
)

result = extractor.process_parquet("data.parquet")
print(result["timing"])
# {
#   "difficulty": {"count": 1000, "avg_ms": 0.5, ...},
#   "readability": {"count": 1000, "avg_ms": 0.3, ...},
# }
```

## API Reference

### CurriculumExtractor

| Method | Description |
|--------|-------------|
| `extract_record(record, source_file)` | Extract metadata from single record |
| `process_batch(records, source_file)` | Process batch of records in memory |
| `process_parquet(input_path)` | Process local parquet file |
| `process_parquet_s3(input_path)` | Process S3 parquet file |
| `get_timing_stats()` | Get per-metric timing statistics |

### StateManager

| Method | Description |
|--------|-------------|
| `register_files(files)` | Register files for processing |
| `mark_completed(file, rows)` | Mark file as successfully processed |
| `mark_failed(file, error)` | Mark file as failed |
| `is_completed(file)` | Check if file was already processed |
| `get_pending_files()` | Get list of unprocessed files |
| `reset()` | Reset all state |
