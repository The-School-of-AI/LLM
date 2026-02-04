# curriculum_reader

Read and analyze curriculum metadata layers for LLM training data preparation.

## Overview

`curriculum_reader` provides tools for:

- **Reading metadata layers** from S3 or local storage
- **Deterministic batch creation** for reproducible training
- **Dataset analysis** and statistics

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Reading Metadata

```python
from curriculum_reader import MetadataReader

reader = MetadataReader("./metadata")

# Basic operations
print(f"Total records: {reader.count_rows()}")
print(f"Columns: {reader.get_column_names()}")

# Read all data
table = reader.read_all()

# Read specific columns
subset = reader.read_all(columns=["uuid", "id", "difficulty_score"])

# Random sample
sample = reader.sample(n=100, seed=42)

# Find specific record
record = reader.get_record_by_id("record-123")
```

### Reading Rejections

```python
from curriculum_reader import RejectionReader

reader = RejectionReader("./rejections")

# Get rejection statistics
table = reader.read_all()
rejections = table.to_pandas()

# Group by rejection reason
print(rejections.groupby("rejected_reason").size())

# Group by rejecting metric
print(rejections.groupby("rejected_at").size())
```

### Deterministic Batch Creation

```python
from curriculum_reader import MetadataReader, BatchCreator, BatchConfig

reader = MetadataReader("./metadata")

config = BatchConfig(
    batch_size=1000,
    seed=42,  # For reproducibility
)

creator = BatchCreator(reader, config, state_path="./batch_state")

# Get specific batch (always returns same records for same seed)
batch_0 = creator.get_batch(0)
batch_1 = creator.get_batch(1)

# Auto-increment mode
batch = creator.get_batch()  # Returns batch 0, advances to 1
batch = creator.get_batch()  # Returns batch 1, advances to 2

# Check current position
print(f"Current batch: {creator.get_current_batch_number()}")
print(f"Total batches: {creator.get_total_batches()}")

# Reset to beginning
creator.reset()

# Seek to specific batch
creator.seek(50)
```

### Iterate Through Batches

```python
# Iterate through range of batches
for batch_num, batch in creator.iter_batches(start_batch=0, end_batch=10):
    print(f"Batch {batch_num}: {len(batch)} records")
    # Process batch...

# Include batch metadata in records
for batch_num, batch in creator.iter_batches(include_batch_info=True):
    # Each record has _batch_number and _batch_seed columns
    pass
```

### Stratified Sampling

```python
from curriculum_reader import MetadataReader, StratifiedBatchCreator, BatchConfig

reader = MetadataReader("./metadata")
config = BatchConfig(batch_size=1000, seed=42)

# Sample proportionally from each band
creator = StratifiedBatchCreator(
    reader,
    config,
    stratify_column="band_id",
)

batch = creator.get_batch(0)  # Maintains band distribution
```

## CLI Usage

### Analyze Metadata

```bash
python -m curriculum_reader.scripts.analyze \
    --path ./metadata/ \
    --summary \
    --band-distribution \
    --export summary.json
```

### Get Deterministic Batch

```bash
python -m curriculum_reader.scripts.get_batch \
    --path ./metadata/ \
    --batch-number 0 \
    --batch-size 1000 \
    --seed 42 \
    --output batch_0.parquet
```

## Deterministic Ordering

The batch creator uses xxhash for deterministic ordering:

1. Each record gets a hash based on its ID and the seed
2. Records are sorted by hash value
3. Batches are created from the sorted order

This ensures:
- **Same seed → Same order** every time
- **Different seeds → Different orders** for augmentation
- **Record-level determinism** - each record's position is fixed for a given seed
- **Batch-level determinism** - batch N always contains the same records

```python
# Reproducibility example
config = BatchConfig(batch_size=100, seed=42)

creator1 = BatchCreator(reader, config)
creator2 = BatchCreator(reader, config)

batch1 = creator1.get_batch(0)
batch2 = creator2.get_batch(0)

# These are identical
assert batch1.column("uuid").to_pylist() == batch2.column("uuid").to_pylist()
```

## S3 Support

```python
import s3fs
from curriculum_reader import MetadataReader

fs = s3fs.S3FileSystem()
reader = MetadataReader("s3://bucket/metadata", filesystem=fs)

# All operations work the same
print(reader.count_rows())
sample = reader.sample(n=100)
```

## Analytics Examples

### Band Distribution

```python
from curriculum_reader import MetadataReader, MetadataAnalyzer

reader = MetadataReader("./metadata")
analyzer = MetadataAnalyzer(reader)

# Get overall summary
summary = analyzer.get_summary()
print(f"Total records: {summary.total_records}")
print(f"Unique sources: {summary.unique_sources}")

# Get band distribution
band_dist = analyzer.get_band_distribution()
for band, count in band_dist.items():
    pct = count / summary.total_records * 100
    print(f"{band}: {count:,} ({pct:.1f}%)")
```

### Compare Subsets

```python
# Compare distributions across sources
comparison = analyzer.compare_by_column("source")
for source, stats in comparison.items():
    print(f"{source}: {stats}")
```

### Export Report

```python
# Export full analysis
analyzer.export_summary_report("analysis_report.json")
```

## API Reference

### MetadataReader

| Method | Description |
|--------|-------------|
| `count_rows()` | Total number of records |
| `get_schema()` | PyArrow schema |
| `get_column_names()` | List of column names |
| `read_all(columns)` | Read all data (optional column filter) |
| `sample(n, seed)` | Random sample of n records |
| `get_record_by_id(id)` | Find specific record |

### BatchCreator

| Method | Description |
|--------|-------------|
| `get_batch(batch_num)` | Get specific batch |
| `get_batch()` | Get next batch (auto-increment) |
| `iter_batches(start, end)` | Iterate through batches |
| `get_current_batch_number()` | Current position |
| `get_total_batches()` | Total number of batches |
| `seek(batch_num)` | Move to specific position |
| `reset()` | Reset to beginning |

### BatchConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | int | 1000 | Records per batch |
| `seed` | int | 42 | Random seed for ordering |
| `shuffle` | bool | True | Whether to shuffle records |
