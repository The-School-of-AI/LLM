# Output Format Configuration Guide

## Overview

The coreset engine supports multiple output formats for selected indices via the `output_index_format` configuration. You can configure it to output results in **parquet**, **jsonl**, or **csv** formats.

## Configuration

### 1. Via YAML Config File

Edit `config/pipeline.yaml`:

```yaml
io:
  output_index_format: parquet  # Options: parquet, jsonl, csv
```

**Default Formats:**
- `parquet` - Binary columnar format (fast, compressed, recommended for large datasets)
- `jsonl` - JSON Lines format (human-readable, one JSON object per line)
- `csv` - CSV format (spreadsheet-compatible)

### 2. Via Python Code

```python
from src.core.config import PipelineConfig

config = PipelineConfig()
config.io.output_index_format = "jsonl"  # Change format

builder = CoresetBuilder(config, curriculum)
# Now indices will be saved as jsonl
```

### 3. Via Programmatic Format Override

```python
from src.io.loaders import CoresetWriter

writer = CoresetWriter("output/coresets")

# Override format at save time
index_path = writer.save_selected_indices(
    stage_name="1B",
    selected_chunks=selected_chunks,
    metadata=metadata_dict,
    format="csv"  # Force CSV even if config says parquet
)
```

## Selected Indices Output Formats

## Output Layouts (Legacy vs Streaming/Sharded)

The engine can run in two broad modes, which affect *which files* you will see on disk:

### Legacy / Single-worker (in-memory builder)

Per stage directory (example `1B`):

- `output/coresets/1B/selected_indices.{parquet|jsonl|csv}`
- `output/coresets/1B/manifest.json`

### Streaming / Batched (optionally sharded)

Per stage directory (example `1B`):

- `output/coresets/1B/selected_indices_part_shard###_batch######.parquet`
  - May fall back to `.jsonl` parts if parquet writing is unavailable.
- `output/coresets/1B/manifest_shard###.json`
- Optional (after running the merge utility): `output/coresets/1B/selected_indices.parquet`

The per-batch `selected_indices_part_...` files use the **same schema** as `selected_indices.{parquet|jsonl|csv}` (field set depends on what the builder wrote). For analytics, you can merge parquet parts into a single `selected_indices.parquet` using `tools/merge_selected_indices.py`.

Notes:
- In sharded runs, each shard writes its own `manifest_shard###.json` and part files.
- If you generate reports in sharded runs, they are written per-shard to avoid concurrent overwrite.

### Parquet Format (Default)
**File:** `output/coresets/{stage}/selected_indices.parquet`

**Advantages:**
- ✅ Most efficient: 10-50x smaller than CSV/JSONL
- ✅ Fast: Binary columnar format
- ✅ Preserves types: chunk_id, token_count, etc. preserved as native types
- ✅ Query-friendly: Can read subset of columns

**Read Example:**
```python
import pandas as pd
df = pd.read_parquet("output/coresets/1B/selected_indices.parquet")
print(df.head())
```

**Sample Columns (typical):**

- `chunk_id`, `dataset_id`, `token_count`, `domain`, `language`, `band`
- `byte_length`, `source_doc_id`, `source_url`
- `source` (when available)


### JSONL Format (Human-Readable)
**File:** `output/coresets/{stage}/selected_indices.jsonl`

**Advantages:**
- ✅ Human-readable: Easy to inspect in text editor
- ✅ Streaming: Process one line at a time
- ✅ Language-agnostic: Pure JSON
- ✅ Git-friendly: Text format for version control

**Read Example:**
```python
import json
with open("output/coresets/1B/selected_indices.jsonl") as f:
    for line in f:
        chunk = json.loads(line)
        print(chunk['chunk_id'], chunk['token_count'])
```

**Sample Output (schema-aligned):**

```json
{"chunk_id":"ch_001","dataset_id":"books","source":"books","token_count":2048,"byte_length":6463,"domain":"literature","language":"en","band":"B0","source_doc_id":"part-00000-...parquet","source_url":"s3://..."}
```

### CSV Format
**File:** `output/coresets/{stage}/selected_indices.csv`

**Advantages:**
- ✅ Excel/Sheets compatible
- ✅ Spreadsheet tools
- ✅ Lightweight

**Read Example:**
```python
import pandas as pd
df = pd.read_csv("output/coresets/1B/selected_indices.csv")
```

**Sample Output (schema-aligned):**

```csv
chunk_id,dataset_id,source,token_count,byte_length,domain,language,band,source_doc_id,source_url
ch_001,books,books,2048,2048,6463,literature,en,B0,part-00000-...parquet,s3://...
```

## Configuration Examples

### Example 1: Large-Scale Production (Use Parquet)
```yaml
# config/pipeline.yaml
io:
  output_index_format: parquet  # Most efficient for billions of chunks
```

### Example 2: Human Inspection (Use JSONL)
```yaml
# config/pipeline.yaml
io:
  output_index_format: jsonl  # Easy to inspect/debug
```

### Example 3: Analytics Team (Use CSV)
```yaml
# config/pipeline.yaml
io:
  output_index_format: csv  # Import into Excel/Sheets
```

### Example 4: Mixed Output in Code

Generate multiple formats programmatically:

```python
from src.io.loaders import CoresetWriter
from src.core.config import PipelineConfig

config = PipelineConfig()
builder = CoresetBuilder(config, curriculum)
selected, stats = builder.select_for_stage("1B")

writer = CoresetWriter(config.io.output_coreset_path)

# Save in all three formats
for fmt in ["parquet", "jsonl", "csv"]:
    writer.save_selected_indices(
        stage_name="1B",
        selected_chunks=selected,
        metadata=metadata_dict,
        format=fmt
    )
```

## What Gets Included in Output

Each row/object contains:
- **chunk_id**: Unique identifier -> maps to record id in source dataset file 
- **dataset_id**: Source dataset (to be removed-DONOTUSE)
- **token_count**: Number of tokens in chunk (canonical)
- **byte_length**: Byte size of chunk
- **domain**: Domain classification (code, math, etc.)
- **language**: Language code (en, hi, zh, etc.)
- **band**: Difficulty band (B0-B5)
- **source**: Original dataset source label when provided (often same as dataset_id)
- **source_doc_id**: Document source file name
- **source_url**: URL if available

* source_url+source_doc_id -->  Leads to the source dataset file and then use chunk_id to pull the exact record data (Raw dataset)

## Performance Comparison

| Format | File Size (1M chunks) | Read Time | Write Time | Compression |
|--------|---------------------|-----------|------------|-------------|
| Parquet | ~500 MB | 0.2s | 0.5s | 10-50x |
| JSONL | 5-10 GB | 2-5s | 3-8s | None |
| CSV | 5-10 GB | 2-5s | 3-8s | None |

## Ablation Configs

The ablation configurations also support format configuration:

```yaml
# config/ablation_high_compression.yaml
io:
  output_index_format: parquet
```

```yaml
# config/ablation_no_diversity.yaml
io:
  output_index_format: jsonl
```

## Programmatic Format Selection

Extend the CoresetWriter to support additional formats:

```python
from src.io.loaders import CoresetWriter
import pickle

class ExtendedCoresetWriter(CoresetWriter):
    def save_selected_indices(self, stage_name, selected_chunks, 
                             metadata, format="parquet"):
        """Extended to support pickle format"""
        if format.lower() == "pickle":
            stage_dir = self.output_path / stage_name
            stage_dir.mkdir(parents=True, exist_ok=True)
            output_file = stage_dir / "selected_indices.pkl"
            
            data = {
                'selected_chunks': list(selected_chunks),
                'metadata': metadata
            }
            with open(output_file, 'wb') as f:
                pickle.dump(data, f)
            return output_file
        
        # Fall back to parent for standard formats
        return super().save_selected_indices(
            stage_name, selected_chunks, metadata, format
        )
```

## Manifest Output (Always JSON)

The manifest file is **always saved as JSON** regardless of index format. The exact filename depends on mode:

- Legacy / single-worker: `output/coresets/{stage}/manifest.json`
- Streaming sharded: `output/coresets/{stage}/manifest_shard###.json`

**Sample Manifest (schema-aligned):**

```json
{
  "stage_name": "1B",
  "coreset_id": "<sha256>",
  "target_tokens": 1000000000,
  "target_tokens_global": 1000000000,
  "target_tokens_shard": 250000000,
  "actual_tokens": 123456789,
  "selected_chunks_count": 987654,
  "selected_chunks_file": "output/coresets/1B/",
  "created_at": "2026-02-13T12:34:56.789012",
  "pipeline_version": "1.0.0",
  "curriculum_version": "0.6.0",
  "seed": 42,
  "config_hash": "<hash>",
  "shard_id": 0,
  "num_shards": 4,
  "stage_target_scale": 1.0,
  "composition": {
    "band_distribution": {"B0": 0.49, "B1": 0.21, "B2": 0.15, "B3": 0.10, "B4": 0.03, "B5": 0.02},
    "domain_distribution": {
      "total": {"web": 0.50, "literature": 0.25, "math": 0.25},
      "by_band": {"B0": {"web": 1.0}}
    },
    "language_distribution": {"en": 0.92, "hi": 0.04, "bn": 0.02, "ta": 0.02}
  },
  "rolling_window_stats": {"window_tokens": 2000000, "max_band_delta": 0.03},
  "availability_stats": {"eligible_unused_tokens_total": 1234567890}
}
```

This allows you to quickly understand what was selected without parsing the potentially large indices file.

## Summary

**Quick Decision Guide:**

| Use Case | Format | Config |
|----------|--------|--------|
| Production pipeline | Parquet | `output_index_format: parquet` |
| Debugging/Inspection | JSONL | `output_index_format: jsonl` |
| Excel/Analytics | CSV | `output_index_format: csv` |
| Custom processing | Add extension | Custom class |

The format can be changed at any time via configuration - no code changes needed!
