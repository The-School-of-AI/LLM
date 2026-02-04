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

## Output Formats Explained

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

**Sample Output:**
```
chunk_id | dataset_id | token_count | domain | language | band
---------|------------|-------------|--------|----------|------
ch_001   | ds_1       | 2048        | code   | en       | B2
ch_002   | ds_1       | 1024        | math   | en       | B3
...
```

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

**Sample Output:**
```json
{"chunk_id":"ch_001","dataset_id":"ds_1","token_count":2048,"domain":"code","language":"en","band":"B2"}
{"chunk_id":"ch_002","dataset_id":"ds_1","token_count":1024,"domain":"math","language":"en","band":"B3"}
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

**Sample Output:**
```csv
chunk_id,dataset_id,token_count,domain,language,band
ch_001,ds_1,2048,code,en,B2
ch_002,ds_1,1024,math,en,B3
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
- **chunk_id**: Unique identifier for the chunk
- **dataset_id**: Source dataset
- **token_count**: Number of tokens in chunk
- **byte_length**: Byte size of chunk
- **domain**: Domain classification (code, math, etc.)
- **language**: Language code (en, hi, zh, etc.)
- **band**: Difficulty band (B0-B5)
- **source_doc_id**: Document source
- **source_url**: URL if available

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

The manifest file is **always saved as JSON** regardless of index format:

```json
{
  "stage": "1B",
  "curriculum_version": "0.0.1",
  "total_selected": 1000000,
  "selected_tokens": 2000000000,
  "composition": {
    "band_distribution": {...},
    "domain_distribution": {...}
  },
  "selected_chunks_file": "output/coresets/1B/selected_indices.parquet",
  "statistics": {...}
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
