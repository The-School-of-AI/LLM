# Dataset Reader Documentation

A Python utility for reading downloaded dataset files (JSON and Parquet formats).

## Overview

`parquet_reader.py` provides both a Python API (`DatasetReader` class) and a CLI for reading and exploring downloaded datasets.

## Installation

No additional installation required - uses the same dependencies as the main downloader.

## CLI Usage

```bash
uv run python parquet_reader.py --dataset <dataset> --subfolder <subfolder> [options]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--dataset` | Dataset name: `dolma`, `sangraha`, `indiccorp`, `ncert` |
| `--subfolder` | Subfolder name (see table below) |

### Subfolder Values by Dataset

| Dataset | Subfolder Examples |
|---------|-------------------|
| `dolma` | `gutenberg`, `wiki`, `reddit`, `c4`, etc. (source name) |
| `sangraha` | `hin`, `mar`, `ben`, `tam`, etc. (language code) |
| `indiccorp` | `hi`, `ta`, `bn`, `mr`, etc. (language code) |
| `ncert` | `_all` (always use `_all`) |

### Optional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--format` | Output format: `json` or `parquet` | `parquet` |
| `--limit` | Limit number of records to display | `5` |
| `--sample` | Show N random sample records | - |
| `--schema` | Show schema information | - |
| `--stats` | Show statistics only | - |
| `--export-json` | Export to JSON file | - |
| `--export-jsonl` | Export to JSONL file | - |

## CLI Examples

### View Sample Records

```bash
# Dolma parquet - 3 samples
uv run python parquet_reader.py --dataset dolma --subfolder gutenberg --format parquet --sample 3

# Sangraha JSON - 5 samples
uv run python parquet_reader.py --dataset sangraha --subfolder hin --format json --sample 5

# IndicCorp - default 5 records
uv run python parquet_reader.py --dataset indiccorp --subfolder hi --format parquet
```

### View Schema

```bash
uv run python parquet_reader.py --dataset dolma --subfolder gutenberg --format parquet --schema
```

Output:
```
Schema:
  id: large_string
  dataset: large_string
  domain: large_string
  source: large_string
  text: large_string
  language: large_string
  metadata: large_string
  added: large_string
  created: large_string
  version: large_string

Rows in first file: 10
```

### View Statistics

```bash
uv run python parquet_reader.py --dataset dolma --subfolder gutenberg --format parquet --stats
```

Output:
```
Statistics:
  files: 1
  records: 10
  size_bytes: 1380925
  size_mb: 1.32
```

### Export Data

```bash
# Export to JSON (array format)
uv run python parquet_reader.py --dataset dolma --subfolder gutenberg --format parquet --export-json output.json

# Export to JSONL (one record per line)
uv run python parquet_reader.py --dataset dolma --subfolder gutenberg --format parquet --export-jsonl output.jsonl

# Export with limit
uv run python parquet_reader.py --dataset sangraha --subfolder hin --format parquet --export-jsonl sample.jsonl --limit 100
```

### Limit Records

```bash
# Show only first 10 records
uv run python parquet_reader.py --dataset sangraha --subfolder hin --format json --limit 10
```

## Python API Usage

### Import and Initialize

```python
from parquet_reader import DatasetReader

# Initialize reader for dolma parquet
reader = DatasetReader(
    dataset="dolma",
    subfolder="gutenberg",
    output_format="parquet"  # or "json"
)
```

### Get Dataset Info

```python
# String representation
print(reader)
# Output: DatasetReader(dataset='dolma', subfolder='gutenberg', format='parquet', files=1, records=10)

# Get display name
print(reader.name)
# Output: Dolma v1.7

# Get statistics
stats = reader.get_stats()
# {'files': 1, 'records': 10, 'size_bytes': 1380925, 'size_mb': 1.32}

# Get schema
schema = reader.get_schema()
# {'schema': {'id': 'large_string', ...}, 'rows': 10}

# List files
files = reader.get_files()
# [PosixPath('.data/dolma/gutenberg/parquet/records_00000.parquet')]
```

### Read Data

```python
# Read all data as DataFrame
df = reader.read_all()

# Read with limit
df = reader.read(limit=100)

# Sample random records
sample_df = reader.sample(5)
```

### Iterate Records (Memory-Efficient)

```python
# Iterate all records
for record in reader.iter_records():
    print(record["id"], record["text"][:100])

# Iterate with limit
for record in reader.iter_records(limit=10):
    process(record)
```

### Export Data

```python
# Export to JSON file
reader.to_json("output.json", limit=100)

# Export to JSONL file
reader.to_jsonl("output.jsonl", limit=100)

# Get as JSON string
json_str = reader.to_json(limit=10)

# Get as JSONL string
jsonl_str = reader.to_jsonl(limit=10)
```

## Record Schema

Each record contains the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique record identifier |
| `hash` | string | SHA-256 hash of text content (for deduplication) |
| `dataset` | string | Source dataset name |
| `domain` | string | Content domain (web, literature, education) |
| `source` | string/null | Source identifier (for Dolma) |
| `text` | string | Main text content |
| `language` | string | Full language name |
| `metadata` | dict | Additional dataset-specific fields |
| `added` | string/null | ISO timestamp when added |
| `created` | string/null | ISO timestamp of creation |
| `version` | string/null | Dataset version |

## File Locations

Data files are stored in the `.data/` directory:

```
.data/
├── dolma/
│   └── gutenberg/
│       ├── json/
│       │   └── records_00000.json
│       └── parquet/
│           └── records_00000.parquet
├── sangraha/
│   └── hin/
│       ├── json/
│       └── parquet/
├── indiccorp/
│   └── hi/
│       ├── json/
│       └── parquet/
└── ncert/
    └── _all/
        ├── json/
        └── parquet/
```

## Error Handling

If no data files are found, the reader will display a helpful message:

```
No parquet files found at: .data/dolma/gutenberg/parquet
Run: uv run python main.py --dataset dolma --scope test --format parquet
```

## Complete Example

```python
from parquet_reader import DatasetReader

# Read Sangraha Hindi parquet files
reader = DatasetReader("sangraha", "hin", "parquet")

# Check if data exists
if reader.get_files():
    print(f"Found {reader.get_stats()['records']} records")
    
    # Get sample
    for idx, row in reader.sample(3).iterrows():
        print(f"ID: {row['id']}")
        print(f"Language: {row['language']}")
        print(f"Text preview: {row['text'][:100]}...")
        print()
    
    # Export first 1000 records to JSONL
    reader.to_jsonl("sangraha_sample.jsonl", limit=1000)
else:
    print("No data found - run downloader first")
```
