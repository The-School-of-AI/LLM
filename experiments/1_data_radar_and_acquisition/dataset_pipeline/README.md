# Dataset Downloader

A Python tool for downloading and standardizing records from multiple NLP datasets with streaming support, resume capability, and chunked file output for TB-scale data.

## Features

- **4 Supported Datasets**: Dolma v1.7, Sangraha, IndicCorp v2, NCERT
- **Streaming Downloads**: Memory-efficient streaming from HuggingFace Hub
- **Resume Support**: Automatically resume interrupted downloads
- **Chunked Output**: Split large datasets into manageable file chunks
- **Auto Chunk Sizing**: Smart defaults based on download scope
- **Multiple Output Formats**: JSON (human-readable) or Parquet (columnar storage)
- **Multiple Storage Options**: Local file system or AWS S3 (with auto bucket creation)
- **Standardized Schema**: Consistent field order across all datasets
- **UTF-8 Encoding**: Proper handling of multilingual text

## Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Quick Start

```bash
# Test download (10 records)
uv run python main.py --dataset sangraha --scope test

# Validation run (10K records)
uv run python main.py --dataset dolma --scope validate

# Production download with Parquet
uv run python main.py --dataset sangraha --scope production --format parquet

# Save directly to S3
uv run python main.py --dataset sangraha --scope production --storage s3 --s3-bucket my-bucket
```

## Usage

```bash
uv run python main.py --dataset <dataset> --scope <scope> [options]
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset: `dolma`, `sangraha`, `indiccorp`, `ncert` | Required |
| `--scope` | Download scope: `test` (10), `validate` (10K), `pre-prod` (100K), `production` (full) | `test` |
| `--resume` | Resume from previous progress if interrupted | `false` |
| `--chunk-size` | Records per output file | Auto by scope |
| `--format` | Output format: `json` or `parquet` | `json` |
| `--storage` | Storage: `local` or `s3` | `local` |
| `--s3-bucket` | S3 bucket name (required if --storage=s3) | - |
| `--s3-region` | AWS region for S3 bucket | `us-east-1` |
| `--s3-prefix` | S3 key prefix/folder | `datasets` |
| `--lang` | Language for sangraha (`hin`, `mar`, etc.) or indiccorp (`hi`, `ta`, etc.) | `hin`/`hi` |

### Default Chunk Sizes

Chunk size is automatically selected based on scope:

| Scope | Records | Chunk Size |
|-------|---------|------------|
| `test` | 10 | 10 |
| `validate` | 10,000 | 1,000 |
| `pre-prod` | 100,000 | 5,000 |
| `production` | full | 10,000 |

## Datasets

### Dolma v1.7
English web corpus from Allen AI with multiple sources (gutenberg, wiki, reddit, etc.). Records are organized by source.

```bash
uv run python main.py --dataset dolma --scope validate
```

### Sangraha
Indian language web corpus from AI4Bharat supporting 23 languages.

**Available languages**: `asm`, `ben`, `bod`, `brx`, `doi`, `gom`, `guj`, `hin`, `kan`, `kas`, `mai`, `mal`, `mar`, `mni`, `nep`, `ori`, `pan`, `san`, `sat`, `snd`, `tam`, `tel`, `urd`

```bash
# Hindi (default - no --lang needed)
uv run python main.py --dataset sangraha --scope validate

# Other languages
uv run python main.py --dataset sangraha --scope validate --lang mar
uv run python main.py --dataset sangraha --scope validate --lang ben
```

### IndicCorp v2
Monolingual corpora for 24 Indian languages from AI4Bharat.

**Available languages**: `as`, `bd`, `bn`, `dg`, `en`, `gom`, `gu`, `hi`, `kha`, `kn`, `ks`, `mai`, `ml`, `mni`, `mr`, `ne`, `or`, `pa`, `sa`, `sat`, `sd`, `ta`, `te`, `ur`

```bash
# Hindi (default - no --lang needed)
uv run python main.py --dataset indiccorp --scope validate

# Other languages
uv run python main.py --dataset indiccorp --scope validate --lang ta
uv run python main.py --dataset indiccorp --scope validate --lang bn
```

### NCERT
Educational content from Indian NCERT textbooks covering multiple subjects (physics, chemistry, biology, mathematics, history, geography, etc.).

```bash
uv run python main.py --dataset ncert --scope validate
```

## Output Formats

### JSON (Default)
- Human-readable, easy to inspect
- Compatible with most tools
- Larger file sizes

### Parquet
- Columnar storage optimized for analytics
- 2-10x smaller with Snappy compression
- Native support in Pandas, Spark, DuckDB
- **Recommended for production**

**Parquet Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | `string` | Unique record identifier |
| `hash` | `string` | SHA-256 hash of text content |
| `dataset` | `string` | Source dataset name |
| `domain` | `string` | Content domain (web, literature, education) |
| `source` | `string` (nullable) | Source identifier (for Dolma) |
| `text` | `string` | Main text content |
| `language` | `string` | Full language name |
| `metadata` | `string` (JSON) | Additional fields as JSON string |
| `added` | `string` (nullable) | ISO timestamp when added |
| `created` | `string` (nullable) | ISO timestamp of creation |
| `version` | `string` (nullable) | Dataset version |

> **Note:** The `metadata` column stores a JSON string for Parquet compatibility. Use `DatasetReader` to automatically parse it back to a dict.

```bash
# Download as Parquet
uv run python main.py --dataset sangraha --scope production --format parquet
```

## Reading Downloaded Data

### Using DatasetReader (Recommended)

```python
from reader import DatasetReader

# Initialize reader
reader = DatasetReader(
    dataset="sangraha",
    subfolder="hin",
    output_format="parquet"  # or "json"
)

# Get dataset info
print(reader)  # DatasetReader(dataset='sangraha', subfolder='hin', format='parquet', files=1, records=10)
print(reader.get_stats())  # {'files': 1, 'records': 10, 'size_bytes': 12345, 'size_mb': 0.01}
print(reader.get_schema())  # {'columns': [...], 'schema': {...}, 'num_rows': 10}

# Read all data
df = reader.read_all()

# Sample records
sample = reader.sample(5)

# Memory-efficient iteration
for record in reader.iter_records():
    print(record["text"][:100])
```

### Using Pandas Directly

```python
import pandas as pd

# Read single Parquet file
df = pd.read_parquet('.data/sangraha/hin/parquet/records_00000.parquet')

# Read all Parquet files in folder
df = pd.read_parquet('.data/sangraha/hin/parquet/')

# Parse metadata JSON
import json
df['metadata'] = df['metadata'].apply(json.loads)
```

### Using DuckDB

```sql
-- Query with DuckDB
duckdb -c "SELECT language, COUNT(*) FROM '.data/sangraha/hin/parquet/*.parquet' GROUP BY language"

-- Full text search
duckdb -c "SELECT id, text FROM '.data/sangraha/hin/parquet/*.parquet' WHERE text LIKE '%keyword%' LIMIT 10"
```

## Storage Options

### Local (Default)

Files saved to `.data/` directory with format-specific subfolders:

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
│       │   └── records_00000.json
│       └── parquet/
│           └── records_00000.parquet
├── indiccorp/
│   └── hi/
│       ├── json/
│       └── parquet/
└── ncert/
    └── _all/
        ├── json/
        └── parquet/
```

> **Note:** JSON and Parquet formats are stored in separate folders, allowing both formats to coexist. Each format also maintains its own `.progress.json` for independent resume tracking.

### AWS S3

Upload directly to S3 with automatic bucket creation and versioning.

```bash
# Basic S3 upload
uv run python main.py --dataset sangraha --scope production --storage s3 --s3-bucket my-bucket

# S3 with custom region and prefix
uv run python main.py --dataset dolma --scope validate --storage s3 \
    --s3-bucket my-bucket --s3-region eu-west-1 --s3-prefix raw-data

# S3 with Parquet (recommended)
uv run python main.py --dataset sangraha --scope production --storage s3 \
    --s3-bucket my-bucket --format parquet
```

**S3 Structure:**
```
s3://my-bucket/
└── datasets/              # --s3-prefix
    ├── dolma/
    │   └── gutenberg/
    │       ├── json/
    │       └── parquet/
    ├── sangraha/
    │   └── hin/
    │       ├── json/
    │       └── parquet/
    └── ncert/
        └── _all/
            ├── json/
            └── parquet/
```

**Prerequisites:**
- AWS credentials via environment variables, `~/.aws/credentials`, or IAM role
- IAM permissions: `s3:CreateBucket`, `s3:PutBucketVersioning`, `s3:PutObject`, `s3:GetObject`, `s3:DeleteObject`, `s3:ListBucket`

## Resume Functionality

Downloads can be resumed after interruption:

```bash
# Start download (Hindi by default)
uv run python main.py --dataset sangraha --scope production

# If interrupted (Ctrl+C), resume with:
uv run python main.py --dataset sangraha --scope production --resume
```

Progress tracked in `.progress.json` files. On resume:
- Already downloaded records are skipped
- Chunks continue from correct index (no overwrites)
- Progress cleared on successful completion

Works with both local and S3 storage.

## Record Schema

All datasets use a standardized schema:

```json
{
  "id": "unique_identifier",
  "hash": "sha256_hash_of_text",
  "dataset": "Sangraha",
  "domain": "web",
  "source": null,
  "text": "The actual text content...",
  "language": "Hindi",
  "metadata": {
    "language_code": "hin",
    "type": "web"
  },
  "added": null,
  "created": null,
  "version": null
}
```

| Field | Description |
|-------|-------------|
| `id` | Unique record identifier |
| `hash` | SHA-256 hash of text content (for deduplication) |
| `dataset` | Source dataset name |
| `domain` | Content domain (web, literature, education, etc.) |
| `source` | Source identifier (for Dolma) |
| `text` | Main text content |
| `language` | Full language name (Hindi, English, etc.) |
| `metadata` | Additional dataset-specific fields |
| `added` | Timestamp when added (if available) |
| `created` | Creation timestamp (if available) |
| `version` | Dataset version (if available) |

## Chunk Size Guidelines

| Dataset | Avg Record Size | Recommended Chunk | ~File Size |
|---------|-----------------|-------------------|------------|
| Dolma | 10-50 KB | 5,000-10,000 | 50-500 MB |
| Sangraha | 5-15 KB | 10,000-20,000 | 50-300 MB |
| IndicCorp | 1-5 KB | 20,000-50,000 | 20-250 MB |
| NCERT | 2-5 KB | 10,000-20,000 | 20-100 MB |

**Tips:**
- Target 100-500 MB per file for optimal transfer/backup
- Smaller chunks = less data loss on interruption
- Override with `--chunk-size` when needed

## Examples

```bash
# Test all datasets (uses defaults)
uv run python main.py --dataset dolma --scope test
uv run python main.py --dataset sangraha --scope test
uv run python main.py --dataset indiccorp --scope test
uv run python main.py --dataset ncert --scope test

# Production with Parquet to S3
uv run python main.py --dataset sangraha --scope production \
    --format parquet --storage s3 --s3-bucket my-nlp-data

# Resume interrupted download
uv run python main.py --dataset dolma --scope production --resume

# Custom chunk size
uv run python main.py --dataset indiccorp --scope validate --chunk-size 500
```

## Dependencies

- `datasets>=4.5.0` - HuggingFace datasets library
- `pyarrow>=15.0.0` - Parquet support
- `pandas>=2.0.0` - Data manipulation
- `boto3>=1.34.0` - AWS S3 support
