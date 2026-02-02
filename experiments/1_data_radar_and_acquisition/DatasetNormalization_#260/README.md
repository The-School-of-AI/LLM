# LLM Lightning

A pipeline to **download** educational and Indic datasets from Hugging Face and **convert** NCERT Q&A data into [Dolma](https://allenai.github.io/dolma/)-harmonized formats for LLM pre-training and inspection.

---

## Features

- **Download**: Stream and save datasets (NCERT, Sangraha, IndicCorp V2, etc.) from Hugging Face to local disk and/or AWS S3, with configurable test/full mode and sharding.
- **Convert**: Transform raw NCERT JSONL shards into:
  - **Inspection JSONL** — plain, uncompressed JSONL for human review.
  - **Dolma-native GZIP** — Dolma-compliant `.jsonl.gz` under a `documents/` directory for training pipelines.

---

## Requirements

- Python 3.9+
- See [requirements.txt](requirements.txt):

| Package     | Purpose                    |
|------------|----------------------------|
| `datasets` | Hugging Face dataset load  |
| `boto3`    | S3 upload (optional)      |
| `pyyaml`   | Config parsing             |
| `duckdb`   | NCERT conversion (SQL)    |
| `pytest`   | Tests                      |

---

## Installation

```bash
git clone <repository-url>
cd LLM_Lightening
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
pip install -r requirements.txt
```

---

## Configuration

Dataset and storage settings live in **`config.yml`** (or a path you pass via `--config`).

| Section    | Key           | Description                                  |
|-----------|---------------|----------------------------------------------|
| `storage` | `mode`        | `local` \| `s3` \| `both`                    |
| `storage` | `local_dir`   | Local base path (e.g. `./data/downloaded_datasets`) |
| `aws`     | `region`      | AWS region (e.g. `us-east-1`)                |
| `aws`     | `s3_bucket`   | S3 bucket name                               |
| `aws`     | `s3_prefix`   | S3 key prefix                                |
| `mode`    | —             | `test` (limited) \| `full` (entire dataset)   |
| `datasets`| *name*        | Per-dataset config (repo, split, limits, paths) |

Example for **NCERT**:

```yaml
datasets:
  ncert:
    repo: ParthKadam2003/NCERT_Dataset
    split: train
    test_limit:
      type: rows
      value: 10000
    s3_path: ncert
    local_path: ncert
```

---

## How to Run

### 1. Download datasets

Uses Hugging Face `datasets` and (optionally) S3. Config is read from `config.yml` unless overridden.

```bash
# Download all datasets from config (respects mode in config)
python download.py

# Download only NCERT, full size, to local disk
python download.py --datasets ncert --storage local --mode full

# Download NCERT in test mode (limited rows)
python download.py --datasets ncert --storage local --mode test

# Custom config path
python download.py --config my_config.yml --datasets ncert --storage local
```

**Options:**

| Option        | Description |
|---------------|-------------|
| `--datasets`  | One or more dataset names (e.g. `ncert`, `sangraha`). Omit to run all in config. |
| `--storage`  | `local` \| `s3` \| `both` (overrides config). |
| `--mode`     | `test` \| `full` (overrides config). |
| `--config`   | Path to YAML config (default: `config.yml`). |
| `--s3-bucket`| S3 bucket (overrides config). |
| `--region`   | AWS region (overrides config). |

Output (local) is under `./data/downloaded_datasets/<dataset>/` as sharded JSONL, e.g. `part-00000.jsonl`, `part-00001.jsonl`, …

---

### 2. Convert NCERT to Dolma formats

Reads the raw NCERT JSONL shards and writes **inspection JSONL** and/or **Dolma GZIP** using a single DuckDB process.

```bash
# Both inspection + Dolma GZIP (default)
python convert_ncert.py --full ./data/downloaded_datasets/ncert/ --format both

# Inspection-only (plain JSONL for eyeballing)
python convert_ncert.py --full ./data/downloaded_datasets/ncert/ --format jsonl

# Dolma GZIP only
python convert_ncert.py --full ./data/downloaded_datasets/ncert/ --format dolma

# Single shard, custom output and language
python convert_ncert.py --part ./data/downloaded_datasets/ncert/part-00000.jsonl --output ./dolma_dataset/ncert/ --lang en
```

**Options:**

| Option     | Description |
|------------|-------------|
| `--full`   | Path to directory containing `*.jsonl` shards. |
| `--part`   | Path to a single `.jsonl` file. (Exactly one of `--full` or `--part` required.) |
| `--format` | `jsonl` \| `dolma` \| `both` (default: `both`). |
| `--output` | Output directory (default: `./dolma_dataset/ncert/`). |
| `--lang`   | Language code for metadata and ID hashing (default: `en`). |

**Output layout:**

- **Inspection**: `{output}/ncert_inspection.jsonl` (uncompressed).
- **Dolma**: `{output}/documents/ncert_harmonized.jsonl.gz` (GZIP).

---

## Output format (detailed)

### Raw downloaded NCERT (input to converter)

Each line is a JSON object with fields such as: `Topic`, `Explanation`, `Question`, `Answer`, `Difficulty`, `StudentLevel`, `QuestionType`, `subject`, `grade`, etc. Column names may vary (e.g. `subject` vs `Subject`); the converter normalizes via DuckDB `read_json_auto`.

---

### Inspection JSONL (`ncert_inspection.jsonl`)

One JSON object per line, UTF-8, uncompressed. Intended for human inspection and debugging.

**Fields:**

| Field       | Type   | Description |
|------------|--------|-------------|
| `id`       | string | Stable document ID (see [ID hash](#id-hash-algorithm) below). |
| `text`     | string | Single blob: Subject, Topic, Explanation, Question, Answer concatenated with newlines. |
| `source`  | string | Dataset label (e.g. `ncert_qa`). |
| `added`   | string | Timestamp when the record was added (ISO 8601-style). |
| `created` | string | Fixed creation date (e.g. `2024-01-01T00:00:00Z`). |
| `metadata`| object | Language, grade, difficulty, student_level, question_type, license, dataset_type. |
| `domain`  | string | Normalized subject (lowercase, spaces → underscores). |

**Example line (pretty-printed):**

```json
{
  "id": "5d6bc62e79571a39e16666601d8097ba",
  "text": "Subject: Physics\nTopic: Electric Charges and Fields\n\nExplanation:\nElectric charges and fields are fundamental concepts...\n\nQuestion:\nWhat is the phenomenon that causes a spark when you take off synthetic clothes in dry weather?\n\nAnswer:\nThe phenomenon is the buildup and sudden discharge of static electricity.",
  "source": "ncert_qa",
  "added": "2024-01-01T12:00:00Z",
  "created": "2024-01-01T00:00:00Z",
  "metadata": {
    "language": "en",
    "grade": "12",
    "difficulty": "Easy",
    "student_level": "Beginner",
    "question_type": "General",
    "license": "MIT",
    "dataset_type": "textbook_qa"
  },
  "domain": "physics"
}
```

`text` is structured as:

```
Subject: <subject>
Topic: <topic>

Explanation:
<explanation>

Question:
<question>

Answer:
<answer>
```

---

### Dolma-native GZIP (`documents/ncert_harmonized.jsonl.gz`)

Same schema as inspection JSONL (same `id`, `text`, `source`, `added`, `created`, `metadata`, `domain`), but:

- Written under `documents/` to match Dolma layout.
- Compressed with **GZIP** (one `.jsonl.gz` file).
- Suitable for Dolma-based training pipelines.

---

## ID hash algorithm

Each output record has a **stable, content-derived ID** so the same logical Q&A pair gets the same ID across runs and shards.

- **Algorithm**: **MD5** (128-bit hash).
- **Input to hash**: A single string built by **concatenating** the following (in order), with `NULL`/missing values treated as empty string:
  1. `dataset_name` (e.g. `ncert_qa`)
  2. `language` (e.g. `en`)
  3. `Subject`
  4. `Topic`
  5. `Question`
  6. `Answer`
  7. `Difficulty`
  8. `grade` (as string)

- **Implementation**: In DuckDB, this is:

  `md5(concat(dataset_name, language, coalesce(Subject,''), coalesce(Topic,''), coalesce(Question,''), coalesce(Answer,''), coalesce(Difficulty,''), cast(grade AS VARCHAR)))`

- **Encoding**: The MD5 result is emitted as a **32-character lowercase hexadecimal** string (e.g. `5d6bc62e79571a39e16666601d8097ba`).

Including content and key metadata in the hash keeps IDs stable for deduplication and reproducible pipelines while changing the ID if the Q&A or its classification changes.

---

## Project layout (after run)

```
LLM_Lightening/
├── config.yml                 # Dataset and storage config
├── download.py                # Download datasets from Hugging Face → local/S3
├── convert_ncert.py           # Convert NCERT shards → inspection + Dolma GZIP
├── requirements.txt
├── README.md
├── data/
│   └── downloaded_datasets/
│       └── ncert/
│           ├── part-00000.jsonl
│           ├── part-00001.jsonl
│           └── ...
└── dolma_dataset/
    └── ncert/
        ├── ncert_inspection.jsonl      # Inspection (when --format jsonl or both)
        └── documents/
            └── ncert_harmonized.jsonl.gz   # Dolma-native (when --format dolma or both)
```

---

## License

Metadata and code: see repository license. Converted NCERT output uses `metadata.license: "MIT"` in the generated files; upstream dataset terms still apply.

---

## Contributing

1. Fork the repo and create a branch for your change.
2. Install deps and run tests: `pytest`.
3. Follow existing style (e.g. argparse, config from YAML, DuckDB for conversion).
4. Open a **Pull Request** with a short description of the change and how you ran it (e.g. `download.py --datasets ncert --storage local --mode test` and `convert_ncert.py --full ... --format both`).
