# S3 Run (Direct Streaming `.txt` -> Contamination Check)

## What Changed

This project now supports scanning a dataset directly from S3 without downloading the input dataset file locally.

### Added / Updated

- `scripts/scan_from_s3.py`
  - Streams a `.txt` file from S3 using `boto3`
  - Converts packed Q&A text into JSONL-style records in memory
  - Runs contamination checks using the existing scanner
  - Writes reports locally to `reports/`
  - Prints a validation summary before scanning
  - Writes a parse-gap report for lines where no Q&A pair was extracted

- `core/scanner.py`
  - Added `scan_records(...)` so the scanner can run on in-memory records
  - Added semantic device config support (`semantic_device`)
  - Existing `scan.py` / file-based flow still works

- `core/detectors.py`
  - Semantic detector now auto-selects embedding device:
    - `cuda` -> `mps` -> `cpu`
  - Prints selected semantic device at startup
  - FAISS remains CPU in the current setup (`faiss-cpu`)

- Dependencies
  - Added `boto3` to `pyproject.toml` and `requirements.txt`

## How It Works (S3 Flow)

1. Read `.txt` from S3 (`s3://bucket/key.txt`) using `boto3`
2. Parse packed Q&A format (`Question? Answer.` repeated on each line)
3. Convert to in-memory JSONL-style records:
   - `{"id": "qa_1", "text": "..."}`
4. Run contamination checks:
   - N-gram
   - MinHash
   - Semantic (if installed; embeddings auto-use GPU if available)
5. Save reports locally in `reports/`

No local copy of the input dataset file is created in this flow.

## Prerequisites

- Python env created by `uv`
- AWS credentials configured (one of these):
  - `aws configure`
  - environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, etc.)
  - IAM role (if running on AWS)
- Benchmark datasets downloaded locally (`benchmarks/`)

## Install / Update Dependencies (uv)

Run from the project folder:

```bash
cd /Users/work/Desktop/T5_data/LLM/experiments/5_data_qa_and_leakage_control/collected

# Install project dependencies without installing the local package itself
uv sync --no-install-project
```

Optional semantic layer (FAISS + sentence-transformers):

```bash
# This project currently has a packaging issue for normal uv project install.
# If you need semantic support, use your existing environment strategy carefully.
# For now, core N-gram + MinHash will work with the command above.
# If semantic dependencies are installed, embeddings auto-select device:
# cuda -> mps -> cpu
```

## One-Time: Download Benchmarks

```bash
.venv/bin/python scripts/download_benchmarks.py
```

This creates `benchmarks/` locally.

## Run the S3 Streaming Scan

```bash
.venv/bin/python scripts/scan_from_s3.py \
  s3://your-bucket/path/group4.txt \
  "Team 4" \
  "group4_batch_01"
```

### Optional flags

```bash
.venv/bin/python scripts/scan_from_s3.py \
  s3://your-bucket/path/group4.txt \
  "Team 4" \
  "group4_batch_01" \
  --benchmarks-dir benchmarks \
  --reports-dir reports
```

## Validation Summary (Printed Before Scan)

The S3 wrapper prints:

- `S3 input`
- `Total lines`
- `Non-empty lines`
- `Parsed lines`
- `Zero-pair lines`
- `Q&A pairs extracted`
- `Extraction rate`
- `Parse gaps report` (if any lines failed parsing)

## Parse Gaps Report (Local)

If some lines did not match the Q&A regex, a local JSONL report is written:

- `reports/<batch_name>_PARSE_GAPS_<timestamp>.jsonl`

Each row includes:

- `s3_uri`
- `line_number`
- `reason` (`no regex match`)
- `line_preview`

Use this report to inspect formatting issues in the source `.txt`.

## Contamination Reports (Local)

The scanner still writes the normal outputs to `reports/`:

- `reports/<batch_name>_<timestamp>.json`
- `reports/<batch_name>_CONTAMINATED_<timestamp>.jsonl` (only if contamination found)
- `reports/run_registry.jsonl`

## Notes

- The input dataset is not saved locally, but records are currently held in memory after parsing.
- `scan_from_s3.py` currently expects a `.txt` S3 object in the same packed Q&A format as `group4.txt`.
- If your S3 file format changes, update the regex/parser in `scripts/scan_from_s3.py`.
- Semantic embeddings now auto-select device by default (`cuda`, else `mps`, else `cpu`).
- In the current project setup, FAISS is still CPU (`faiss-cpu`), even when semantic embeddings run on GPU/MPS.
