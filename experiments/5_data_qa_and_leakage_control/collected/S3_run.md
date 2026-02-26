# S3 Run (Direct Streaming `.txt` -> Contamination Check)

## What Changed

This project now supports scanning a dataset directly from S3 without downloading the input dataset file locally.

### Added / Updated

- `scripts/S3_run.py`
  - Reads `S3_scan.json` (and optional `S3_aws.json`)
  - Auto-downloads benchmarks if missing (configurable)
  - Auto-installs semantic runtime dependencies on first run
    - CUDA GPU host: tries `faiss-gpu` + `sentence-transformers`
    - Fallback/no-GPU: uses `faiss-cpu` + `sentence-transformers`
  - Runs the S3 scan wrapper with one command
  - Supports CPU-only override (`--cpu` or `S3_FORCE_CPU=1`)

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
  - FAISS backend auto-selects:
    - GPU FAISS (`gpu:0`) on CUDA hosts if GPU FAISS is installed
    - CPU FAISS fallback otherwise

- Dependencies
  - Added `boto3` to `pyproject.toml` and `requirements.txt`

- `S3_scan.json`
  - Main config file for S3 scan runs (S3 path, team, batch, AWS region/profile, output dirs)

- `S3_aws.json`
  - Optional local AWS credentials/settings file for one-command local runs
  - Keep this local-only (do not commit real credentials)

## How It Works (S3 Flow)

1. Read `.txt` from S3 (`s3://bucket/key.txt`) using `boto3`
2. Parse packed Q&A format (`Question? Answer.` repeated on each line)
3. Convert to in-memory JSONL-style records:
   - `{"id": "qa_1", "text": "..."}`
4. Run contamination checks:
   - N-gram
   - MinHash
   - Semantic (installed automatically by `S3_run.py`; embeddings/FAISS use GPU when available)
5. Save reports locally in `reports/`

No local copy of the input dataset file is created in this flow.

## Prerequisites

- Python env created by `uv`
- AWS credentials configured (one of these):
  - `aws configure`
  - environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, etc.)
  - IAM role (if running on AWS)
- Benchmark datasets downloaded locally (`benchmarks/`)

## Config Files (Short S3 Names)

### `S3_scan.json` (main config)

Stores non-secret run settings:

- `s3_uri`
- `team_name`
- `batch_name`
- `aws_region`
- `aws_profile`
- `benchmarks_dir`
- `reports_dir`
- `auto_download_benchmarks`

### `S3_aws.json` (optional local credentials)

If present, `scripts/S3_run.py` loads it and sets AWS env vars for the run.

Recommended:
- Use `~/.aws/credentials` or environment variables for real credentials
- Keep `S3_aws.json` local only if you use it

## Install / Update Dependencies (uv)

Run from the project folder:

```bash
cd /Users/work/Desktop/T5_data/LLM/experiments/5_data_qa_and_leakage_control/collected

# Install project dependencies without installing the local package itself
uv sync --no-install-project
```

Optional semantic layer (FAISS + sentence-transformers):

```bash
# You do not need a separate semantic install step when using scripts/S3_run.py.
# It auto-installs semantic runtime dependencies on first run.
# GPU hosts: tries faiss-gpu + sentence-transformers
# CPU hosts / fallback: uses faiss-cpu + sentence-transformers
```

## One-Time: Download Benchmarks

```bash
.venv/bin/python scripts/download_benchmarks.py
```

This creates `benchmarks/` locally.

If you use `scripts/S3_run.py`, benchmark download can happen automatically
when `auto_download_benchmarks` is `true` and `benchmarks/` is missing.

## One-Command Run (Recommended)

1. Update `S3_scan.json` with your S3 path/team/batch
2. Optionally fill `S3_aws.json` (or use `aws configure`)
3. Run:

```bash
.venv/bin/python scripts/S3_run.py
```

This will:
- load config
- optionally load AWS settings
- auto-download benchmarks if missing
- auto-install semantic runtime dependencies (GPU/CPU-aware)
- auto-detect semantic device / FAISS backend
- run the S3 streaming contamination scan

### CPU-Only Override (Optional)

If you want to force CPU mode even on a GPU machine:

```bash
.venv/bin/python scripts/S3_run.py --cpu
```

or

```bash
S3_FORCE_CPU=1 .venv/bin/python scripts/S3_run.py
```

This forces semantic embeddings to CPU and uses CPU FAISS.

## Run the S3 Streaming Scan

Direct CLI mode (without config runner):

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
- `scripts/S3_run.py` auto-installs semantic dependencies on first run, so the first run may take longer.
- On CUDA GPU hosts, `scripts/S3_run.py` tries `faiss-gpu`; if install/init fails, the scan falls back to CPU FAISS automatically.
- `scripts/S3_run.py` uses the current Python interpreter (`.venv/bin/python` if you launch it that way).
