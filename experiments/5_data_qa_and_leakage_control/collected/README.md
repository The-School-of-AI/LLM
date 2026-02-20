# Data QA Contamination Scanner

This is the runnable scanner package.

## Quick Start

```bash
cd /home/ubuntu/LLM/experiments/5_data_qa_and_leakage_control/collected
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `venv` is unavailable on your Ubuntu machine:

```bash
python3 -m pip install --user -r requirements.txt
```

## Download Benchmarks (One-Time)

Run this only if `benchmarks/*_test.jsonl` files are missing.

```bash
python scripts/download_benchmarks.py
cp benchmark_registry/*_test.jsonl benchmarks/
ls benchmarks/*_test.jsonl
```

## Run a Scan

```bash
python scripts/scan.py <input_file.jsonl> <team_name> <batch_name>
```

Example:

```bash
python scripts/scan.py group4.jsonl "Team 4" "group4_batch_01"
```

## Input Format

JSONL with a `text` field on each row:

```jsonl
{"id":"1","text":"Sample training text here"}
{"id":"2","text":"Another sample"}
```

## Output

- `reports/<batch_name>_<timestamp>.json`
- `reports/<batch_name>_CONTAMINATED_<timestamp>.jsonl` (only if contamination is found)

Exit code:
- `0` = approved
- `1` = rejected
