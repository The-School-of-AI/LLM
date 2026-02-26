# LLM Contamination Scanner

Three-layer contamination detection pipeline for LLM training datasets.

## How It Works

| Layer | Method | Catches | Severity | Threshold |
|---|---|---|---|---|
| 1 | N-gram (13-word) | Exact / copy-paste matches | CRITICAL | 100% |
| 2 | MinHash (word bigrams) | Near-identical wording | HIGH | ≥ 80% Jaccard |
| 3 | Semantic (MiniLM + FAISS) | Paraphrased / reworded | MEDIUM | ≥ 90% cosine |

Each layer only flags samples not already caught by a stricter layer above it.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Install it first if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install all dependencies:

```bash
cd experiments/5_data_qa_and_leakage_control/collected
uv sync
```

## Download Benchmarks (one-time)

Downloads all 14 benchmark test sets into `benchmarks/`. Only needed once — skipped automatically on subsequent runs if the files already exist.

```bash
uv run python scripts/download_benchmarks.py
```

### Benchmarks included

| Benchmark | Samples | Best for |
|---|---|---|
| MMLU | ~14k | General knowledge (57 domains) |
| MMLU-Pro | ~12k | General knowledge (harder) |
| TriviaQA | ~11k | Factual / trivia |
| TruthfulQA | ~817 | Factual accuracy |
| ARC-Challenge | ~1.2k | Science / school knowledge |
| BoolQ | ~3.3k | Yes/no factual |
| HellaSwag | ~10k | Commonsense |
| Winogrande | ~1.3k | Commonsense reasoning |
| GSM8K | ~1.3k | Math word problems |
| MATH | ~5k | Advanced mathematics |
| HumanEval | 164 | Code generation |
| PIQA | ~1.8k | Physical reasoning |
| IFEval | 541 | Instruction following |
| BBH | ~6.5k | Mixed reasoning (27 tasks) |

## Scanning a Local File

```bash
uv run python scripts/scan.py <input_file.jsonl> <team_name> <batch_name>
```

Example:

```bash
uv run python scripts/scan.py group4.jsonl "Team 4" "group4_batch_01"
```

### Input Format

JSONL with a `text` field on each line:

```jsonl
{"id": "1", "text": "Sample training text here"}
{"id": "2", "text": "Another sample"}
```

## Scanning from S3

Edit `config.json` with your S3 URI and team details, then run:

```bash
uv run python scripts/run.py
```

This will:
1. Auto-download benchmarks if missing
2. Stream the `.txt` file from S3
3. Parse Q&A pairs, run all three detection layers
4. Write reports to `reports/`

For AWS credentials, fill in `aws.json` (never commit this file — it is gitignored):

```json
{
  "access_key_id": "...",
  "secret_access_key": "...",
  "session_token": "",
  "region": "us-east-1",
  "profile": ""
}
```

You can also run the S3 scanner directly:

```bash
uv run python scripts/scan_from_s3.py s3://bucket/path/file.txt "Team 4" "group4_batch_01"
```

## Output

- `reports/<batch_name>_<timestamp>.json` — full report with findings per layer
- `reports/<batch_name>_CONTAMINATED_<timestamp>.jsonl` — one line per flagged sample
- `reports/run_registry.jsonl` — permanent audit log of every run

Each contaminated sample shows which layer caught it, which benchmark it matched, and the actual similarity score.

Exit codes:
- `0` = APPROVED (no contamination found)
- `1` = REJECTED (contamination found)

## Run Registry (Audit Trail)

Every scan writes at least two lines to `reports/run_registry.jsonl` — a `STARTED` entry before detection begins and a `COMPLETED` (or `FAILED`) entry at the end.

```jsonl
{"run_id": "a3f2...", "status": "STARTED", "team": "Team 4", "dataset": "batch_01", ...}
{"run_id": "a3f2...", "status": "COMPLETED", "result": "APPROVED", "total_samples": 10000, ...}
```

## Replaying a Past Run

```bash
# Show metadata for a given run_id
uv run python scripts/replay.py <run_id>

# Re-execute the scan
uv run python scripts/replay.py <run_id> --execute
```

## Programmatic Usage

```python
from core import ContaminationScanner

scanner = ContaminationScanner({
    "benchmarks_path": "benchmarks",
    "reports_path": "reports",
    "ngram_size": 13,
    "minhash_threshold": 0.8,
    "minhash_permutations": 128,
    "semantic_threshold": 0.9,
    "semantic_model": "all-MiniLM-L6-v2",
    "semantic_batch_size": 512,
    "report_sample_limit": 50,
})

approved, report = scanner.scan_dataset("data.jsonl", "team-a", "batch-01")
```
