# Data QA Contamination Scanner

Three-layer contamination detection pipeline for LLM training datasets.

## How It Works

| Layer | Method | Catches | Severity | Threshold |
|---|---|---|---|---|
| 1 | N-gram (13-word) | Exact / copy-paste matches | CRITICAL | 100% |
| 2 | MinHash (word bigrams) | Near-identical wording | HIGH | ≥ 80% Jaccard |
| 3 | Semantic (MiniLM + FAISS) | Paraphrased / reworded | MEDIUM | ≥ 90% cosine |

Each layer only flags samples not already caught by a stricter layer above it.
Confidence values in results are real computed scores, not hardcoded labels.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
cd experiments/5_data_qa_and_leakage_control/collected

# Install core layers (N-gram + MinHash)
uv sync

# Also enable the Semantic layer (dense embeddings + FAISS — ~2 GB download)
uv sync --extra semantic
```

> **uv not installed?** Run `curl -LsSf https://astral.sh/uv/install.sh | sh` first.

## Download Benchmarks (One-Time)

Downloads all benchmark test sets into `benchmarks/`. Only needed if `benchmarks/*_test.jsonl` files are missing.

```bash
uv run python scripts/download_benchmarks.py

# Optional: write to a custom location
uv run python scripts/download_benchmarks.py --output-dir /data/benchmarks
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

> **Note:** If your dataset is general knowledge / facts, prioritise: TriviaQA, MMLU, MMLU-Pro, TruthfulQA, ARC.

## Run a Scan

```bash
uv run python scripts/scan.py <input_file.jsonl> <team_name> <batch_name>
```

Example:

```bash
uv run python scripts/scan.py group4.jsonl "Team 4" "group4_batch_01"
```

## Input Format

JSONL with a `text` field on each row:

```jsonl
{"id": "1", "text": "Sample training text here"}
{"id": "2", "text": "Another sample"}
```

## Output

- `reports/<batch_name>_<timestamp>.json` — full report with findings per layer
- `reports/<batch_name>_CONTAMINATED_<timestamp>.jsonl` — one line per flagged sample
- `reports/run_registry.jsonl` — permanent audit log of every run (see below)

Each contaminated sample shows which layer caught it, which benchmark it matched, what it matched against, and the actual similarity score.

Exit code:
- `0` = APPROVED (no contamination found)
- `1` = REJECTED (contamination found)

## Run Registry (Audit Trail)

Every scan writes at least two lines to `reports/run_registry.jsonl` — a `STARTED` entry before detection begins and a `COMPLETED` (or `FAILED`) entry at the end. This ensures every run is permanently recorded even if the process crashes.

```jsonl
{"run_id": "a3f2...", "status": "STARTED",    "team": "Team 4", "dataset": "batch_01", "scanner_commit": "86e95c...", "repo_dirty": false, "input_file": "/data/batch_01.jsonl", "config": {...}}
{"run_id": "a3f2...", "status": "COMPLETED",  "result": "APPROVED", "total_samples": 10000, "contaminated_count": 0}
```

Failed runs are recorded with a `failure_type` field (`INVALID_INPUT`, `OUT_OF_MEMORY`, or `UNEXPECTED_ERROR`).

## Replaying a Past Run

```bash
# Show what was used for a given run_id
uv run python scripts/replay.py <run_id>

# Also re-execute the scan
uv run python scripts/replay.py <run_id> --execute
```

This prints the exact git commit, input file, team, config, and outcome — and the `git checkout` + `scan.py` command to reproduce it.

## Configuration

Pass a config dict when using the scanner programmatically:

```python
from core.scanner import ContaminationScanner

scanner = ContaminationScanner({
    "benchmarks_path": "benchmarks",      # directory with *_test.jsonl files
    "reports_path": "reports",            # directory where reports are written
    "ngram_size": 13,                     # words per n-gram (default: 13)
    "minhash_threshold": 0.8,             # Jaccard threshold (default: 0.8)
    "minhash_permutations": 128,          # MinHash accuracy (default: 128)
    "semantic_threshold": 0.9,            # cosine threshold (default: 0.9)
    "semantic_model": "all-MiniLM-L6-v2", # embedding model
    "semantic_batch_size": 512,           # samples per batch
    "report_sample_limit": 50,            # max samples shown per layer in report
})
```

The semantic layer is automatically disabled (with a warning) if `faiss-cpu` and `sentence-transformers` are not installed.
