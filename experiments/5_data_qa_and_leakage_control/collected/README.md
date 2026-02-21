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

```bash
cd /home/ubuntu/LLM/experiments/5_data_qa_and_leakage_control/collected
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download Benchmarks (One-Time)

Downloads all benchmark test sets into `benchmarks/`. Only needed if `benchmarks/*_test.jsonl` files are missing.

```bash
python scripts/download_benchmarks.py
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
python scripts/scan.py <input_file.jsonl> <team_name> <batch_name>
```

Example:

```bash
python scripts/scan.py group4.jsonl "Team 4" "group4_batch_01"
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

Each contaminated sample shows which layer caught it, which benchmark it matched, what it matched against, and the actual similarity score.

Exit code:
- `0` = APPROVED (no contamination found)
- `1` = REJECTED (contamination found)

## Configuration

Pass a config dict when using the scanner programmatically:

```python
from core.scanner import ContaminationScanner

scanner = ContaminationScanner({
    "ngram_size": 13,              # words per n-gram (default: 13)
    "minhash_threshold": 0.8,      # Jaccard threshold (default: 0.8)
    "minhash_permutations": 128,   # MinHash accuracy (default: 128)
    "semantic_threshold": 0.9,     # cosine threshold (default: 0.9)
    "semantic_model": "all-MiniLM-L6-v2",  # embedding model
    "semantic_batch_size": 512,    # samples per batch
    "benchmarks_path": "benchmarks",
})
```

The semantic layer is automatically disabled (with a warning) if `faiss-cpu` and `sentence-transformers` are not installed.
