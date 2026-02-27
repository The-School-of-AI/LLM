# LLM Contamination Scanner

Three-layer contamination detection pipeline for LLM training datasets.

## How It Works

| Layer | Method | Catches | Severity | Threshold |
|---|---|---|---|---|
| 1 | N-gram (13-word) | Exact / copy-paste matches | CRITICAL | 100% |
| 2 | MinHash (word bigrams) | Near-identical wording | HIGH | ≥ 80% Jaccard |
| 3 | Semantic (MiniLM + FAISS) | Paraphrased / reworded | MEDIUM | ≥ 90% cosine |

Each layer only flags samples not already caught by a stricter layer above it.

---

## Setup

### 1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies
```bash
cd experiments/5_data_qa_and_leakage_control/collected
uv sync
```

### 3. Download benchmarks (one-time)
```bash
uv run python scripts/download_benchmarks.py
```

Benchmarks are saved to `benchmarks/`. Downloads **26 benchmarks** (~850k+ questions, English + Indic).

**Special requirements:**
- **GPQA Diamond** (gated): accept terms at huggingface.co/datasets/Idavidrein/gpqa, then `export HF_TOKEN=<token>`
- **RULER**: not downloadable — synthetic generator, see [github.com/NVIDIA/RULER](https://github.com/NVIDIA/RULER)
- **ARC-C-IN / AIME 2026-II**: not yet available on HF Hub
- **Disk**: ~500 MB output + ~10 GB HF cache. To redirect cache: `export HF_HOME=/path/to/bigger/disk`

---

## Running a Scan

### From S3

Edit `config.json` with your S3 URI, team name, and batch name:

```json
{
  "s3_uri": "s3://your-bucket/path/file.txt",
  "team_name": "Team 4",
  "batch_name": "group4_batch_01",
  "aws_region": "us-east-1",
  "aws_profile": "",
  "benchmarks_dir": "benchmarks",
  "reports_dir": "reports",
  "auto_download_benchmarks": true,
  "enable_semantic": true
}
```

For AWS credentials, fill in `aws.json` (this file is gitignored — never commit it):

```json
{
  "access_key_id": "...",
  "secret_access_key": "...",
  "session_token": "",
  "region": "us-east-1",
  "profile": ""
}
```

Then run:

```bash
uv run python scripts/run.py
```

To run without semantic (N-gram + MinHash only), set this in `config.json` and run the same command:

```json
"enable_semantic": false
```

### From a local file

Input must be JSONL with a `text` field on each line:

```jsonl
{"id": "1", "text": "Your training sample here"}
{"id": "2", "text": "Another sample"}
```

```bash
uv run python scripts/scan.py data.jsonl "Team Name" "batch_id"
```

---

## Output

| File | Description |
|---|---|
| `reports/<batch>_<timestamp>.json` | Full report with per-layer findings |
| `reports/<batch>_CONTAMINATED_<timestamp>.jsonl` | One line per flagged sample |
| `reports/run_registry.jsonl` | Permanent audit log of every run |

Exit codes:
- `0` = APPROVED
- `1` = REJECTED

---

## Replaying a Past Run

```bash
uv run python scripts/replay.py <run_id>           # show metadata
uv run python scripts/replay.py <run_id> --execute  # re-run
```

---

## Programmatic Usage

```python
from core import ContaminationScanner

scanner = ContaminationScanner({
    "benchmarks_path": "benchmarks",
    "reports_path": "reports",
})

approved, report = scanner.scan_dataset("data.jsonl", "team-a", "batch-01")
```

Available config keys (all optional, shown with defaults):

| Key | Default | Description |
|---|---|---|
| `benchmarks_path` | `"benchmarks"` | Directory with `*_test.jsonl` files |
| `reports_path` | `"reports"` | Directory where reports are written |
| `ngram_size` | `13` | N-gram width |
| `minhash_threshold` | `0.8` | Jaccard similarity threshold |
| `minhash_permutations` | `128` | MinHash accuracy |
| `semantic_threshold` | `0.9` | Cosine similarity threshold |
| `semantic_model` | `"all-MiniLM-L6-v2"` | Embedding model |
| `semantic_batch_size` | `512` | Encoding batch size |
| `enable_semantic` | `true` | Enable/disable semantic detector layer |
| `report_sample_limit` | `50` | Max flagged samples shown per layer |
| `build_workers` | `CPU count` | Worker threads for N-gram/MinHash index build |
| `cache_indexes` | `true` | Reuse persisted detector indexes across runs |
| `cache_dir` | `".cache/indexes"` | Root directory for persisted index caches |

Index caching:
- First run builds indexes and writes cache artifacts under `cache_dir/<fingerprint>/`.
- Later runs with unchanged benchmarks + relevant config load caches automatically.
- Any benchmark/config change generates a new fingerprint and rebuilds safely.

No-semantic fast path:
- Use `uv run python scripts/scan_no_semantic.py data.jsonl "Team Name" "batch_id"` to run only N-gram + MinHash.
