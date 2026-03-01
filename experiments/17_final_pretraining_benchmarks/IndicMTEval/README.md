# IndicMT-Eval Benchmark

A production-ready evaluation harness for the [IndicMT-Eval](https://github.com/AI4Bharat/IndicMT-Eval) benchmark (ACL 2023). This tool computes MT metric correlations with human MQM (Multidimensional Quality Metric) judgments across 5 Indian languages.

**Paper:** [IndicMT Eval: A Dataset to Meta-Evaluate Machine Translation Metrics for Indian Languages](https://arxiv.org/abs/2212.10180)

## What This Does

IndicMT-Eval is a **meta-evaluation benchmark** — it measures how well automatic MT metrics (BLEU, chrF, COMET, etc.) agree with human quality judgments. Given translations from 7 MT systems annotated by human experts, this harness:

1. Computes selected MT metric scores for each translation
2. Measures correlation (Pearson ρ, Kendall τ) between metric scores and human MQM scores
3. Reports results at **segment level** (per-sentence) and **system level** (per-MT-system average)

## Dataset

| Property | Value |
|----------|-------|
| Languages | Hindi, Tamil, Marathi, Malayalam, Gujarati |
| MT Systems | mBART, mT5, IndicTrans, CVIT, NLLB, Azure, Google Translate |
| Total annotations | ~7,000 (1,400 per language) |
| Human scoring | MQM: up to 5 errors/segment, 13 error categories, 5 severity levels |
| Score range | s = 25 − Σ(wᵢ × eᵢ), normalized to [0, 1] |
| Splits | train (~1000), val (~200), test (~276) per language |

**JSONL fields:** `src` (English source), `ref` (reference translation), `translation` (MT output), `mqm_norm_score` (human score), `adequacy_score`, `fluency_score`, `full_score`, `completion` (error spans)

## Quick Start

### 1. Install

```bash
cd experiments/17_final_pretraining_benchmarks/IndicMTEval
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Verify (~1 min)

```bash
bash scripts/run_verify.sh
```

Runs Hindi only, 10 samples, overlap metrics (BLEU, chrF, TER, ROUGE-L). Produces `results_verify.json`.

### 3. Dev (~5 min)

```bash
bash scripts/run_dev.sh
```

Hindi + Tamil, 50 samples, segment + system level.

### 4. Full Test

```bash
bash scripts/run_test.sh
```

All 5 languages, all samples, segment + system level.

## Usage

### CLI

```bash
# Minimal run
python -m benchmark_indic_mt_eval --languages hi --max-samples 10 --metrics bleu chrf -o results.json

# With YAML config
python -m benchmark_indic_mt_eval --config configs/dev.yaml

# Override config options
python -m benchmark_indic_mt_eval --config configs/test.yaml --languages hi ta --max-samples 100

# Verbose output
python -m benchmark_indic_mt_eval --config configs/verify.yaml -v
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | YAML config file | None |
| `--languages` | Language codes or `all` | `hi ta mr ml gu` |
| `--split` | Dataset split: train, val, test | `test` |
| `--max-samples` | Max samples per language | None (all) |
| `--data-dir` | Local data directory | Download from GitHub |
| `--metrics` | Metrics to compute | `bleu chrf ter` |
| `--device` | Device for neural metrics | `auto` |
| `--output`, `-o` | Output JSON path | `results.json` |
| `--levels` | Evaluation levels | `segment system` |
| `--verbose`, `-v` | Verbose logging | false |

### Programmatic API

```python
from benchmark_indic_mt_eval.config import BenchmarkConfig
from benchmark_indic_mt_eval.runner import run_benchmark

config = BenchmarkConfig.from_dict({
    "data": {"languages": ["hi"], "max_samples": 50},
    "metrics": {"metrics": ["bleu", "chrf", "ter"]},
    "run": {"output": "my_results.json", "levels": ["segment", "system"]},
})
results = run_benchmark(config)
```

## Project Structure

| Module | Purpose |
|--------|---------|
| `benchmark_indic_mt_eval/config.py` | Dataclass configs (Data, Metric, Run, Benchmark) |
| `benchmark_indic_mt_eval/data/loader.py` | Load MQM data from GitHub or local JSONL |
| `benchmark_indic_mt_eval/metrics/registry.py` | Metric factory: register/discover metrics |
| `benchmark_indic_mt_eval/metrics/overlap.py` | BLEU, chrF, TER, ROUGE-L (CPU, no deps) |
| `benchmark_indic_mt_eval/metrics/embedding.py` | BERTScore (optional, GPU recommended) |
| `benchmark_indic_mt_eval/metrics/trained.py` | COMET (optional, GPU recommended) |
| `benchmark_indic_mt_eval/evaluation/correlation.py` | Pearson ρ, Kendall τ computation |
| `benchmark_indic_mt_eval/evaluation/evaluator.py` | Per-language eval orchestration |
| `benchmark_indic_mt_eval/runner.py` | End-to-end benchmark orchestrator |
| `benchmark_indic_mt_eval/cli.py` | Argparse CLI with config merging |
| `configs/` | YAML configs: verify, dev, test |
| `scripts/` | Shell wrappers: run_verify.sh, run_dev.sh, run_test.sh |
| `tests/` | pytest suite: 36 tests |

## Metrics

### Tier 1: Overlap (CPU, fast)

| Metric | Description | Package |
|--------|-------------|---------|
| `bleu` | SacreBLEU sentence-level BLEU | `sacrebleu` |
| `chrf` | Character n-gram F-score (chrF++) | `sacrebleu` |
| `ter` | Translation Edit Rate | `sacrebleu` |
| `rouge_l` | ROUGE-L F-measure | `rouge-score` |

### Tier 2: Embedding (GPU optional)

| Metric | Description | Package |
|--------|-------------|---------|
| `bertscore` | BERTScore F1 (multilingual) | `bert-score` |

### Tier 3: Trained (GPU recommended)

| Metric | Description | Package |
|--------|-------------|---------|
| `comet` | COMET-22 (regression, source-aware) | `unbabel-comet` |

Install neural metrics:
```bash
uv pip install bert-score unbabel-comet torch
```

## Evaluation Protocol

### Segment Level

For each (source, hypothesis, reference, human_score) tuple:
1. Compute metric score for the hypothesis
2. Collect all metric scores and human scores for a language
3. Compute Pearson ρ and Kendall τ correlation

### System Level

1. Group samples by source sentence (each source has ~7 MT system translations)
2. Average metric scores and human scores per source group
3. Compute Pearson ρ and Kendall τ on the averaged values

## Output Format

```json
{
  "config": {
    "languages": ["hi"],
    "split": "test",
    "metrics": ["bleu", "chrf", "ter", "rouge_l"],
    "levels": ["segment"],
    "elapsed_seconds": 1.2
  },
  "results": {
    "segment_level": {
      "hi": {
        "bleu": {"pearson": 0.49, "kendall_tau": 0.40, "n": 276},
        "chrf": {"pearson": 0.35, "kendall_tau": 0.35, "n": 276}
      }
    }
  },
  "summary": {
    "segment_level_avg": {
      "bleu": {"pearson": 0.49, "kendall_tau": 0.40},
      "chrf": {"pearson": 0.35, "kendall_tau": 0.35}
    }
  }
}
```

## Flows

| Flow | Config | Languages | Samples | Metrics | Levels | Runtime |
|------|--------|-----------|---------|---------|--------|---------|
| verify | `configs/verify.yaml` | hi | 10 | overlap | segment | ~1 min |
| dev | `configs/dev.yaml` | hi, ta | 50 | overlap | segment, system | ~5 min |
| test | `configs/test.yaml` | all 5 | all | overlap | segment, system | ~10 min |

## Testing

```bash
# Run all tests
source .venv/bin/activate
PYTHONPATH=. python -m pytest tests/ -v

# Run specific test file
PYTHONPATH=. python -m pytest tests/test_metrics.py -v

# Run with coverage
PYTHONPATH=. python -m pytest tests/ --cov=benchmark_indic_mt_eval --cov-report=term-missing
```

Test suite: **36 tests passing**, 3 skipped (neural metrics require optional deps).

## Configuration

Configuration uses three-level merging: **defaults → YAML file → CLI overrides**.

```python
# 1. Defaults (in config.py)
BenchmarkConfig()  # all 5 languages, test split, bleu/chrf/ter, segment+system

# 2. YAML overrides defaults
load_config("configs/verify.yaml")

# 3. CLI overrides everything
python -m benchmark_indic_mt_eval --config configs/verify.yaml --languages hi ta
```

## References

- **Paper:** Sai et al., "IndicMT Eval: A Dataset to Meta-Evaluate Machine Translation Metrics for Indian Languages", ACL 2023
- **Dataset:** [AI4Bharat/IndicMT-Eval](https://github.com/AI4Bharat/IndicMT-Eval)
- **Arxiv:** [2212.10180](https://arxiv.org/abs/2212.10180)
