# IndicMTEval Benchmark — Design Document

**Date:** 2026-02-28
**Branch:** `p17/benchmark-eval-scripts/indic-mte-eval`
**Reference implementations:** `p17/RiddleBench`, `p17/scripts/benchmark-scripts/indic-rag-suite`

## Overview

IndicMTEval is a **meta-evaluation benchmark** (ACL 2023) that measures how well automatic MT metrics correlate with human MQM (Multidimensional Quality Metric) judgments for 5 Indian languages. The dataset contains 7,000 annotations across 7 MT systems.

**Paper:** [IndicMT Eval: A Dataset to Meta-Evaluate Machine Translation Metrics for Indian Languages](https://arxiv.org/abs/2212.10180)
**Source:** [AI4Bharat/IndicMT-Eval](https://github.com/AI4Bharat/IndicMT-Eval)

## Task Definition

Given:
- Source English text
- Machine translation output (hypothesis)
- Reference translation
- Human MQM scores (0–25 scale)

Compute various MT metrics and measure their **correlation** (Pearson ρ, Kendall τ) with human judgments at:
- **Segment level**: per-sentence metric vs human score
- **System level**: average metric per MT system vs average human score per system

## Dataset Details

| Field | Description |
|-------|-------------|
| Languages | Hindi, Tamil, Marathi, Malayalam, Gujarati |
| MT Systems | mBART, mT5, IndicTrans, CVIT, NLLB, Azure, Google Translate |
| Annotations | ~1,400 per language, 7,000 total |
| Scoring | MQM: up to 5 errors/segment, 13 error categories, 5 severity levels |
| Human Score | s = 25 − Σ(wᵢ × eᵢ) |

The data is hosted on the [GitHub repository](https://github.com/AI4Bharat/IndicMT-Eval/tree/master/Dataset) as CSV/TSV files.

## Architecture

Modular package following the Indic-RAG-Suite pattern:

```
IndicMTEval/
├── benchmark_indic_mt_eval/          # Main Python package
│   ├── __init__.py                   # Public API exports
│   ├── __main__.py                   # python -m entry point
│   ├── cli.py                        # Argparse CLI → config overrides
│   ├── config.py                     # Dataclass configs (DataConfig, MetricConfig, RunConfig)
│   ├── runner.py                     # End-to-end orchestrator
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py                 # Load MQM data from GitHub repo / local files
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── registry.py               # Metric factory: register/discover metrics
│   │   ├── overlap.py                # BLEU, chrF++, TER, ROUGE-L (no GPU)
│   │   ├── embedding.py              # BERTScore (GPU optional)
│   │   └── trained.py                # COMET, BLEURT, Indic-COMET (GPU preferred)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── correlation.py            # Pearson ρ, Kendall τ computation
│   │   └── evaluator.py              # Per-language eval orchestration
│   └── utils.py                      # Logging, device detection
├── configs/
│   ├── verify.yaml                   # 10 samples, overlap metrics only (~1 min)
│   ├── dev.yaml                      # 50 samples, all metrics (~5 min)
│   └── test.yaml                     # Full dataset, all metrics
├── scripts/
│   ├── run_verify.sh
│   ├── run_dev.sh
│   └── run_test.sh
├── tests/
│   ├── __init__.py
│   ├── test_loader.py                # Data loading, normalization, filtering
│   ├── test_metrics.py               # Each metric category
│   ├── test_correlation.py           # Pearson/Kendall computation
│   └── test_evaluator.py             # End-to-end integration
├── pyproject.toml
└── README.md
```

## Component Details

### 1. Data Loader (`data/loader.py`)

- Loads MQM-annotated data from the IndicMT-Eval GitHub repo (CSV/TSV)
- Normalizes to a common schema: `MTSample(source, hypothesis, reference, human_score, system_name, language)`
- Supports filtering by language, system, and sample limits
- Caches downloaded data locally

### 2. Metric Registry (`metrics/registry.py`)

Factory pattern with decorator-based registration:

```python
@register_metric("bleu")
def compute_bleu(hypothesis: str, reference: str, **kwargs) -> float: ...

score = get_metric("bleu")(hypothesis, reference)
```

### 3. Metrics

**Tier 1 — Overlap (fast, CPU-only):**
- `sacrebleu` for BLEU, chrF++, TER
- `rouge-score` for ROUGE-L

**Tier 2 — Embedding (moderate):**
- `bert-score` for BERTScore (with multilingual model)

**Tier 3 — Trained (GPU-preferred):**
- `unbabel-comet` for COMET-22
- Indic-COMET via COMET framework with custom checkpoint
- BLEURT (optional, heavy dependency)

### 4. Correlation Evaluator (`evaluation/`)

- **Segment-level**: Compute metric scores for all segments, correlate with human MQM scores
- **System-level**: Average metric scores per MT system, correlate with average human scores
- Both Pearson ρ and Kendall τ computed per language and overall
- Results stratified by language and metric

### 5. Configuration (`config.py`)

Three-level merging: defaults → YAML → CLI overrides.

```python
@dataclass
class DataConfig:
    languages: list[str]       # ["hi", "ta", "mr", "ml", "gu"] or subset
    max_samples: int | None    # Per-language cap
    data_dir: str | None       # Local data path (else download from GitHub)

@dataclass
class MetricConfig:
    metrics: list[str]         # ["bleu", "chrf", "ter", "bertscore", "comet"]
    comet_model: str           # "Unbabel/wmt22-comet-da"
    device: str                # "auto", "cpu", "cuda"
    batch_size: int            # For neural metrics

@dataclass
class RunConfig:
    output: str                # Results JSON path
    levels: list[str]          # ["segment", "system"]
    verbose: bool
```

### 6. Verify / Dev / Test Flows

| Flow | Samples | Metrics | Runtime | Purpose |
|------|---------|---------|---------|---------|
| verify | 10/lang | overlap only | ~1 min | Sanity check |
| dev | 50/lang | all | ~5 min | Development iteration |
| test | all | all | ~30+ min | Full evaluation |

## Output Format

```json
{
  "config": {
    "languages": ["hi", "ta", "mr", "ml", "gu"],
    "metrics": ["bleu", "chrf", "ter", "bertscore", "comet"],
    "levels": ["segment", "system"]
  },
  "results": {
    "segment_level": {
      "hi": {
        "bleu": {"pearson": 0.35, "kendall_tau": 0.28, "n": 1400},
        "comet": {"pearson": 0.72, "kendall_tau": 0.55, "n": 1400}
      }
    },
    "system_level": {
      "hi": {
        "bleu": {"pearson": 0.82, "kendall_tau": 0.71, "n": 7},
        "comet": {"pearson": 0.95, "kendall_tau": 0.90, "n": 7}
      }
    }
  },
  "summary": {
    "segment_level_avg": {
      "bleu": {"pearson": 0.33, "kendall_tau": 0.26},
      "comet": {"pearson": 0.70, "kendall_tau": 0.53}
    },
    "system_level_avg": {
      "bleu": {"pearson": 0.80, "kendall_tau": 0.69},
      "comet": {"pearson": 0.93, "kendall_tau": 0.88}
    }
  }
}
```

## Dependencies

```
# Core
sacrebleu>=2.0
rouge-score
scipy                    # Pearson, Kendall
pandas                   # Data loading
requests                 # Download from GitHub

# Optional (Tier 2+)
bert-score               # BERTScore
unbabel-comet>=2.0       # COMET, Indic-COMET

# Dev
pytest
```

## Testing Strategy

1. **Unit tests** for data loading (schema validation, filtering, edge cases)
2. **Unit tests** for each metric (known input/output pairs)
3. **Unit tests** for correlation computation (synthetic data with known correlations)
4. **Integration test** using verify config (end-to-end pipeline)
5. All tests runnable without GPU using overlap metrics
