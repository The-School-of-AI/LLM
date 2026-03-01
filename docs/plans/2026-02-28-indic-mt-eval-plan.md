# IndicMTEval Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a production-ready benchmark evaluation harness for IndicMT-Eval that computes MT metric correlations with human MQM judgments across 5 Indian languages.

**Architecture:** Modular Python package with separate data loading, metric computation, correlation evaluation, and CLI modules. Follows the Indic-RAG-Suite pattern with factory registry for metrics, YAML configs for verify/dev/test flows, and comprehensive pytest tests.

**Tech Stack:** Python 3.10+, sacrebleu, rouge-score, bert-score, unbabel-comet, scipy, pandas, pytest, PyYAML

---

### Task 1: Create branch and project scaffolding

**Files:**
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/benchmark_indic_mt_eval/__init__.py`
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/benchmark_indic_mt_eval/__main__.py`
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/pyproject.toml`
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/tests/__init__.py`

**Step 1: Create branch from main**

```bash
git checkout main && git pull
git checkout -b p17/benchmark-eval-scripts/indic-mte-eval
```

**Step 2: Create directory structure**

```bash
mkdir -p experiments/17_final_pretraining_benchmarks/IndicMTEval/benchmark_indic_mt_eval/{data,metrics,evaluation}
mkdir -p experiments/17_final_pretraining_benchmarks/IndicMTEval/{configs,scripts,tests}
```

**Step 3: Write pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "benchmark-indic-mt-eval"
version = "0.1.0"
description = "IndicMT-Eval: Meta-evaluate MT metrics against human MQM judgments for Indian languages"
requires-python = ">=3.10"
dependencies = [
    "sacrebleu>=2.0",
    "rouge-score",
    "scipy",
    "pandas",
    "requests",
    "pyyaml",
]

[project.optional-dependencies]
neural = [
    "bert-score",
    "unbabel-comet>=2.0",
    "torch",
]
dev = [
    "pytest>=7.0",
    "pytest-cov",
]
all = [
    "benchmark-indic-mt-eval[neural,dev]",
]

[project.scripts]
indic-mt-eval = "benchmark_indic_mt_eval.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 4: Write `__init__.py`**

```python
"""IndicMT-Eval: Meta-evaluate MT metrics for Indian languages."""

from benchmark_indic_mt_eval.config import BenchmarkConfig, load_config
from benchmark_indic_mt_eval.runner import run_benchmark

__all__ = ["BenchmarkConfig", "load_config", "run_benchmark"]
```

**Step 5: Write `__main__.py`**

```python
"""Allow running as: python -m benchmark_indic_mt_eval"""

from benchmark_indic_mt_eval.cli import main

if __name__ == "__main__":
    main()
```

**Step 6: Write empty `__init__.py` files for subpackages**

Create empty `__init__.py` in `data/`, `metrics/`, `evaluation/`, and `tests/`.

**Step 7: Commit**

```bash
git add experiments/17_final_pretraining_benchmarks/IndicMTEval/
git commit -m "feat(indic-mt-eval): scaffold project structure with pyproject.toml"
```

---

### Task 2: Config system

**Files:**
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/benchmark_indic_mt_eval/config.py`
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
import pytest
from benchmark_indic_mt_eval.config import (
    BenchmarkConfig,
    DataConfig,
    MetricConfig,
    RunConfig,
    load_config,
)


def test_default_config():
    cfg = BenchmarkConfig()
    assert cfg.data.languages == ["hi", "ta", "mr", "ml", "gu"]
    assert cfg.data.split == "test"
    assert cfg.data.max_samples is None
    assert cfg.metrics.metrics == ["bleu", "chrf", "ter"]
    assert cfg.metrics.device == "auto"
    assert cfg.run.levels == ["segment", "system"]
    assert cfg.run.output == "results.json"


def test_config_from_dict():
    d = {
        "data": {"languages": ["hi"], "max_samples": 10},
        "metrics": {"metrics": ["bleu"]},
    }
    cfg = BenchmarkConfig.from_dict(d)
    assert cfg.data.languages == ["hi"]
    assert cfg.data.max_samples == 10
    assert cfg.metrics.metrics == ["bleu"]
    # Defaults preserved for unset fields
    assert cfg.run.output == "results.json"


def test_load_config_from_yaml(tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(
        "data:\n  languages: [hi, ta]\n  max_samples: 5\nmetrics:\n  metrics: [bleu]\n"
    )
    cfg = load_config(str(yaml_file))
    assert cfg.data.languages == ["hi", "ta"]
    assert cfg.data.max_samples == 5


def test_config_all_languages_shortcut():
    cfg = BenchmarkConfig.from_dict({"data": {"languages": ["all"]}})
    assert cfg.data.languages == ["hi", "ta", "mr", "ml", "gu"]
```

**Step 2: Run test to verify it fails**

Run: `cd experiments/17_final_pretraining_benchmarks/IndicMTEval && python -m pytest tests/test_config.py -v`
Expected: FAIL (module not found)

**Step 3: Write implementation**

```python
# benchmark_indic_mt_eval/config.py
"""Configuration dataclasses for IndicMT-Eval benchmark."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ALL_LANGUAGES = ["hi", "ta", "mr", "ml", "gu"]

LANGUAGE_NAMES = {
    "hi": "Hindi",
    "ta": "Tamil",
    "mr": "Marathi",
    "ml": "Malayalam",
    "gu": "Gujarati",
}

# Mapping from language code to filename prefix used in the GitHub repo
LANG_CODE_TO_PREFIX = {
    "hi": "Hin",
    "ta": "Tam",
    "mr": "Mar",
    "ml": "Mal",
    "gu": "Guj",
}


@dataclass
class DataConfig:
    languages: list[str] = field(default_factory=lambda: list(ALL_LANGUAGES))
    split: str = "test"
    max_samples: int | None = None
    data_dir: str | None = None  # Local path; if None, download from GitHub

    def __post_init__(self):
        if self.languages == ["all"]:
            self.languages = list(ALL_LANGUAGES)


@dataclass
class MetricConfig:
    metrics: list[str] = field(default_factory=lambda: ["bleu", "chrf", "ter"])
    device: str = "auto"
    batch_size: int = 32
    comet_model: str = "Unbabel/wmt22-comet-da"


@dataclass
class RunConfig:
    output: str = "results.json"
    levels: list[str] = field(default_factory=lambda: ["segment", "system"])
    verbose: bool = False


@dataclass
class BenchmarkConfig:
    data: DataConfig = field(default_factory=DataConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)
    run: RunConfig = field(default_factory=RunConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchmarkConfig:
        data_cfg = DataConfig(**d.get("data", {}))
        metric_cfg = MetricConfig(**d.get("metrics", {}))
        run_cfg = RunConfig(**d.get("run", {}))
        return cls(data=data_cfg, metrics=metric_cfg, run=run_cfg)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(yaml_path: str, overrides: dict[str, Any] | None = None) -> BenchmarkConfig:
    with open(yaml_path) as f:
        file_cfg = yaml.safe_load(f) or {}
    if overrides:
        file_cfg = _deep_merge(file_cfg, overrides)
    return BenchmarkConfig.from_dict(file_cfg)
```

**Step 4: Run test to verify it passes**

Run: `cd experiments/17_final_pretraining_benchmarks/IndicMTEval && python -m pytest tests/test_config.py -v`
Expected: 4 PASS

**Step 5: Commit**

```bash
git add benchmark_indic_mt_eval/config.py tests/test_config.py
git commit -m "feat(indic-mt-eval): add config system with dataclasses and YAML loading"
```

---

### Task 3: Data loader

**Files:**
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/benchmark_indic_mt_eval/data/loader.py`
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/tests/test_loader.py`

**Step 1: Write the failing test**

```python
# tests/test_loader.py
import json
import pytest
from pathlib import Path
from benchmark_indic_mt_eval.data.loader import (
    MTSample,
    load_language_data,
    load_benchmark_data,
    GITHUB_RAW_BASE,
)


SAMPLE_JSONL_ROW = {
    "src": "The cat sat on the mat.",
    "ref": "बिल्ली चटाई पर बैठी।",
    "translation": "बिल्ली मैट पर बैठ गई।",
    "mqm_norm_score": "0.76",
    "da_norm_score": "0.72",
    "adequacy_score": "19",
    "fluency_score": "25",
    "full_score": "19",
    "completion": [],
    "prompt": "...",
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_mt_sample_fields():
    s = MTSample(
        source="hello",
        hypothesis="नमस्ते",
        reference="नमस्कार",
        human_score=0.8,
        language="hi",
        adequacy_score=20.0,
        fluency_score=22.0,
        full_score=17.0,
    )
    assert s.source == "hello"
    assert s.human_score == 0.8


def test_load_language_data_from_local(tmp_path):
    rows = [SAMPLE_JSONL_ROW] * 5
    _write_jsonl(tmp_path / "Hin_test.jsonl", rows)
    samples = load_language_data("hi", "test", data_dir=str(tmp_path))
    assert len(samples) == 5
    assert samples[0].source == "The cat sat on the mat."
    assert samples[0].hypothesis == "बिल्ली मैट पर बैठ गई।"
    assert samples[0].human_score == 0.76
    assert samples[0].language == "hi"


def test_load_language_data_with_limit(tmp_path):
    rows = [SAMPLE_JSONL_ROW] * 20
    _write_jsonl(tmp_path / "Hin_test.jsonl", rows)
    samples = load_language_data("hi", "test", data_dir=str(tmp_path), max_samples=5)
    assert len(samples) == 5


def test_load_benchmark_data_multiple_langs(tmp_path):
    for prefix, code in [("Hin", "hi"), ("Tam", "ta")]:
        _write_jsonl(tmp_path / f"{prefix}_test.jsonl", [SAMPLE_JSONL_ROW] * 3)
    data = load_benchmark_data(
        languages=["hi", "ta"], split="test", data_dir=str(tmp_path)
    )
    assert set(data.keys()) == {"hi", "ta"}
    assert len(data["hi"]) == 3
    assert len(data["ta"]) == 3


def test_load_language_data_invalid_lang():
    with pytest.raises(ValueError, match="Unknown language"):
        load_language_data("xx", "test")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_loader.py -v`
Expected: FAIL (import error)

**Step 3: Write implementation**

```python
# benchmark_indic_mt_eval/data/loader.py
"""Load IndicMT-Eval MQM-annotated data."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import requests

from benchmark_indic_mt_eval.config import LANG_CODE_TO_PREFIX

logger = logging.getLogger(__name__)

GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/AI4Bharat/IndicMT-Eval/master/"
    "Dataset/Indic%20MT%20Eval"
)


@dataclass
class MTSample:
    source: str
    hypothesis: str
    reference: str
    human_score: float  # mqm_norm_score (0-1)
    language: str
    adequacy_score: float | None = None
    fluency_score: float | None = None
    full_score: float | None = None


def _parse_row(row: dict, language: str) -> MTSample:
    return MTSample(
        source=row["src"],
        hypothesis=row["translation"],
        reference=row["ref"],
        human_score=float(row["mqm_norm_score"]),
        language=language,
        adequacy_score=float(row["adequacy_score"]) if row.get("adequacy_score") else None,
        fluency_score=float(row["fluency_score"]) if row.get("fluency_score") else None,
        full_score=float(row["full_score"]) if row.get("full_score") else None,
    )


def _download_file(url: str, dest: Path) -> None:
    logger.info("Downloading %s -> %s", url, dest)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(resp.text, encoding="utf-8")


def load_language_data(
    language: str,
    split: str,
    data_dir: str | None = None,
    max_samples: int | None = None,
) -> list[MTSample]:
    if language not in LANG_CODE_TO_PREFIX:
        raise ValueError(
            f"Unknown language '{language}'. "
            f"Valid: {list(LANG_CODE_TO_PREFIX.keys())}"
        )

    prefix = LANG_CODE_TO_PREFIX[language]
    filename = f"{prefix}_{split}.jsonl"

    if data_dir:
        filepath = Path(data_dir) / filename
    else:
        cache_dir = Path.home() / ".cache" / "indic_mt_eval"
        filepath = cache_dir / filename
        if not filepath.exists():
            url = f"{GITHUB_RAW_BASE}/{filename}"
            _download_file(url, filepath)

    samples: list[MTSample] = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            samples.append(_parse_row(row, language))
            if max_samples and len(samples) >= max_samples:
                break

    logger.info("Loaded %d samples for %s/%s", len(samples), language, split)
    return samples


def load_benchmark_data(
    languages: list[str],
    split: str,
    data_dir: str | None = None,
    max_samples: int | None = None,
) -> dict[str, list[MTSample]]:
    data: dict[str, list[MTSample]] = {}
    for lang in languages:
        data[lang] = load_language_data(lang, split, data_dir, max_samples)
    return data
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_loader.py -v`
Expected: 4 PASS

**Step 5: Commit**

```bash
git add benchmark_indic_mt_eval/data/ tests/test_loader.py
git commit -m "feat(indic-mt-eval): add data loader for MQM-annotated JSONL files"
```

---

### Task 4: Metric registry and overlap metrics

**Files:**
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/benchmark_indic_mt_eval/metrics/registry.py`
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/benchmark_indic_mt_eval/metrics/overlap.py`
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/tests/test_metrics.py`

**Step 1: Write failing tests**

```python
# tests/test_metrics.py
import pytest
from benchmark_indic_mt_eval.metrics.registry import get_metric, list_metrics
from benchmark_indic_mt_eval.metrics.overlap import (
    compute_bleu,
    compute_chrf,
    compute_ter,
    compute_rouge_l,
)


class TestRegistry:
    def test_list_metrics_includes_overlap(self):
        names = list_metrics()
        assert "bleu" in names
        assert "chrf" in names
        assert "ter" in names

    def test_get_metric_returns_callable(self):
        fn = get_metric("bleu")
        assert callable(fn)

    def test_get_metric_unknown_raises(self):
        with pytest.raises(KeyError):
            get_metric("nonexistent_metric")


class TestBLEU:
    def test_perfect_match(self):
        score = compute_bleu("hello world", "hello world")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_no_match(self):
        score = compute_bleu("aaa bbb ccc ddd", "xxx yyy zzz www")
        assert score == pytest.approx(0.0, abs=0.01)

    def test_partial_match(self):
        score = compute_bleu("the cat sat", "the cat lay")
        assert 0.0 < score < 1.0


class TestChrF:
    def test_perfect_match(self):
        score = compute_chrf("hello world", "hello world")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_partial_match(self):
        score = compute_chrf("the cat sat", "the cat lay")
        assert 0.0 < score < 1.0


class TestTER:
    def test_perfect_match(self):
        score = compute_ter("hello world", "hello world")
        assert score == pytest.approx(0.0, abs=0.01)

    def test_different(self):
        score = compute_ter("hello world", "goodbye earth")
        assert score > 0.0


class TestROUGEL:
    def test_perfect_match(self):
        score = compute_rouge_l("hello world", "hello world")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_partial_match(self):
        score = compute_rouge_l("the cat sat on mat", "the cat lay on mat")
        assert 0.0 < score < 1.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL

**Step 3: Write registry**

```python
# benchmark_indic_mt_eval/metrics/registry.py
"""Metric factory registry."""

from __future__ import annotations

from typing import Callable

MetricFn = Callable[..., float]

_REGISTRY: dict[str, MetricFn] = {}


def register_metric(name: str):
    def decorator(fn: MetricFn) -> MetricFn:
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_metric(name: str) -> MetricFn:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown metric '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_metrics() -> list[str]:
    return list(_REGISTRY.keys())
```

**Step 4: Write overlap metrics**

```python
# benchmark_indic_mt_eval/metrics/overlap.py
"""Overlap-based MT metrics: BLEU, chrF++, TER, ROUGE-L."""

from __future__ import annotations

import sacrebleu
from rouge_score import rouge_scorer

from benchmark_indic_mt_eval.metrics.registry import register_metric


@register_metric("bleu")
def compute_bleu(hypothesis: str, reference: str, **kwargs) -> float:
    result = sacrebleu.sentence_bleu(hypothesis, [reference])
    return result.score / 100.0


@register_metric("chrf")
def compute_chrf(hypothesis: str, reference: str, **kwargs) -> float:
    result = sacrebleu.sentence_chrf(hypothesis, [reference])
    return result.score / 100.0


@register_metric("ter")
def compute_ter(hypothesis: str, reference: str, **kwargs) -> float:
    result = sacrebleu.sentence_ter(hypothesis, [reference])
    return result.score / 100.0


_rouge_scorer_instance = None


def _get_rouge_scorer():
    global _rouge_scorer_instance
    if _rouge_scorer_instance is None:
        _rouge_scorer_instance = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return _rouge_scorer_instance


@register_metric("rouge_l")
def compute_rouge_l(hypothesis: str, reference: str, **kwargs) -> float:
    scorer = _get_rouge_scorer()
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure
```

**Step 5: Update `metrics/__init__.py` to auto-register**

```python
# benchmark_indic_mt_eval/metrics/__init__.py
"""MT metrics for IndicMT-Eval benchmark."""

# Import to trigger registration
import benchmark_indic_mt_eval.metrics.overlap  # noqa: F401
```

**Step 6: Run tests**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add benchmark_indic_mt_eval/metrics/ tests/test_metrics.py
git commit -m "feat(indic-mt-eval): add metric registry and overlap metrics (BLEU, chrF, TER, ROUGE-L)"
```

---

### Task 5: Neural/trained metrics (BERTScore, COMET)

**Files:**
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/benchmark_indic_mt_eval/metrics/embedding.py`
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/benchmark_indic_mt_eval/metrics/trained.py`
- Modify: `experiments/17_final_pretraining_benchmarks/IndicMTEval/benchmark_indic_mt_eval/metrics/__init__.py`
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/tests/test_neural_metrics.py`

**Step 1: Write failing tests**

```python
# tests/test_neural_metrics.py
"""Tests for neural metrics. Skipped if dependencies not installed."""

import pytest

try:
    import bert_score
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

try:
    import comet
    HAS_COMET = True
except ImportError:
    HAS_COMET = False


@pytest.mark.skipif(not HAS_BERTSCORE, reason="bert-score not installed")
class TestBERTScore:
    def test_perfect_match(self):
        from benchmark_indic_mt_eval.metrics.embedding import compute_bertscore
        score = compute_bertscore("hello world", "hello world")
        assert score > 0.9

    def test_different_sentences(self):
        from benchmark_indic_mt_eval.metrics.embedding import compute_bertscore
        score = compute_bertscore("the cat sat", "purple elephants fly")
        assert score < 0.9


@pytest.mark.skipif(not HAS_COMET, reason="unbabel-comet not installed")
class TestCOMET:
    def test_comet_returns_float(self):
        from benchmark_indic_mt_eval.metrics.trained import compute_comet
        score = compute_comet(
            hypothesis="the cat sat",
            reference="the cat sat",
            source="the cat sat",
        )
        assert isinstance(score, float)
```

**Step 2: Write embedding metrics**

```python
# benchmark_indic_mt_eval/metrics/embedding.py
"""Embedding-based MT metrics: BERTScore."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from bert_score import score as bert_score_fn
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

from benchmark_indic_mt_eval.metrics.registry import register_metric


@register_metric("bertscore")
def compute_bertscore(hypothesis: str, reference: str, **kwargs) -> float:
    if not HAS_BERTSCORE:
        raise ImportError("bert-score not installed. Install with: pip install bert-score")
    P, R, F1 = bert_score_fn(
        [hypothesis], [reference],
        lang="hi",  # multilingual model handles all Indic langs
        verbose=False,
    )
    return F1[0].item()
```

**Step 3: Write trained metrics**

```python
# benchmark_indic_mt_eval/metrics/trained.py
"""Trained MT metrics: COMET."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from comet import download_model, load_from_checkpoint
    HAS_COMET = True
except ImportError:
    HAS_COMET = False

from benchmark_indic_mt_eval.metrics.registry import register_metric

_comet_model = None


def _get_comet_model(model_name: str = "Unbabel/wmt22-comet-da"):
    global _comet_model
    if _comet_model is None:
        model_path = download_model(model_name)
        _comet_model = load_from_checkpoint(model_path)
    return _comet_model


@register_metric("comet")
def compute_comet(hypothesis: str, reference: str, source: str = "", **kwargs) -> float:
    if not HAS_COMET:
        raise ImportError("unbabel-comet not installed. Install with: pip install unbabel-comet")
    model = _get_comet_model(kwargs.get("comet_model", "Unbabel/wmt22-comet-da"))
    data = [{"src": source, "mt": hypothesis, "ref": reference}]
    output = model.predict(data, batch_size=1, gpus=0)
    return float(output.scores[0])
```

**Step 4: Update `metrics/__init__.py`**

```python
# benchmark_indic_mt_eval/metrics/__init__.py
"""MT metrics for IndicMT-Eval benchmark."""

import benchmark_indic_mt_eval.metrics.overlap  # noqa: F401

# Optional neural metrics — import only if dependencies available
try:
    import benchmark_indic_mt_eval.metrics.embedding  # noqa: F401
except ImportError:
    pass

try:
    import benchmark_indic_mt_eval.metrics.trained  # noqa: F401
except ImportError:
    pass
```

**Step 5: Run tests**

Run: `python -m pytest tests/test_neural_metrics.py -v`
Expected: PASS (or skip if deps not installed)

**Step 6: Commit**

```bash
git add benchmark_indic_mt_eval/metrics/ tests/test_neural_metrics.py
git commit -m "feat(indic-mt-eval): add neural metrics (BERTScore, COMET) with graceful degradation"
```

---

### Task 6: Correlation computation

**Files:**
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/benchmark_indic_mt_eval/evaluation/correlation.py`
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/tests/test_correlation.py`

**Step 1: Write failing tests**

```python
# tests/test_correlation.py
import pytest
import numpy as np
from benchmark_indic_mt_eval.evaluation.correlation import (
    compute_pearson,
    compute_kendall_tau,
    compute_correlations,
)


class TestPearson:
    def test_perfect_positive(self):
        r = compute_pearson([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        assert r == pytest.approx(1.0, abs=0.001)

    def test_perfect_negative(self):
        r = compute_pearson([1, 2, 3, 4, 5], [10, 8, 6, 4, 2])
        assert r == pytest.approx(-1.0, abs=0.001)

    def test_no_correlation(self):
        # Known uncorrelated values
        r = compute_pearson([1, 2, 3, 4, 5], [2, 4, 1, 5, 3])
        assert -0.5 < r < 0.5

    def test_constant_returns_zero(self):
        r = compute_pearson([1, 1, 1], [1, 2, 3])
        assert r == pytest.approx(0.0, abs=0.001)


class TestKendallTau:
    def test_perfect_concordance(self):
        tau = compute_kendall_tau([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
        assert tau == pytest.approx(1.0, abs=0.001)

    def test_perfect_discordance(self):
        tau = compute_kendall_tau([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
        assert tau == pytest.approx(-1.0, abs=0.001)


class TestComputeCorrelations:
    def test_returns_both_metrics(self):
        result = compute_correlations([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        assert "pearson" in result
        assert "kendall_tau" in result
        assert result["pearson"] == pytest.approx(1.0, abs=0.001)
        assert result["kendall_tau"] == pytest.approx(1.0, abs=0.001)

    def test_too_few_samples(self):
        result = compute_correlations([1], [2])
        assert result["pearson"] == 0.0
        assert result["kendall_tau"] == 0.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_correlation.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# benchmark_indic_mt_eval/evaluation/correlation.py
"""Correlation metrics: Pearson and Kendall-tau."""

from __future__ import annotations

import logging
from scipy import stats

logger = logging.getLogger(__name__)

MIN_SAMPLES = 3


def compute_pearson(predictions: list[float], human_scores: list[float]) -> float:
    if len(predictions) < MIN_SAMPLES:
        return 0.0
    # Handle constant arrays
    if len(set(predictions)) <= 1 or len(set(human_scores)) <= 1:
        return 0.0
    r, _ = stats.pearsonr(predictions, human_scores)
    return float(r)


def compute_kendall_tau(predictions: list[float], human_scores: list[float]) -> float:
    if len(predictions) < MIN_SAMPLES:
        return 0.0
    tau, _ = stats.kendalltau(predictions, human_scores)
    return float(tau)


def compute_correlations(
    predictions: list[float], human_scores: list[float]
) -> dict[str, float]:
    return {
        "pearson": compute_pearson(predictions, human_scores),
        "kendall_tau": compute_kendall_tau(predictions, human_scores),
    }
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_correlation.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add benchmark_indic_mt_eval/evaluation/ tests/test_correlation.py
git commit -m "feat(indic-mt-eval): add Pearson and Kendall-tau correlation computation"
```

---

### Task 7: Evaluator (per-language and system-level orchestration)

**Files:**
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/benchmark_indic_mt_eval/evaluation/evaluator.py`
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/tests/test_evaluator.py`

**Step 1: Write failing tests**

```python
# tests/test_evaluator.py
import pytest
from benchmark_indic_mt_eval.data.loader import MTSample
from benchmark_indic_mt_eval.evaluation.evaluator import (
    compute_metric_scores,
    evaluate_segment_level,
    evaluate_system_level,
    evaluate_language,
)


def _make_samples(n: int = 10) -> list[MTSample]:
    """Create synthetic samples with varying quality."""
    samples = []
    for i in range(n):
        # Higher human_score => hypothesis closer to reference
        if i < n // 2:
            hyp = f"good translation number {i}"
            ref = f"good translation number {i}"
            score = 0.9
        else:
            hyp = f"bad output {i} wrong"
            ref = f"correct translation number {i}"
            score = 0.3
        samples.append(
            MTSample(
                source=f"source sentence {i}",
                hypothesis=hyp,
                reference=ref,
                human_score=score,
                language="hi",
            )
        )
    return samples


class TestComputeMetricScores:
    def test_returns_list_of_floats(self):
        samples = _make_samples(5)
        scores = compute_metric_scores(samples, "bleu")
        assert len(scores) == 5
        assert all(isinstance(s, float) for s in scores)


class TestSegmentLevel:
    def test_returns_correlations(self):
        samples = _make_samples(20)
        result = evaluate_segment_level(samples, ["bleu", "chrf"])
        assert "bleu" in result
        assert "chrf" in result
        assert "pearson" in result["bleu"]
        assert "kendall_tau" in result["bleu"]
        assert "n" in result["bleu"]
        assert result["bleu"]["n"] == 20


class TestSystemLevel:
    def test_returns_correlations_over_systems(self):
        # We don't have real system info in MTSample, so system_level
        # averages across all samples (treated as 1 system).
        # With only 1 system, correlation is 0.0 (too few points).
        samples = _make_samples(10)
        result = evaluate_system_level(samples, ["bleu"])
        assert "bleu" in result
        # With synthetic data, n will be small
        assert "n" in result["bleu"]


class TestEvaluateLanguage:
    def test_returns_both_levels(self):
        samples = _make_samples(20)
        result = evaluate_language(samples, ["bleu"], levels=["segment", "system"])
        assert "segment_level" in result
        assert "system_level" in result
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evaluator.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# benchmark_indic_mt_eval/evaluation/evaluator.py
"""Per-language evaluation orchestration."""

from __future__ import annotations

import logging
from collections import defaultdict

from benchmark_indic_mt_eval.data.loader import MTSample
from benchmark_indic_mt_eval.metrics.registry import get_metric
from benchmark_indic_mt_eval.evaluation.correlation import compute_correlations

logger = logging.getLogger(__name__)


def compute_metric_scores(
    samples: list[MTSample], metric_name: str
) -> list[float]:
    metric_fn = get_metric(metric_name)
    scores: list[float] = []
    for sample in samples:
        try:
            score = metric_fn(
                hypothesis=sample.hypothesis,
                reference=sample.reference,
                source=sample.source,
            )
        except Exception as e:
            logger.warning("Metric %s failed on sample: %s", metric_name, e)
            score = 0.0
        scores.append(score)
    return scores


def evaluate_segment_level(
    samples: list[MTSample], metric_names: list[str]
) -> dict[str, dict[str, float]]:
    human_scores = [s.human_score for s in samples]
    results: dict[str, dict[str, float]] = {}
    for name in metric_names:
        logger.info("Computing segment-level %s for %d samples", name, len(samples))
        predicted = compute_metric_scores(samples, name)
        corr = compute_correlations(predicted, human_scores)
        corr["n"] = len(samples)
        results[name] = corr
    return results


def evaluate_system_level(
    samples: list[MTSample], metric_names: list[str]
) -> dict[str, dict[str, float]]:
    """Group samples by source sentence, average metric and human scores per 'system'.

    In IndicMT-Eval, each source has 7 MT system translations. Samples sharing
    the same source text belong to different systems. We group by source, then
    treat each source group as a system-level data point (average metric score
    vs average human score). This gives us ~200 data points for correlation.

    Note: The dataset doesn't have explicit system IDs, so we group by source
    sentence to get system-level variation.
    """
    # Group by source sentence
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        groups[s.source].append(i)

    results: dict[str, dict[str, float]] = {}
    for name in metric_names:
        logger.info("Computing system-level %s", name)
        predicted = compute_metric_scores(samples, name)
        human_scores = [s.human_score for s in samples]

        # Average per source-group
        avg_predicted: list[float] = []
        avg_human: list[float] = []
        for source, indices in groups.items():
            avg_predicted.append(sum(predicted[i] for i in indices) / len(indices))
            avg_human.append(sum(human_scores[i] for i in indices) / len(indices))

        corr = compute_correlations(avg_predicted, avg_human)
        corr["n"] = len(groups)
        results[name] = corr

    return results


def evaluate_language(
    samples: list[MTSample],
    metric_names: list[str],
    levels: list[str] | None = None,
) -> dict[str, dict]:
    if levels is None:
        levels = ["segment", "system"]

    result: dict[str, dict] = {}
    if "segment" in levels:
        result["segment_level"] = evaluate_segment_level(samples, metric_names)
    if "system" in levels:
        result["system_level"] = evaluate_system_level(samples, metric_names)
    return result
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_evaluator.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add benchmark_indic_mt_eval/evaluation/ tests/test_evaluator.py
git commit -m "feat(indic-mt-eval): add per-language evaluator with segment and system level"
```

---

### Task 8: CLI and runner

**Files:**
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/benchmark_indic_mt_eval/cli.py`
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/benchmark_indic_mt_eval/runner.py`

**Step 1: Write runner**

```python
# benchmark_indic_mt_eval/runner.py
"""End-to-end benchmark orchestrator."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from benchmark_indic_mt_eval.config import BenchmarkConfig, LANGUAGE_NAMES
from benchmark_indic_mt_eval.data.loader import load_benchmark_data
from benchmark_indic_mt_eval.evaluation.evaluator import evaluate_language

# Ensure metrics are registered
import benchmark_indic_mt_eval.metrics  # noqa: F401

logger = logging.getLogger(__name__)


def run_benchmark(config: BenchmarkConfig) -> dict:
    start = time.time()

    # Validate requested metrics are available
    from benchmark_indic_mt_eval.metrics.registry import list_metrics, get_metric
    available = list_metrics()
    for m in config.metrics.metrics:
        if m not in available:
            raise ValueError(
                f"Metric '{m}' not available. "
                f"Installed: {available}. "
                f"Install optional deps for neural metrics."
            )

    # Load data
    logger.info("Loading data for languages: %s", config.data.languages)
    data = load_benchmark_data(
        languages=config.data.languages,
        split=config.data.split,
        data_dir=config.data.data_dir,
        max_samples=config.data.max_samples,
    )

    # Evaluate per language
    results: dict = {
        "config": {
            "languages": config.data.languages,
            "split": config.data.split,
            "max_samples": config.data.max_samples,
            "metrics": config.metrics.metrics,
            "levels": config.run.levels,
        },
        "results": {},
        "summary": {},
    }

    per_lang_results: dict[str, dict] = {}
    for lang, samples in data.items():
        lang_name = LANGUAGE_NAMES.get(lang, lang)
        logger.info("Evaluating %s (%s): %d samples", lang_name, lang, len(samples))
        per_lang_results[lang] = evaluate_language(
            samples, config.metrics.metrics, config.run.levels
        )

    # Restructure: level -> lang -> metric -> correlations
    for level in config.run.levels:
        level_key = f"{level}_level"
        results["results"][level_key] = {}
        for lang in config.data.languages:
            if level_key in per_lang_results.get(lang, {}):
                results["results"][level_key][lang] = per_lang_results[lang][level_key]

    # Compute summary (average across languages)
    for level in config.run.levels:
        level_key = f"{level}_level"
        if level_key not in results["results"]:
            continue
        summary: dict[str, dict[str, float]] = {}
        for metric_name in config.metrics.metrics:
            pearson_vals = []
            kendall_vals = []
            for lang_data in results["results"][level_key].values():
                if metric_name in lang_data:
                    pearson_vals.append(lang_data[metric_name]["pearson"])
                    kendall_vals.append(lang_data[metric_name]["kendall_tau"])
            if pearson_vals:
                summary[metric_name] = {
                    "pearson": sum(pearson_vals) / len(pearson_vals),
                    "kendall_tau": sum(kendall_vals) / len(kendall_vals),
                }
        results["summary"][f"{level}_level_avg"] = summary

    elapsed = time.time() - start
    results["config"]["elapsed_seconds"] = round(elapsed, 2)

    # Write output
    output_path = Path(config.run.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results written to %s (%.1fs)", output_path, elapsed)

    return results
```

**Step 2: Write CLI**

```python
# benchmark_indic_mt_eval/cli.py
"""Command-line interface for IndicMT-Eval benchmark."""

from __future__ import annotations

import argparse
import logging
import sys

from benchmark_indic_mt_eval.config import (
    ALL_LANGUAGES,
    BenchmarkConfig,
    load_config,
)
from benchmark_indic_mt_eval.runner import run_benchmark


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IndicMT-Eval: Meta-evaluate MT metrics for Indian languages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config
    p.add_argument("--config", type=str, help="YAML config file path")

    # Data
    p.add_argument("--languages", nargs="+", default=None,
                    help=f"Languages to evaluate. Options: {ALL_LANGUAGES} or 'all'")
    p.add_argument("--split", choices=["train", "val", "test"], default=None)
    p.add_argument("--max-samples", type=int, default=None,
                    help="Max samples per language")
    p.add_argument("--data-dir", type=str, default=None,
                    help="Local data directory (skips download)")

    # Metrics
    p.add_argument("--metrics", nargs="+", default=None,
                    help="Metrics to compute (e.g., bleu chrf ter bertscore comet)")
    p.add_argument("--device", default=None, help="Device for neural metrics")

    # Run
    p.add_argument("--output", "-o", default=None, help="Output JSON path")
    p.add_argument("--levels", nargs="+", default=None,
                    choices=["segment", "system"])
    p.add_argument("--verbose", "-v", action="store_true")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Build overrides from CLI args
    overrides: dict = {}
    if args.languages:
        overrides.setdefault("data", {})["languages"] = args.languages
    if args.split:
        overrides.setdefault("data", {})["split"] = args.split
    if args.max_samples is not None:
        overrides.setdefault("data", {})["max_samples"] = args.max_samples
    if args.data_dir:
        overrides.setdefault("data", {})["data_dir"] = args.data_dir
    if args.metrics:
        overrides.setdefault("metrics", {})["metrics"] = args.metrics
    if args.device:
        overrides.setdefault("metrics", {})["device"] = args.device
    if args.output:
        overrides.setdefault("run", {})["output"] = args.output
    if args.levels:
        overrides.setdefault("run", {})["levels"] = args.levels
    if args.verbose:
        overrides.setdefault("run", {})["verbose"] = True

    # Load config
    if args.config:
        config = load_config(args.config, overrides=overrides)
    else:
        config = BenchmarkConfig.from_dict(overrides)

    # Run
    results = run_benchmark(config)

    # Print summary
    print("\n=== IndicMT-Eval Results ===")
    for level_key, summary in results.get("summary", {}).items():
        print(f"\n{level_key}:")
        for metric, corrs in summary.items():
            print(f"  {metric:12s}  Pearson={corrs['pearson']:.4f}  "
                  f"Kendall-τ={corrs['kendall_tau']:.4f}")


if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add benchmark_indic_mt_eval/cli.py benchmark_indic_mt_eval/runner.py
git commit -m "feat(indic-mt-eval): add CLI and runner for end-to-end benchmark execution"
```

---

### Task 9: YAML configs and shell scripts

**Files:**
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/configs/verify.yaml`
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/configs/dev.yaml`
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/configs/test.yaml`
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/scripts/run_verify.sh`
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/scripts/run_dev.sh`
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/scripts/run_test.sh`

**Step 1: Write verify.yaml**

```yaml
# verify.yaml — Quick sanity check (~1 min, CPU only)
data:
  languages: [hi]
  split: test
  max_samples: 10

metrics:
  metrics: [bleu, chrf, ter, rouge_l]

run:
  output: results_verify.json
  levels: [segment]
  verbose: true
```

**Step 2: Write dev.yaml**

```yaml
# dev.yaml — Development iteration (~5 min)
data:
  languages: [hi, ta]
  split: test
  max_samples: 50

metrics:
  metrics: [bleu, chrf, ter, rouge_l]

run:
  output: results_dev.json
  levels: [segment, system]
```

**Step 3: Write test.yaml**

```yaml
# test.yaml — Full evaluation (all languages, all samples)
data:
  languages: [hi, ta, mr, ml, gu]
  split: test

metrics:
  metrics: [bleu, chrf, ter, rouge_l]

run:
  output: results_test.json
  levels: [segment, system]
```

**Step 4: Write shell scripts**

```bash
#!/usr/bin/env bash
# scripts/run_verify.sh — Quick sanity check
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
python -m benchmark_indic_mt_eval --config configs/verify.yaml "$@"
```

```bash
#!/usr/bin/env bash
# scripts/run_dev.sh — Development iteration
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
python -m benchmark_indic_mt_eval --config configs/dev.yaml "$@"
```

```bash
#!/usr/bin/env bash
# scripts/run_test.sh — Full evaluation
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
python -m benchmark_indic_mt_eval --config configs/test.yaml "$@"
```

**Step 5: Make scripts executable and commit**

```bash
chmod +x scripts/*.sh
git add configs/ scripts/
git commit -m "feat(indic-mt-eval): add verify/dev/test YAML configs and shell scripts"
```

---

### Task 10: Integration test

**Files:**
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
"""End-to-end integration tests using local synthetic data."""

import json
import pytest
from pathlib import Path

from benchmark_indic_mt_eval.config import BenchmarkConfig
from benchmark_indic_mt_eval.runner import run_benchmark


def _create_synthetic_data(data_dir: Path) -> None:
    """Create minimal synthetic JSONL files for testing."""
    for prefix in ["Hin", "Tam"]:
        filepath = data_dir / f"{prefix}_test.jsonl"
        rows = []
        for i in range(15):
            score = round(0.3 + 0.05 * i, 2)
            rows.append(json.dumps({
                "src": f"Source sentence number {i}.",
                "ref": f"Reference translation number {i}.",
                "translation": f"Reference translation number {i}." if i < 7
                    else f"Wrong output {i} bad.",
                "mqm_norm_score": str(score),
                "da_norm_score": str(score),
                "adequacy_score": str(int(score * 25)),
                "fluency_score": str(int(score * 25)),
                "full_score": str(int(score * 20)),
                "completion": [],
                "prompt": "...",
            }, ensure_ascii=False))
        filepath.write_text("\n".join(rows) + "\n", encoding="utf-8")


class TestEndToEnd:
    def test_verify_flow(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _create_synthetic_data(data_dir)

        output_path = tmp_path / "results.json"
        config = BenchmarkConfig.from_dict({
            "data": {
                "languages": ["hi"],
                "split": "test",
                "max_samples": 10,
                "data_dir": str(data_dir),
            },
            "metrics": {"metrics": ["bleu", "chrf"]},
            "run": {
                "output": str(output_path),
                "levels": ["segment"],
            },
        })

        results = run_benchmark(config)

        # Verify structure
        assert "config" in results
        assert "results" in results
        assert "summary" in results
        assert "segment_level" in results["results"]
        assert "hi" in results["results"]["segment_level"]
        assert "bleu" in results["results"]["segment_level"]["hi"]
        assert "pearson" in results["results"]["segment_level"]["hi"]["bleu"]
        assert "kendall_tau" in results["results"]["segment_level"]["hi"]["bleu"]

        # Verify output file
        assert output_path.exists()
        with open(output_path) as f:
            saved = json.load(f)
        assert saved["config"]["languages"] == ["hi"]

    def test_multi_language_flow(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _create_synthetic_data(data_dir)

        output_path = tmp_path / "results.json"
        config = BenchmarkConfig.from_dict({
            "data": {
                "languages": ["hi", "ta"],
                "split": "test",
                "data_dir": str(data_dir),
            },
            "metrics": {"metrics": ["bleu"]},
            "run": {
                "output": str(output_path),
                "levels": ["segment", "system"],
            },
        })

        results = run_benchmark(config)

        assert "hi" in results["results"]["segment_level"]
        assert "ta" in results["results"]["segment_level"]
        assert "hi" in results["results"]["system_level"]

        # Summary averages
        assert "segment_level_avg" in results["summary"]
        assert "bleu" in results["summary"]["segment_level_avg"]

    def test_system_level_correlation(self, tmp_path):
        """Verify system-level produces valid correlation values."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _create_synthetic_data(data_dir)

        output_path = tmp_path / "results.json"
        config = BenchmarkConfig.from_dict({
            "data": {
                "languages": ["hi"],
                "split": "test",
                "data_dir": str(data_dir),
            },
            "metrics": {"metrics": ["bleu", "ter"]},
            "run": {
                "output": str(output_path),
                "levels": ["system"],
            },
        })

        results = run_benchmark(config)
        sys_results = results["results"]["system_level"]["hi"]
        for metric in ["bleu", "ter"]:
            assert -1.0 <= sys_results[metric]["pearson"] <= 1.0
            assert -1.0 <= sys_results[metric]["kendall_tau"] <= 1.0
```

**Step 2: Run integration tests**

Run: `python -m pytest tests/test_integration.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test(indic-mt-eval): add end-to-end integration tests with synthetic data"
```

---

### Task 11: README documentation

**Files:**
- Create: `experiments/17_final_pretraining_benchmarks/IndicMTEval/README.md`

**Step 1: Write comprehensive README**

Write a README covering:

1. **Overview** — What IndicMT-Eval is, ACL 2023 paper link, GitHub repo link
2. **Dataset** — 5 languages, 7 MT systems, 7000 MQM annotations, fields
3. **Quick Start** — 3 steps: install, verify, dev
4. **Usage** — CLI reference, config files, programmatic API
5. **Project Structure** — Module table
6. **Metrics** — Table of all metrics with descriptions and tiers
7. **Evaluation Protocol** — Segment-level vs system-level, Pearson vs Kendall
8. **Flows** — verify (~1 min), dev (~5 min), test (full)
9. **Optional Neural Metrics** — How to install and use BERTScore, COMET

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs(indic-mt-eval): add comprehensive README with usage guide and metrics reference"
```

---

### Task 12: Run verify flow with real data and validate

**Step 1: Install the package**

```bash
cd experiments/17_final_pretraining_benchmarks/IndicMTEval
uv pip install -e ".[dev]"
```

**Step 2: Run verify**

```bash
bash scripts/run_verify.sh
```

Expected: Runs to completion, produces `results_verify.json` with segment-level correlations for Hindi with 10 samples.

**Step 3: Inspect results**

```bash
cat results_verify.json | python -m json.tool
```

**Step 4: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: All tests pass.

**Step 5: Commit results (optional sample)**

```bash
git add results_verify.json
git commit -m "chore(indic-mt-eval): add sample verify results"
```

---

### Task 13: Final cleanup and push

**Step 1: Run all tests one final time**

```bash
python -m pytest tests/ -v --tb=short
```

**Step 2: Review all files**

```bash
git diff main --stat
```

**Step 3: Push branch**

```bash
git push -u origin p17/benchmark-eval-scripts/indic-mte-eval
```
