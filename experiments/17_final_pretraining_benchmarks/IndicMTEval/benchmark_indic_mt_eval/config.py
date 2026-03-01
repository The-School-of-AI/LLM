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
    data_dir: str | None = None

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
