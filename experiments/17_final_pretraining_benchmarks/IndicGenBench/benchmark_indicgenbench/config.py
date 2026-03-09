"""Benchmark configuration: dataclasses, YAML loading, CLI overrides."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# All 29 languages in IndicGenBench
ALL_LANGUAGES = [
    "as", "awa", "bgc", "bho", "bn", "bo", "brx", "gbm", "gom", "gu",
    "hi", "hne", "hoj", "kn", "mai", "ml", "mni", "mr", "mup", "mwr",
    "ne", "or", "pa", "ps", "sa", "sat", "ta", "te", "ur",
]

# XQuAD-IN only covers 12 languages
XQUAD_LANGUAGES = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur"]

LANGUAGE_NAMES = {
    "as": "Assamese", "awa": "Awadhi", "bgc": "Haryanvi", "bho": "Bhojpuri",
    "bn": "Bengali", "bo": "Tibetan", "brx": "Bodo", "gbm": "Garhwali",
    "gom": "Konkani", "gu": "Gujarati", "hi": "Hindi", "hne": "Chhattisgarhi",
    "hoj": "Rajasthani", "kn": "Kannada", "mai": "Maithili", "ml": "Malayalam",
    "mni": "Manipuri", "mr": "Marathi", "mup": "Malvi", "mwr": "Marwari",
    "ne": "Nepali", "or": "Odia", "pa": "Punjabi", "ps": "Pashto",
    "sa": "Sanskrit", "sat": "Santali", "ta": "Tamil", "te": "Telugu", "ur": "Urdu",
}

ALL_TASKS = ["crosssum", "flores", "xquad", "xorqa"]

HF_DATASETS = {
    "crosssum": "google/IndicGenBench_crosssum_in",
    "flores": "google/IndicGenBench_flores_in",
    "xquad": "google/IndicGenBench_xquad_in",
    "xorqa": "google/IndicGenBench_xorqa_in",
}


@dataclass
class DataConfig:
    split: str = "dev"
    languages: list[str] = field(default_factory=lambda: ["hi"])
    max_samples_per_lang: int | None = None
    cache_dir: str | None = None


@dataclass
class ModelConfig:
    backend: str = "small"
    model_name_or_path: str | None = None
    device: str = "cpu"
    max_new_tokens: int = 128
    torch_dtype: str = "auto"


@dataclass
class RunConfig:
    tasks: list[str] = field(default_factory=lambda: ALL_TASKS.copy())
    output_file: str | None = None
    output_dir: str | None = None
    seed: int = 42
    log_level: str = "INFO"
    save_predictions: bool = False


@dataclass
class BenchmarkConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def resolve_languages(self) -> list[str]:
        if self.data.languages == ["all"]:
            return ALL_LANGUAGES.copy()
        return list(self.data.languages)

    def resolve_max_samples(self) -> int | None:
        if self.data.max_samples_per_lang is not None:
            return self.data.max_samples_per_lang
        if self.data.split == "dev":
            return 20
        return None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> BenchmarkConfig:
    import yaml

    cfg_dict: dict[str, Any] = {
        "data": {"split": "dev", "languages": ["hi"], "max_samples_per_lang": None, "cache_dir": None},
        "model": {"backend": "small", "model_name_or_path": None, "device": "cpu", "max_new_tokens": 128, "torch_dtype": "auto"},
        "run": {"tasks": ALL_TASKS.copy(), "output_file": None, "output_dir": None, "seed": 42, "log_level": "INFO", "save_predictions": False},
    }

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg_dict = _deep_merge(cfg_dict, file_cfg)
        logger.info("Loaded config from %s", config_path)

    if overrides:
        cfg_dict = _deep_merge(cfg_dict, overrides)

    def to_dataclass(d: dict[str, Any], cls: type) -> Any:
        field_names = set(cls.__dataclass_fields__)
        return cls(**{k: d[k] for k in field_names if k in d})

    return BenchmarkConfig(
        data=to_dataclass(cfg_dict.get("data", {}), DataConfig),
        model=to_dataclass(cfg_dict.get("model", {}), ModelConfig),
        run=to_dataclass(cfg_dict.get("run", {}), RunConfig),
    )
