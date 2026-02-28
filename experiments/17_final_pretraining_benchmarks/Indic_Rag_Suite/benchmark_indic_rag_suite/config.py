"""
Benchmark configuration: dataclasses, YAML loading, CLI overrides.
Supports both Indic-Rag-Suite and IndicMSMARCO with correct evaluation protocol.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Indic-Rag-Suite: 18 languages
INDIC_RAG_SUITE_LANGUAGES = [
    "as", "bn", "en", "gu", "hi", "kn", "ks", "mai", "ml", "mni", "mr", "ne", "or", "pa", "sat", "ta", "te", "ur",
]

# IndicMSMARCO: 13 languages (paper benchmark)
INDIC_MSMARCO_LANGUAGES = [
    "as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "ta", "te", "ur",
]

DEFAULT_DATASET = "ai4bharat/Indic-Rag-Suite"
INDIC_MSMARCO_DATASET = "ai4bharat/IndicMSMARCO"


@dataclass
class DataConfig:
    """Data loading configuration."""
    dataset_name: str = DEFAULT_DATASET
    split: str = "dev"
    max_samples_per_lang: int | None = None
    languages: list[str] = field(default_factory=lambda: ["hi"])
    cache_dir: str | None = None
    shard_index: int = 0
    shard_total: int = 1


@dataclass
class ModelConfig:
    """Model backend configuration."""
    retrieval_backend: str = "small"
    generation_backend: str = "small"
    device: str = "cpu"
    retrieval_batch_size: int = 16
    generation_max_new_tokens: int = 64
    retrieval_model_name_or_path: str | None = None
    generation_model_name_or_path: str | None = None


@dataclass
class RunConfig:
    """Run-level configuration."""
    tasks: list[str] = field(default_factory=lambda: ["retrieval", "generation"])
    output_dir: str | None = None
    output_file: str | None = None
    seed: int = 42
    log_level: str = "INFO"
    # Paper uses monolingual retrieval. If True, pool = this lang + other langs (not paper protocol).
    retrieval_add_cross_lang_negatives: bool = False
    # MRR@K cutoff (10 = paper standard for MS MARCO / IndicMSMARCO)
    retrieval_mrr_at_k: int = 10
    # Retrieval: k values for Recall@k and NDCG@k (e.g. [1, 5, 10, 20])
    recall_at_k_list: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    ndcg_at_k_list: list[int] = field(default_factory=lambda: [5, 10])
    # Generation: report token F1 alongside EM
    use_f1: bool = True
    # SQuAD-style normalization for EM/F1 (articles, punctuation removed)
    use_squad_normalize: bool = False
    # Optional: BLEU / ROUGE-L for generation (requires nltk / rouge-score)
    use_bleu: bool = False
    use_rouge: bool = False
    # Save per-sample predictions to a JSON file for debugging
    save_predictions: bool = False
    # Generation evaluator: "default" (EM/F1 only), "ragas" (optional extra metrics)
    generation_evaluator: str = "default"


@dataclass
class BenchmarkConfig:
    """Full benchmark configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def resolve_max_samples(self) -> int | None:
        if self.data.max_samples_per_lang is not None:
            return self.data.max_samples_per_lang
        if self.data.split == "dev":
            if INDIC_MSMARCO_DATASET in self.data.dataset_name:
                return None
            return 20
        return None

    def resolve_languages(self) -> list[str]:
        if self.data.languages == ["all"] or (len(self.data.languages) == 1 and self.data.languages[0] == "all"):
            if INDIC_MSMARCO_DATASET in self.data.dataset_name:
                return INDIC_MSMARCO_LANGUAGES.copy()
            return INDIC_RAG_SUITE_LANGUAGES.copy()
        return list(self.data.languages)

    def is_indic_msmarco(self) -> bool:
        return INDIC_MSMARCO_DATASET in self.data.dataset_name


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
        "data": {
            "dataset_name": DEFAULT_DATASET,
            "split": "dev",
            "max_samples_per_lang": None,
            "languages": ["hi"],
            "cache_dir": None,
            "shard_index": 0,
            "shard_total": 1,
        },
        "model": {
            "retrieval_backend": "small",
            "generation_backend": "small",
            "device": "cpu",
            "retrieval_batch_size": 16,
            "generation_max_new_tokens": 64,
            "retrieval_model_name_or_path": None,
            "generation_model_name_or_path": None,
        },
        "run": {
            "tasks": ["retrieval", "generation"],
            "output_dir": None,
            "output_file": None,
            "seed": 42,
            "log_level": "INFO",
            "retrieval_add_cross_lang_negatives": False,
            "retrieval_mrr_at_k": 10,
            "recall_at_k_list": [1, 5, 10, 20],
            "ndcg_at_k_list": [5, 10],
            "use_f1": True,
            "use_squad_normalize": False,
            "use_bleu": False,
            "use_rouge": False,
            "save_predictions": False,
            "generation_evaluator": "default",
        },
    }

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg_dict = _deep_merge(cfg_dict, file_cfg)
        logger.info("Loaded config from %s", config_path)

    if overrides:
        cfg_dict = _deep_merge(cfg_dict, overrides)

    def to_dataclass(d: dict[str, Any], cls: type) -> Any:
        if hasattr(cls, "__dataclass_fields__"):
            field_names = set(cls.__dataclass_fields__)
            return cls(**{k: d[k] for k in field_names if k in d})
        return d

    return BenchmarkConfig(
        data=to_dataclass(cfg_dict.get("data", {}), DataConfig),
        model=to_dataclass(cfg_dict.get("model", {}), ModelConfig),
        run=to_dataclass(cfg_dict.get("run", {}), RunConfig),
    )
