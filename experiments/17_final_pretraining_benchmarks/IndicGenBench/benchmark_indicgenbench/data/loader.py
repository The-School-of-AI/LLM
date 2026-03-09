"""Load IndicGenBench datasets from HuggingFace.

Data sources:
- CrossSum-IN: crosssum_english-{lang}_{split}.json  (splits: train/dev/test)
- Flores-IN:   flores_{lang}_en_{split}.json          (splits: dev/test)
- XQuAD-IN:    xquad_{lang}_{split}.json              (splits: train/dev/test)
- XorQA-IN:    xorqa_{lang}_{split}.json              (splits: train/dev/test)

CrossSum and Flores work via HuggingFace `datasets` library (split names: train/validation/test).
XQuAD and XorQA have schema issues with `datasets`, so we download JSON directly via `huggingface_hub`.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from benchmark_indicgenbench.config import HF_DATASETS, XQUAD_LANGUAGES

logger = logging.getLogger(__name__)

# Map our split names to what the dataset uses
_HF_SPLIT_MAP = {"dev": "validation", "test": "test", "train": "train", "validation": "validation"}


def _load_via_datasets(task: str, lang: str, split: str, cache_dir: str | None = None) -> list[dict[str, Any]]:
    """Load crosssum/flores via HuggingFace datasets library. Returns list of example dicts."""
    from datasets import load_dataset

    dataset_name = HF_DATASETS[task]
    hf_split = _HF_SPLIT_MAP.get(split, split)

    try:
        ds = load_dataset(dataset_name, split=hf_split, cache_dir=cache_dir)
    except Exception as e:
        logger.warning("Failed to load %s split=%s: %s", dataset_name, hf_split, e)
        return []

    # Filter by language — each row has examples.lang
    samples = []
    for row in ds:
        ex = row.get("examples", row)
        if isinstance(ex, dict) and ex.get("lang") == lang:
            samples.append(ex)

    return samples


def _load_via_json(task: str, lang: str, split: str, cache_dir: str | None = None) -> list[dict[str, Any]]:
    """Load xquad/xorqa by downloading JSON files directly from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    dataset_name = HF_DATASETS[task]
    filename = f"{task}_{lang}_{split}.json"

    try:
        path = hf_hub_download(dataset_name, filename, repo_type="dataset", cache_dir=cache_dir)
    except Exception as e:
        logger.warning("Failed to download %s/%s: %s", dataset_name, filename, e)
        return []

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("Failed to parse %s: %s", path, e)
        return []

    examples = data.get("examples", [])
    if isinstance(examples, dict):
        examples = [examples]

    return examples


def _load_task_samples(task: str, lang: str, split: str, cache_dir: str | None = None) -> list[dict[str, Any]]:
    """Load samples for a single task/lang/split."""
    if task in ("crosssum", "flores"):
        return _load_via_datasets(task, lang, split, cache_dir)
    else:
        # xquad, xorqa: direct JSON download
        return _load_via_json(task, lang, split, cache_dir)


def load_task_data(
    task: str,
    languages: list[str],
    split: str = "dev",
    max_samples_per_lang: int | None = None,
    cache_dir: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load data for a single task across requested languages.

    Returns: {lang: [sample_dicts]}
    """
    if task == "xquad":
        languages = [l for l in languages if l in XQUAD_LANGUAGES]
        if not languages:
            logger.warning("No requested languages are available for XQuAD-IN")
            return {}

    data_by_lang: dict[str, list[dict[str, Any]]] = {}

    for lang in languages:
        samples = _load_task_samples(task, lang, split, cache_dir)
        if not samples:
            logger.warning("No data loaded for %s/%s/%s", task, lang, split)
            continue

        if max_samples_per_lang and len(samples) > max_samples_per_lang:
            samples = samples[:max_samples_per_lang]

        data_by_lang[lang] = samples
        logger.info("  %s/%s: %d samples", task, lang, len(samples))

    return data_by_lang
