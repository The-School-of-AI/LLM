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
    human_score: float
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
