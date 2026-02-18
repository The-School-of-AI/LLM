"""Domain mismatch analysis.

This file started as an exploratory script and is kept as a test so it can be
run in CI when the optional sample dataset is available.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest
from src.core.types import DifficultyBand, difficulty_band_order
from src.curriculum.loader import CurriculumLoader

_CHUNKS_FILE = Path("data/datasets/large_sample_chunks.jsonl")

if not _CHUNKS_FILE.exists():
    pytest.skip(
        f"Optional sample dataset not found: {_CHUNKS_FILE}. Skipping domain mismatch check.",
        allow_module_level=True,
    )


def test_domain_mismatch_smoke():
    loader = CurriculumLoader("config/curriculum.yaml")
    ok, errors = loader.load()
    assert ok, f"Curriculum failed to load: {errors}"

    domains_by_band = defaultdict(set)
    with open(_CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            band = chunk.get("band")
            domain = chunk.get("domain")
            domains_by_band[band].add(domain)

    # Minimal sanity: each band defined in curriculum should have allowed_domains.
    for band_name in difficulty_band_order():
        band_def = loader.bands.get(DifficultyBand(band_name))
        if band_def:
            assert isinstance(band_def.allowed_domains, list)
