"""
Benchmark decontamination filter.
Removes SFT examples whose prompt hash matches any hash in the benchmark test sets.
Reuses the same normalization/hash functions as the existing
01_sft_data/scripts/decontaminate_against_benchmarks.py for consistency.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from pipeline.filters.base import BaseFilter
from pipeline.config import BenchmarkDecontamConfig
from pipeline.filters.exact_dedup import normalize_for_hash

logger = logging.getLogger(__name__)


def _hash_prompt(record: dict) -> str:
    turns = record.get("conversations", [])
    parts = [normalize_for_hash(t.get("content") or "") for t in turns if t.get("role") == "user"]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _hash_full(record: dict) -> str:
    turns = record.get("conversations", [])
    parts = [
        f"{t.get('role', '')}:{normalize_for_hash(t.get('content') or '')}"
        for t in turns
    ]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def load_hashes_from_file(path: Path) -> set[str]:
    hashes: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            h = line.strip()
            if h:
                hashes.add(h)
    return hashes


class BenchmarkDecontam(BaseFilter):
    name = "benchmark_decontam"

    def __init__(self, cfg: BenchmarkDecontamConfig) -> None:
        self._cfg = cfg
        self._hashes: set[str] = set()
        # Always prompt-mode per plan (hash only user turns to match benchmark questions)
        self._hash_fn = _hash_prompt
        self._load_all_hashes()

    def _load_all_hashes(self) -> None:
        hashes_dir = Path(self._cfg.benchmark_hashes_dir)
        if hashes_dir.is_dir():
            for p in sorted(hashes_dir.iterdir()):
                if p.is_file():
                    loaded = load_hashes_from_file(p)
                    self._hashes |= loaded
                    logger.info("Loaded %d hashes from %s", len(loaded), p.name)

        for fpath in self._cfg.benchmark_hash_files:
            p = Path(fpath)
            if p.exists():
                loaded = load_hashes_from_file(p)
                self._hashes |= loaded
                logger.info("Loaded %d hashes from %s", len(loaded), p.name)

        logger.info("Benchmark decontam: %d total hashes loaded", len(self._hashes))

    def filter(self, record: dict) -> tuple[bool, str]:
        if not self._hashes:
            return True, ""
        h = self._hash_fn(record)
        if h in self._hashes:
            return False, f"benchmark_contamination:hash={h[:16]}"
        return True, ""
