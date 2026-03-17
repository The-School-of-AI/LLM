"""
Exact deduplication — SHA-256 hash-based.
Reuses the same normalization and hashing logic from
01_sft_data/scripts/decontaminate_against_benchmarks.py for consistency.
"""
from __future__ import annotations

import hashlib

from pipeline.filters.base import BaseFilter
from pipeline.config import ExactDedupConfig


# ---------------------------------------------------------------------------
# Hash helpers (mirrored from decontaminate_against_benchmarks.py)
# ---------------------------------------------------------------------------

def normalize_for_hash(text: str) -> str:
    """Strip and collapse whitespace — identical to the existing scripts."""
    return " ".join((text or "").strip().split())


def _hash_prompt(record: dict) -> str:
    """Hash only user-side content (prompt)."""
    turns = record.get("conversations", [])
    parts = [normalize_for_hash(t.get("content") or "") for t in turns if t.get("role") == "user"]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _hash_full(record: dict) -> str:
    """Hash full conversation (all roles + content)."""
    turns = record.get("conversations", [])
    parts = [
        f"{t.get('role', '')}:{normalize_for_hash(t.get('content') or '')}"
        for t in turns
    ]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

class ExactDedup(BaseFilter):
    name = "exact_dedup"

    def __init__(self, cfg: ExactDedupConfig) -> None:
        self._cfg = cfg
        self._seen: set[str] = set()
        self._hash_fn = _hash_prompt if cfg.hash_mode == "prompt" else _hash_full

    def filter(self, record: dict) -> tuple[bool, str]:
        h = self._hash_fn(record)
        if h in self._seen:
            return False, f"exact_duplicate:{h[:16]}"
        self._seen.add(h)
        return True, ""
