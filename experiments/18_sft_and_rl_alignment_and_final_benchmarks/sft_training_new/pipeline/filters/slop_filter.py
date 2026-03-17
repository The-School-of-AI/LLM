"""
Slop / verbosity filter — drops responses containing too many low-quality filler phrases.
Pattern list is loaded from config (one phrase per line, case-insensitive).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from pipeline.filters.base import BaseFilter
from pipeline.config import SlopFilterConfig

logger = logging.getLogger(__name__)


class SlopFilter(BaseFilter):
    name = "slop_filter"

    def __init__(self, cfg: SlopFilterConfig) -> None:
        self._cfg = cfg
        self._patterns: list[re.Pattern] = []
        self._load_patterns(cfg.slop_patterns_path)

    def _load_patterns(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            logger.warning("Slop patterns file not found: %s — slop_filter will pass all records", p)
            return
        with open(p, encoding="utf-8") as f:
            for line in f:
                phrase = line.strip()
                if phrase and not phrase.startswith("#"):
                    self._patterns.append(re.compile(re.escape(phrase), re.IGNORECASE))

    def filter(self, record: dict) -> tuple[bool, str]:
        if not self._patterns:
            return True, ""

        turns = record.get("conversations", [])
        for turn in turns:
            if turn.get("role") not in self._cfg.check_roles:
                continue
            text = turn.get("content") or ""
            if not text.strip():
                continue

            word_count = max(1, len(text.split()))
            slop_words = 0
            for pat in self._patterns:
                m = pat.search(text)
                if m:
                    slop_words += len(m.group(0).split())

            ratio = slop_words / word_count
            if ratio > self._cfg.max_slop_ratio:
                return False, f"slop:ratio={ratio:.3f}>{self._cfg.max_slop_ratio}"

        return True, ""
