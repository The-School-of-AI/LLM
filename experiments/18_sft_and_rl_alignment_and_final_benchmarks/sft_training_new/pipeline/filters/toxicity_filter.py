"""
Toxicity filter — drops records containing toxic content.
Two backends:
  - keyword: Fast blocklist-based (no model, no GPU).
  - detoxify: ML-based scoring (requires ``pip install detoxify``).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from pipeline.filters.base import BaseFilter
from pipeline.config import ToxicityFilterConfig

logger = logging.getLogger(__name__)


def _get_role_texts(record: dict, roles: list[str]) -> list[str]:
    return [
        t.get("content") or ""
        for t in record.get("conversations", [])
        if t.get("role") in roles
    ]


class ToxicityFilter(BaseFilter):
    name = "toxicity_filter"

    def __init__(self, cfg: ToxicityFilterConfig) -> None:
        self._cfg = cfg
        self._keywords: list[re.Pattern] = []
        self._detoxify_model = None

        if cfg.backend == "keyword":
            self._load_keywords(cfg.keyword_blocklist_path)
        elif cfg.backend == "detoxify":
            self._load_detoxify(cfg.detoxify_model)
        else:
            raise ValueError(f"Unknown toxicity_filter backend: {cfg.backend}")

    def _load_keywords(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            logger.warning("Toxicity blocklist not found: %s — filter will pass all records", p)
            return
        with open(p, encoding="utf-8") as f:
            for line in f:
                kw = line.strip()
                if kw and not kw.startswith("#"):
                    self._keywords.append(re.compile(re.escape(kw), re.IGNORECASE))

    def _load_detoxify(self, model_name: str) -> None:
        try:
            from detoxify import Detoxify
            self._detoxify_model = Detoxify(model_name)
        except ImportError:
            raise ImportError("detoxify is required: pip install detoxify")

    def filter(self, record: dict) -> tuple[bool, str]:
        texts = _get_role_texts(record, self._cfg.check_roles)
        combined = " ".join(texts)
        if not combined.strip():
            return True, ""

        if self._cfg.backend == "keyword":
            for pat in self._keywords:
                if pat.search(combined):
                    return False, f"toxicity:keyword_match:{pat.pattern[:40]}"
            return True, ""

        if self._cfg.backend == "detoxify":
            results = self._detoxify_model.predict(combined)
            toxicity_score = results.get("toxicity", 0.0)
            if toxicity_score > self._cfg.max_score:
                return False, f"toxicity:score={toxicity_score:.3f}>{self._cfg.max_score}"
            return True, ""

        return True, ""
