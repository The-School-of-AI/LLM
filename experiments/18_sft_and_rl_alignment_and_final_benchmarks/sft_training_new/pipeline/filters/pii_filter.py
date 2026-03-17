"""
PII filter — detects and optionally redacts personally identifiable information.
Two backends:
  - regex: Fast, no model. Patterns configurable in YAML.
  - presidio: ML-based NER (requires ``pip install presidio-analyzer presidio-anonymizer``).
"""
from __future__ import annotations

import logging
import re

from pipeline.filters.base import BaseFilter
from pipeline.config import PIIFilterConfig

logger = logging.getLogger(__name__)

_REDACT_PLACEHOLDER = "[REDACTED]"


def _get_role_texts(record: dict, roles: list[str]) -> dict[str, list[int]]:
    """Return mapping of turn index → role for the specified roles."""
    result = {}
    for i, t in enumerate(record.get("conversations", [])):
        if t.get("role") in roles:
            result[i] = t.get("role")
    return result


class PIIFilter(BaseFilter):
    name = "pii_filter"

    def __init__(self, cfg: PIIFilterConfig) -> None:
        self._cfg = cfg
        self._patterns: dict[str, re.Pattern] = {}
        self._analyzer = None
        self._anonymizer = None

        if cfg.backend == "regex":
            for name, pattern in cfg.regex_patterns.items():
                self._patterns[name] = re.compile(pattern, re.IGNORECASE)
        elif cfg.backend == "presidio":
            self._load_presidio(cfg.presidio_entities)
        else:
            raise ValueError(f"Unknown pii_filter backend: {cfg.backend}")

    def _load_presidio(self, entities: list[str]) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            self._entities = entities
        except ImportError:
            raise ImportError(
                "presidio is required: pip install presidio-analyzer presidio-anonymizer spacy && "
                "python -m spacy download en_core_web_lg"
            )

    def filter(self, record: dict) -> tuple[bool, str]:
        conversations = record.get("conversations", [])
        detected = False
        pii_label = ""

        for i, turn in enumerate(conversations):
            if turn.get("role") not in self._cfg.check_roles:
                continue
            text = turn.get("content") or ""
            if not text:
                continue

            found, label, cleaned = self._check(text)
            if found:
                if self._cfg.action == "drop":
                    return False, f"pii:{label}"
                elif self._cfg.action == "redact":
                    turn["content"] = cleaned
                    detected = True
                    pii_label = label

        if detected:
            # Record was redacted; keep it with metadata
            record["_pii_redacted"] = True
        return True, ""

    def _check(self, text: str) -> tuple[bool, str, str]:
        """Return (found, label, cleaned_text)."""
        if self._cfg.backend == "regex":
            cleaned = text
            for name, pat in self._patterns.items():
                if pat.search(text):
                    cleaned = pat.sub(_REDACT_PLACEHOLDER, cleaned)
                    return True, name, cleaned
            return False, "", text

        if self._cfg.backend == "presidio":
            results = self._analyzer.analyze(text=text, entities=self._entities, language="en")
            if not results:
                return False, "", text
            from presidio_anonymizer.entities import RecognizerResult, OperatorConfig
            anonymized = self._anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators={"DEFAULT": OperatorConfig("replace", {"new_value": _REDACT_PLACEHOLDER})},
            )
            label = ",".join(r.entity_type for r in results[:3])
            return True, label, anonymized.text

        return False, "", text
