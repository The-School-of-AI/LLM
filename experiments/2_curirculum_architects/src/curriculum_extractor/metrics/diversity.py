"""Diversity metric based on rare token usage."""

import re
from collections import Counter
from typing import Any, Dict

from ..core.plugin import MetricPlugin


class DiversityMetric(MetricPlugin):
    """Calculates vocabulary diversity using rare token ratio."""

    name = "diversity"

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Compute rare ratio (tokens appearing once / total tokens).

        Returns:
            rare_ratio: Fraction of unique tokens
            token_count: Total tokens considered
        """
        text = sample.get("text", "")
        tokens = self._tokenize(text)

        if not tokens:
            return {"rare_ratio": 0.0, "token_count": 0}

        freq = Counter(tokens)
        rare_count = sum(1 for t in tokens if freq[t] == 1)
        rare_ratio = rare_count / len(tokens)

        return {"rare_ratio": round(rare_ratio, 3), "token_count": len(tokens)}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple word tokenization."""
        return re.findall(r"\b[\w']+\b", text.lower())
