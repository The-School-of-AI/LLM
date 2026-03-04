"""Entropy metric for detecting noise or high-density text."""

import math
from collections import Counter
from typing import Any, Dict

from ..core.plugin import MetricPlugin


class EntropyMetric(MetricPlugin):
    """Calculates character-level Shannon entropy."""

    name = "entropy"

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Compute character entropy of the text.

        Returns:
            score: Shannon entropy value
        """
        text = sample.get("text", "")

        # Sample first 4000 chars for efficiency
        s = text[:4000]
        if not s:
            return {"score": 0.0}

        freq = Counter(s)
        n = len(s)

        entropy = 0.0
        for count in freq.values():
            p = count / n
            entropy -= p * math.log(p + 1e-12)

        return {"score": round(entropy, 3)}
