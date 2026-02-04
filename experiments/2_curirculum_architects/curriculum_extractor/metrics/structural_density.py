"""Structural density metric."""

import re
from typing import Any, Dict

from ..core.plugin import MetricPlugin


class StructuralDensityMetric(MetricPlugin):
    """Calculate structural and symbolic density of text."""

    name = "structural_density"

    # Regex patterns
    STRUCTURAL_PATTERN = re.compile(r"([{}<>\[\]`#*]|\n\s{2,})")
    SYMBOLIC_PATTERN = re.compile(r"[∑∫√≈≠≤≥→∞=+\-*/^|&!]")

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Compute structural and symbolic density."""
        text = sample.get("text", "")
        if not text:
            return {"structural_density": 0.0, "symbolic_density": 0.0, "char_count": 0}

        char_count = len(text)

        # Count structural chars
        structural_matches = self.STRUCTURAL_PATTERN.findall(text)
        structural_count = sum(len(m) for m in structural_matches)

        # Count symbolic chars
        symbolic_matches = self.SYMBOLIC_PATTERN.findall(text)
        symbolic_count = sum(len(m) for m in symbolic_matches)

        return {
            "structural_density": round(structural_count / char_count, 4),
            "symbolic_density": round(symbolic_count / char_count, 4),
            "char_count": char_count,
        }
