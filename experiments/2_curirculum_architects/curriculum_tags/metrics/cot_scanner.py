"""Chain of Thought (CoT) Scanner Metric."""

import re
from typing import Any, Dict

from ..core.plugin import MetricPlugin


class COTScannerMetric(MetricPlugin):
    """Scan for Chain of Thought reasoning patterns and agentic traces."""

    name = "cot_scanner"

    # Patterns identifying CoT/Reasoning
    COT_PATTERNS = [
        r"let's think step by step",
        r"reasoning:",
        r"chain of thought:",
        r"thinking process:",
        r"explanation:",
    ]
    
    # Patterns identifying Agentic traces
    AGENTIC_PATTERNS = [
        r"Action:",
        r"Observation:",
        r"Thought:",
        r"Final Answer:",
        r"Tool:",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.cot_regex = re.compile("|".join(self.COT_PATTERNS), re.IGNORECASE)
        self.agentic_regex = re.compile("|".join(self.AGENTIC_PATTERNS), re.IGNORECASE)

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Compute CoT and Agentic metrics."""
        text = sample.get("text", "")
        
        has_cot = bool(self.cot_regex.search(text))
        has_agentic = bool(self.agentic_regex.search(text))
        
        # Estimate token lengths (rough approx: chars / 4)
        # In a real scenario, we might want to use the tokenizer if available, 
        # but plugins are independent. If TokenizerMetric ran, maybe we can use its count?
        # But this metric might run before or after.
        # For "cot_token_length", we'd ideally extract the CoT section. 
        # Since we are just scanning presence, we can't easily measure *just* the CoT length 
        # without parsing.
        # For now, we will return boolean flags and density estimates.
        
        return {
            "has_cot": has_cot,
            "has_agentic": has_agentic,
            "cot_density": self._count_matches(text, self.cot_regex) / max(1, len(text.split())),
            "agentic_density": self._count_matches(text, self.agentic_regex) / max(1, len(text.split()))
        }

    def _count_matches(self, text: str, regex: re.Pattern) -> int:
        return len(regex.findall(text))
