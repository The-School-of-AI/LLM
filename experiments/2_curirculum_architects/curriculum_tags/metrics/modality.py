"""Modality detection metric."""

import re
from typing import Any, Dict

from ..core.plugin import MetricPlugin


class ModalityMetric(MetricPlugin):
    """Detect content modalities (code, math, reasoning, etc.).

    Can use difficulty band from previous plugins to adjust detection.
    """

    name = "modality"

    # Regex patterns
    CODE_PATTERN = re.compile(
        r"```|def\s+\w+\(|class\s+\w+|function\s+\w+|import\s+\w+",
        re.IGNORECASE | re.MULTILINE,
    )
    MATH_PATTERN = re.compile(r"[∑∫√≈≠≤≥→∞]|\\(frac|sum|int|sqrt|begin\{equation\})", re.IGNORECASE)
    REASONING_PATTERN = re.compile(r"let's think step by step|therefore|thus|hence|reasoning:", re.IGNORECASE)
    AGENTIC_PATTERN = re.compile(r'"(tool|action|observation|thought)"\s*:|Observation:|Action:', re.IGNORECASE)
    RE_RESEARCH_PAPER = re.compile(
        r"\bAbstract[:\s]|"
        r"\bReferences[:\s]|"
        r"\b(?:arXiv|doi):\s*\d|"
        r"\bet al\.|"
        r"\[[\d,\s]+\].*\[[\d,\s]+\]",
        re.IGNORECASE,
    )

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Detect modalities in text.

        Can access previous plugin results from sample['curriculum_tags'].

        Returns:
            has_code: Code presence
            has_math: Mathematical notation
            has_reasoning: Chain-of-thought reasoning
            has_agentic: Tool/agent traces
            has_research_paper: Research paper features
            primary_modality: Dominant modality
        """
        text = sample.get("text", "")

        # Detect each modality
        has_code = bool(self.CODE_PATTERN.search(text))
        has_math = bool(self.MATH_PATTERN.search(text))
        has_reasoning = bool(self.REASONING_PATTERN.search(text))
        has_agentic = bool(self.AGENTIC_PATTERN.search(text))
        has_research_paper = bool(self.RE_RESEARCH_PAPER.search(text))

        # Determine primary modality
        primary = "general_text"
        if has_agentic:
            primary = "agentic_traces"
        elif has_research_paper:
            primary = "research_papers"
        elif has_code and has_math:
            primary = "technical_text"
        elif has_code:
            primary = "code"
        elif has_math:
            primary = "math"
        elif has_reasoning:
            primary = "reasoning"

        return {
            "has_code": has_code,
            "has_math": has_math,
            "has_reasoning": has_reasoning,
            "has_agentic": has_agentic,
            "has_research_paper": has_research_paper,
            "primary_modality": primary,
        }
