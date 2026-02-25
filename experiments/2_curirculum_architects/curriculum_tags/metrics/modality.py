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
    RE_MATH = re.compile(
        r"\\frac|\\sum|\\int|\^\{|\\alpha|\\beta|\\gamma|\\theta|= \d",
        re.IGNORECASE | re.MULTILINE,
    )

    # NCERT Mapping
    NCERT_SUBJECT_MAPPING = {
        "physics": "technical_text",
        "chemistry": "technical_text",
        "biology": "technical_text",
        "mathematics": "math",
        "history": "structured_knowledge",
        "political_science": "structured_knowledge",
        "geography": "structured_knowledge",
        "economics": "technical_text",
        "accounting": "technical_text",
        "english": "general_text",
        "hindi": "general_text",
    }
    CODE_PATTERN = re.compile(
        r"```|"
        r"def\s+\w+\(|"
        r"class\s+\w+\s*[:{]|"  # Class followed by : or { (Java/C++/Python simple)
        r"class\s+\w+\([^\)]+\)\s*:|"  # Python class with inheritance: class Foo(Bar):
        r"^\s*function\s+\w+\s*[\({]|"  # JS function at start of line
        r"^\s*import\s+\w+|"  # Import at start of line
        r"from\s+[\w.]+\s+import\s+\w+|"  # Python from ... import (supports .module)
        r"from\s+\.\s+import\s+\w+",  # Python from . import
        re.IGNORECASE | re.MULTILINE,
    )
    MATH_PATTERN = re.compile(
        r"[∑∫√≈≠≤≥∞]|"  # Removed arrow →
        r"\\("
        r"frac|sum|int|sqrt|begin\{equation\}|"
        r"alpha|beta|gamma|delta|theta|pi|sigma|omega|phi|"
        r"partial|cdot|times|pm"
        r")|"
        r"\\\[|\\\(",
        re.IGNORECASE,
    )

    # Merged CoT Patterns from cot_scanner.py
    # Removed single-word connectives (therefore, thus, hence) to ensure high precision
    REASONING_PATTERN = re.compile(
        r"let's think step by step|"
        r"^\s*(Reasoning|Chain of [Tt]hought|Thinking [Pp]rocess|Explanation):",
        re.IGNORECASE | re.MULTILINE,
    )

    # Merged Agentic Patterns from cot_scanner.py
    AGENTIC_PATTERN = re.compile(
        r"^\s*(Action|Observation|Thought|Final Answer|Tool):|"  # Start of line
        r'"(tool|action|observation|thought)"\s*:',  # JSON key
        re.IGNORECASE | re.MULTILINE,
    )

    RE_RESEARCH_PAPER = re.compile(
        r"^\s*(?:Abstract|References|Bibliography)(?:[:\n]|$)"  # Header followed by colon, newline, or end of line
        r"|\b(?:arXiv|doi)[:/]\s*\d"  # arXiv/doi with colon or slash
        r"|\bdoi\.org/10\."  # doi.org
        r"|\bet al\."  # et al.
        r"|\[[\d,\s]+\].*\[[\d,\s]+\]",  # Multiple Citations
        re.IGNORECASE | re.MULTILINE,
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

            # Additional features for CoT/Agentic
            has_cot: Alias for has_reasoning (for compatibility)
            cot_density: Density of CoT patterns
            agentic_density: Density of Agentic patterns
        """
        if "metadata" in sample:
            meta = sample["metadata"]
            if isinstance(meta, str):
                import json

                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    meta = {}

            # Check for NCERT specific logic
            if (
                isinstance(meta, dict) and meta.get("source_type") == "textbook"
            ):  # Heuristic for NCERT or similar
                # Or check dataset name if available in sample
                dataset = sample.get("dataset", "").lower()
                if dataset == "ncert" or "ncert" in str(sample.get("id", "")).lower():
                    subject = meta.get("subject", "").lower().replace(" ", "_")
                    if subject in self.NCERT_SUBJECT_MAPPING:
                        mapped_mod = self.NCERT_SUBJECT_MAPPING[subject]
                        # Return immediately with high confidence? Or merge?
                        # Let's compute others as fallback but prioritize this mapping
                        return {
                            "has_code": False,
                            "has_math": mapped_mod == "math",
                            "has_reasoning": False,
                            "has_agentic": False,
                            "has_research_paper": False,
                            "primary_modality": mapped_mod,
                        }

        text = sample.get("text", "")

        # Detect each modality
        has_code = bool(self.CODE_PATTERN.search(text))
        has_math = bool(self.MATH_PATTERN.search(text))
        has_reasoning = bool(self.REASONING_PATTERN.search(text))
        has_agentic = bool(self.AGENTIC_PATTERN.search(text))
        has_research_paper = bool(self.RE_RESEARCH_PAPER.search(text))

        # Calculate densities (word count based normalization)
        word_count = max(1, len(text.split()))
        cot_matches = len(self.REASONING_PATTERN.findall(text))
        agentic_matches = len(self.AGENTIC_PATTERN.findall(text))

        cot_density = cot_matches / word_count
        agentic_density = agentic_matches / word_count

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
            # Enhanced outputs
            "has_cot": has_reasoning,  # Alias
            "cot_density": round(cot_density, 6),
            "agentic_density": round(agentic_density, 6),
        }

        # Apply Tag Overrides
        return self._check_dataset_tags(sample, result)  # noqa: F821

    def _check_dataset_tags(self, sample, result):
        """Refine modality based on explicit dataset tags."""
        domain_tag = sample.get("domain", "").lower()
        source_tag = sample.get("source", "").lower()

        # Code
        if domain_tag == "code" or source_tag == "stack":
            result["has_code"] = True
            result["primary_modality"] = "code"

        # Math
        if domain_tag == "math":
            result["has_math"] = True
            if result["primary_modality"] not in ["code"]:
                result["primary_modality"] = "math"

        # Research
        if source_tag == "arxiv":
            result["has_research_paper"] = True
            result["primary_modality"] = "research_papers"

        return result
