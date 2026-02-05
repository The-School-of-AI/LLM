"""Modality detection metric."""

import re
from typing import Any, Dict

from curriculum_extractor.core.plugin import MetricPlugin


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
        r"```|def\s+\w+\(|class\s+\w+|function\s+\w+|import\s+\w+",
        re.IGNORECASE | re.MULTILINE,
    )
    MATH_PATTERN = re.compile(
        r"[∑∫√≈≠≤≥→∞]|\\(frac|sum|int|sqrt|begin\{equation\})", re.IGNORECASE
    )

    # Merged CoT Patterns from cot_scanner.py
    # Removed single-word connectives (therefore, thus, hence) to ensure high precision
    REASONING_PATTERN = re.compile(
        r"let's think step by step|"
        r"reasoning:|chain of thought:|thinking process:|explanation:",
        re.IGNORECASE,
    )

    # Merged Agentic Patterns from cot_scanner.py
    AGENTIC_PATTERN = re.compile(
        r"Action:|Observation:|Thought:|Final Answer:|Tool:|"
        + r'"(tool|action|observation|thought)"\s*:',
        re.IGNORECASE,
    )

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
            if isinstance(meta, dict) and meta.get("source_type") == "textbook": # Heuristic for NCERT or similar
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
