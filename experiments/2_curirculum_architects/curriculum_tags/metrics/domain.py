"""Domain tagging metric."""

import re
from typing import Any, Dict

from curriculum_extractor.core.plugin import MetricPlugin


class DomainMetric(MetricPlugin):
    """Assigns domain tags based on modality and content heuristics.

    Domains are defined in curriculum.yaml:
    - general_web_clean
    - encyclopedic
    - news_nonpolitical
    - technical_docs
    - math_science
    - code_repos
    - dialogue_chat
    - planning_reasoning_curated
    """

    name = "domain"

    # NCERT Mapping
    NCERT_DOMAIN_MAPPING = {
        "physics": "math_science",
        "chemistry": "math_science",
        "biology": "math_science",
        "mathematics": "math_science",
        "history": "encyclopedic",
        "political_science": "encyclopedic",
        "geography": "encyclopedic",
        "economics": "technical_docs",
        "accounting": "technical_docs",
        "english": "general_web_clean",
        "hindi": "general_web_clean",
    }

    # Regex patterns for heuristic classification of general text
    RE_DIALOGUE = re.compile(
        r"^(User|Assistant|System|Human|AI|A|B):|^(Q|A)\.|<\|user\|>|<\|assistant\|>",
        re.IGNORECASE | re.MULTILINE,
    )

    # 1. Dataset Domain Mapping (User-provided)
    DATASET_DOMAIN_MAPPING = {
        "literature": "general_web_clean",
        "web": "general_web_clean",
        "news": "news_nonpolitical",
        "math": "math_science",
        "social": "general_web_clean",
        "science": "math_science",
        "qa": "dialogue_chat",
        "code": "code_repos",
        "instruction": "technical_docs",  # or planning
        "encyclopedia": "encyclopedic",
        "education": "technical_docs",
    }

    # 2. Dolma Source Mapping (Precedence)
    DOLMA_SOURCE_MAPPING = {
        "cc_en_head": "general_web_clean",
        "cc_en_middle": "general_web_clean",
        "stack": "code_repos",
        "arxiv": "math_science",
        "wiki": "encyclopedic",
        "c4": "general_web_clean",
    }

    RE_ENCYCLOPEDIC = re.compile(
        r"\[\d+\]|\[edit\]|Coordinates:|External links|See also|References|Bibliography",
        re.IGNORECASE,
    )

    RE_TECHNICAL = re.compile(
        r"\b(API|SDK|DOI|ISBN|Usage:|Installation:|Arguments:|Returns:)\b",
        re.IGNORECASE,
    )

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Determine domain based on modality signals and text features."""
        print(f"DEBUG: DomainMetric.compute executing for id={sample.get('id')}")

        # --- 0. HIGH PRECEDENCE: Dataset Tags ---
        dataset_domain_tag = (
            sample.get("domain", "").lower() if sample.get("domain") else None
        )
        dataset_source_tag = (
            sample.get("source", "").lower() if sample.get("source") else None
        )

        # A. Dolma Source Mapping (Highest Prio for Dolma-specifics)
        if dataset_source_tag and dataset_source_tag in self.DOLMA_SOURCE_MAPPING:
            return {
                "primary_domain": self.DOLMA_SOURCE_MAPPING[dataset_source_tag],
                "confidence": 1.0,  # High confidence for explicit source
                "reason": f"dolma_source_{dataset_source_tag}",
            }

        # B. Dataset Domain Mapping (Precedence for high-signal tags: math, code, qa)
        HIGH_SIGNAL_DOMAINS = [
            "math",
            "code",
            "qa",
            "science",
            "instruction",
            "encyclopedia",
            "news",
        ]

        if dataset_domain_tag and dataset_domain_tag in self.DATASET_DOMAIN_MAPPING:
            if dataset_domain_tag in HIGH_SIGNAL_DOMAINS:
                return {
                    "primary_domain": self.DATASET_DOMAIN_MAPPING[dataset_domain_tag],
                    "confidence": 1.0,
                    "reason": f"dataset_tag_{dataset_domain_tag}",
                }
            # For generic tags like "web" or "social", we fall through to heuristics
            # unless we want to use them as defaults?
            # Let's use them as strong defaults if heuristics fail.

        # 0. NCERT Override
        if "metadata" in sample:
            meta = sample["metadata"]
            if isinstance(meta, str):
                import json

                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    meta = {}

            if isinstance(meta, dict) and meta.get("source_type") == "textbook":
                dataset = sample.get("dataset", "").lower()
                if dataset == "ncert" or "ncert" in str(sample.get("id", "")).lower():
                    subject = meta.get("subject", "").lower().replace(" ", "_")
                    print(f"DEBUG: Domain Checking Subject: {subject}")
                    if subject in self.NCERT_DOMAIN_MAPPING:
                        return {
                            "primary_domain": self.NCERT_DOMAIN_MAPPING[subject],
                            "confidence": 1.0,
                            "reason": f"ncert_metadata_{subject}",
                        }

        # 1. Get Modality Signals (High Confidence)
        tags = sample.get("curriculum_tags", {})
        modality_tags = tags.get("modality", {})
        primary_modality = modality_tags.get("primary_modality")

        # Also check boolean flags if primary isn't set (robustness)
        has_code = modality_tags.get("has_code", False)
        has_math = modality_tags.get("has_math", False)
        has_agentic = modality_tags.get("has_agentic", False)
        has_research = modality_tags.get("has_research_paper", False)

        # 2. Logic: Modality overrides text heuristics
        primary_domain = None
        confidence = 0.0
        reason = "heuristic"

        if primary_modality == "agentic_traces" or has_agentic:
            primary_domain = "planning_reasoning_curated"
            confidence = 1.0
            reason = "modality_signal"

        elif primary_modality == "code" or (has_code and not has_math):
            # Pure code is likely a repo
            primary_domain = "code_repos"
            confidence = 0.95
            reason = "modality_signal"

        elif primary_modality == "math" or (has_math and not has_code):
            primary_domain = "math_science"
            confidence = 0.9
            reason = "modality_signal"

        elif primary_modality == "research_papers" or has_research:
            # Research papers are usually math/science
            primary_domain = "math_science"
            confidence = 0.9
            reason = "modality_signal"

        elif primary_modality == "technical_text":
            # Could be docs or math/science. Default to technical_docs for now unless it looks like a paper
            if has_research:
                primary_domain = "math_science"
            else:
                primary_domain = "technical_docs"
            confidence = 0.8
            reason = "modality_signal"

        # 3. Fallback: Heuristics for General Text
        if not primary_domain:
            text = sample.get("text", "")

            # Check Dialogue
            if self.RE_DIALOGUE.search(text):
                primary_domain = "dialogue_chat"
                confidence = 0.85

            # Check Encyclopedic
            elif self.RE_ENCYCLOPEDIC.search(text):
                primary_domain = "encyclopedic"
                confidence = 0.8

            # Check Technical Docs (Weak signal)
            elif self.RE_TECHNICAL.search(text):
                primary_domain = "technical_docs"
                confidence = 0.6

            else:
                # Default
                primary_domain = "general_web_clean"
                confidence = 0.5
                reason = "default"

        return {
            "primary_domain": primary_domain,
            "confidence": confidence,
            "reason": reason,
        }
