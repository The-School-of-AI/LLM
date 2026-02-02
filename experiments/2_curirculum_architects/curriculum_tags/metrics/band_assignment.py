"""Metric for final band assignment based on aggregated signals."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.plugin import MetricPlugin


@dataclass
class BandThresholds:
    """Thresholds for a specific difficulty band."""
    min_readability_grade: float = 0.0
    min_difficulty_score: float = 0.0
    min_entropy: float = 0.0
    min_diversity: float = 0.0
    allowed_modalities: List[str] = field(default_factory=list)


@dataclass
class BandAssignmentConfig:
    """Configuration for band assignment logic."""
    
    # Mapping from DifficultyMetric levels (L0-L5) to Curriculum Bands
    difficulty_level_map: Dict[str, str] = field(default_factory=lambda: {
        "L0": "B0",
        "L1": "B0",
        "L2": "B1",
        "L3": "B2", # Conservative mapping
        "L4": "B4", # As requested
        "L5": "B5"
    })

    # Default thresholds derived from curriculum.yaml intents
    # These can be overridden by config
    bands: Dict[str, BandThresholds] = field(default_factory=lambda: {
        "B0": BandThresholds(
            min_readability_grade=0.0,
            min_difficulty_score=0.0,
            min_entropy=0.0,
            min_diversity=0.0,
            allowed_modalities=["general_text"]
        ),
        "B1": BandThresholds(
            min_readability_grade=4.0, 
            min_difficulty_score=0.20,
            min_entropy=3.5,
            min_diversity=0.10,
            allowed_modalities=["general_text", "clean_exposition"]
        ),
        "B2": BandThresholds(
            min_readability_grade=8.0, 
            min_difficulty_score=0.35,
            min_entropy=4.0,
            min_diversity=0.15,
            allowed_modalities=["general_text", "structured_knowledge"]
        ),
        "B3": BandThresholds(
            min_readability_grade=12.0, 
            min_difficulty_score=0.55,
            min_entropy=4.5,
            min_diversity=0.20,
            allowed_modalities=["structured_knowledge", "technical_text", "code"]
        ),
        "B4": BandThresholds(
            min_readability_grade=14.0, 
            min_difficulty_score=0.75,
            min_entropy=5.0,
            min_diversity=0.25,
            allowed_modalities=["technical_text", "math", "code", "planning_reasoning_curated"]
        ),
        "B5": BandThresholds(
            min_readability_grade=16.0, 
            min_difficulty_score=0.90,
            min_entropy=5.5,
            min_diversity=0.30,
            allowed_modalities=["hard_reasoning", "math", "advanced_code", "planning"]
        )
    })


class BandAssignmentMetric(MetricPlugin):
    """Assigns final curriculum band based on all available signals."""

    name = "band_assignment"

    def __init__(self, config):
        super().__init__(config)
        self.logic_config = BandAssignmentConfig()
        # TODO: Hydrate logic_config from self.config if needed

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Compute final band assignment.

        Uses a hierarchical decision process:
        1. Specialized Modality Overrides (e.g. Agentic -> B5 target)
        2. Hard Complexity Constraints (Readability, Entropy, Diversity)
        3. Fallback to Content Analysis
        """
        # Get signals from previous metrics
        tags = sample.get("curriculum_tags", {})
        
        modality_tags = tags.get("modality", {})
        readability_tags = tags.get("readability", {})
        difficulty_tags = tags.get("difficulty", {})
        entropy_tags = tags.get("entropy", {})
        diversity_tags = tags.get("diversity", {})
        
        # 1. Extract Core Signals
        has_agentic = modality_tags.get("has_agentic", False)
        has_math = modality_tags.get("has_math", False)
        has_code = modality_tags.get("has_code", False)
        has_research = modality_tags.get("has_research_paper", False)
        
        fk_grade = readability_tags.get("flesch_kincaid_grade", 0.0)
        diff_score = difficulty_tags.get("score", 0.0)
        diff_level = difficulty_tags.get("level", "L0")
        entropy = entropy_tags.get("score", 0.0)
        diversity = diversity_tags.get("rare_ratio", 0.0)
        
        # COT & Agentic signals from scanner
        cot_tags = tags.get("cot_scanner", {})
        has_cot_trace = cot_tags.get("has_cot", False)
        has_agentic_trace = cot_tags.get("has_agentic", False)
        
        # 2. Hard Modality Constraints (Overrides)
        
        # Agentic traces are distinctly B5 (or late B4) per curriculum
        # We check both the regex scanner and the modality metric
        if has_agentic or has_agentic_trace:
            return self._result("B5", "Contains agentic traces")
            
        # Research papers are typically B4+
        if has_research:
            if fk_grade > 16.0 or diff_score > 0.8:
                return self._result("B5", "Complex research paper")
            return self._result("B4", "Research paper")

        # COT Floor: Curriculum forbids COT in B0-B2
        if has_cot_trace:
            # If we have COT, we must be at least B3.
            # We skip the check for lower bands and start potentially at B3
            # But we still allow it to go higher (B4, B5) if other metrics support it.
            pass # Logic handled below by essentially floor-clamping
            
        # 3. Code & Math Logic
        if has_code or has_math:
            if diff_score > 0.8 or diversity > 0.4:
                return self._result("B5", "Advanced technical content")
            elif diff_score > 0.6:
                return self._result("B4", "Technical content")
            elif diff_score > 0.4:
                return self._result("B3", "Standard code/math")
            else:
                return self._result("B2", "Introductory code/math")

        # 4. General Text / Structured Knowledge Flow
        # Use Difficulty Level Mapping as the primary constraint
        mapped_band = self.logic_config.difficulty_level_map.get(diff_level, "B0")
        
        target_bands = ["B5", "B4", "B3", "B2", "B1", "B0"]
        start_index = target_bands.index(mapped_band)
        
        # Helper to check secondary thresholds (Readability, Entropy, Diversity)
        # Note: We ignore difficulty_score here because we trust the Level map
        def meets_secondary_thresholds(band_name: str) -> bool:
            thresholds = self.logic_config.bands[band_name]
            return (
                fk_grade >= thresholds.min_readability_grade and
                entropy >= thresholds.min_entropy and
                diversity >= thresholds.min_diversity
            )

        for band in target_bands[start_index:]:
            if meets_secondary_thresholds(band):
                # We found a candidate band based on complexity
                
                # ENFORCE FLOORS
                # If has COT, cannot be below B3
                if has_cot_trace and band in ["B0", "B1", "B2"]:
                     return self._result("B3", "COT trace forces min B3")
                     
                return self._result(band, f"Mapped from {diff_level} + validated stats")
            
        # Default B0
        if has_cot_trace:
             return self._result("B3", "COT trace forces min B3 (fallback)")
             
        return self._result("B0", "Baseline complexity")

    def _result(self, band: str, reason: str) -> Dict[str, Any]:
        """Format the output result."""
        return {
            "band": band,
            "reason": reason
        }
