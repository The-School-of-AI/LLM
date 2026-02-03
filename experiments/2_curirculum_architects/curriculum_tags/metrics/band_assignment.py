"""Metric for final band assignment based on aggregated signals."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..core.plugin import MetricPlugin


@dataclass
class BandConstraints:
    """Constraints for a specific difficulty band."""

    # Inclusive ranges (min, max)
    readability_range: Tuple[float, float] = (0.0, float("inf"))
    difficulty_score_range: Tuple[float, float] = (0.0, float("inf"))
    entropy_range: Tuple[float, float] = (0.0, float("inf"))
    diversity_range: Tuple[float, float] = (0.0, float("inf"))
    structural_density_range: Tuple[float, float] = (0.0, float("inf"))

    # Allowed inputs
    allowed_difficulty_levels: List[str] = field(default_factory=list)
    allowed_tokenizer_levels: List[str] = field(default_factory=list)
    allowed_modalities: List[str] = field(default_factory=list)


@dataclass
class BandAssignmentConfig:
    """Configuration for band assignment logic."""

    overlap_policy: str = "highest"
    bands: Dict[str, BandConstraints] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str):
        """Load from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        bands = {}
        for b_name, b_data in data.get("bands", {}).items():
            bands[b_name] = BandConstraints(
                allowed_difficulty_levels=b_data.get("allowed_difficulty_levels", []),
                allowed_tokenizer_levels=b_data.get("allowed_tokenizer_levels", []),
                readability_range=tuple(
                    b_data.get("readability_range", (0, float("inf")))
                ),
                difficulty_score_range=tuple(
                    b_data.get("difficulty_score_range", (0, float("inf")))
                ),
                entropy_range=tuple(b_data.get("entropy_range", (0, float("inf")))),
                diversity_range=tuple(b_data.get("diversity_range", (0, float("inf")))),
                structural_density_range=tuple(
                    b_data.get("structural_density_range", (0, float("inf")))
                ),
                allowed_modalities=b_data.get("allowed_modalities", []),
            )

        return cls(overlap_policy=data.get("overlap_policy", "highest"), bands=bands)


class BandAssignmentMetric(MetricPlugin):
    """Assign final curriculum band based on aggregated metric signals and constraints."""

    name = "band_assignment"

    def __init__(self, config):
        # We need to preserve the passed config
        super().__init__(config)

        # Priority 1: Config passed via constructor (e.g. from tests or explicit override)
        # Note: In tests, we often pass a dict directly as config.
        # Ideally, we should check if `band_assignment_config` or similar key exists,
        # or if we are just testing with an empty config.

        # But wait, the standard Plugin architecture usually passes the whole config dict for that plugin instance.
        # So we should look for keys inside `self.config`?
        # Actually, let's keep it simple:
        # If the file exists on disk, we load it as the BASE defaults.
        # Then, if specific overrides are in self.config, we could apply them.

        logic_config = BandAssignmentConfig()

        # 1. Load from YAML if available (Next to curriculum.yaml)
        if hasattr(self.config, "path") and self.config.path:
            config_dir = Path(self.config.path).parent
            yaml_path = config_dir / "band_assignment.yaml"

            if yaml_path.exists():
                try:
                    logic_config = BandAssignmentConfig.from_yaml(str(yaml_path))
                except Exception as e:
                    print(f"Error loading band_assignment.yaml: {e}")
                    pass

        # 2. Allow programmatic override (e.g. from tests) doesn't easily map to dataclass yet
        # unless strict mapping.
        # Since tests rely on DEFAULT behavior mostly, the YAML load should be fine.
        # The FAILURES in tests are likely due to missing implicit behavior or strict constraints?

        self.logic_config = logic_config

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Compute the final band assignment."""
        # text = sample.get("text", "")
        tags = sample.get("curriculum_tags", {})

        # 1. Extract Signals
        modality_tags = tags.get("modality", {})
        primary_modality = modality_tags.get("primary_modality", "general_text")

        difficulty_tags = tags.get("difficulty", {})
        readability_tags = tags.get("readability", {})
        entropy_tags = tags.get("entropy", {})
        diversity_tags = tags.get("diversity", {})
        diversity_tags = tags.get("diversity", {})
        tokenizer_tags = tags.get("tokenizer_difficulty", {})
        structural_tags = tags.get("structural_density", {})

        fk_grade = readability_tags.get("flesch_kincaid_grade", 0.0)
        diff_score = difficulty_tags.get("score", 0.0)
        diff_level = difficulty_tags.get("level", "L0")
        entropy = entropy_tags.get("score", 0.0)
        diversity = diversity_tags.get("rare_ratio")

        # Fallback for diversity: read from difficulty features if dedicated metric is disabled
        if diversity is None:
            diversity = difficulty_tags.get("features", {}).get("rare_ratio", 0.0)

        tokenizer_level = tokenizer_tags.get("level", "T0")
        structural_density = structural_tags.get("structural_density", 0.0)

        # Extract consolidated reasoning signals from modality
        has_cot_trace = modality_tags.get("has_cot", False) or modality_tags.get(
            "has_reasoning", False
        )
        has_agentic_trace = modality_tags.get("has_agentic", False)

        # Legacy Modality Signals (for compatibility)
        has_agentic = modality_tags.get("has_agentic", False)
        has_math = modality_tags.get("has_math", False)
        has_code = modality_tags.get("has_code", False)
        has_research = modality_tags.get("has_research_paper", False)

        # 2. Hard Overrides (Modality & Safety)
        if has_agentic or has_agentic_trace or primary_modality == "agentic_traces":
            return self._result("B5", "Agentic Override")

        if has_research or primary_modality == "research_papers":
            if fk_grade > 16.0 or diff_score > 0.8:
                return self._result("B5", "Complex Research Paper")
            return self._result("B4", "Research Paper")

        # 3. Code & Math Logic (Special Progression)
        if has_code or has_math:
            if diff_score > 0.8 or diversity > 0.4:
                return self._result("B5", "Advanced technical content")
            elif diff_score > 0.6:
                return self._result("B4", "Technical content")
            elif diff_score > 0.4:
                return self._result("B3", "Standard code/math")
            else:
                return self._result("B2", "Introductory code/math")

        # 4. Constraint-Based Classification
        candidates = []

        def in_range(val, r):
            return r[0] <= val <= r[1]

        for band_name, constraints in self.logic_config.bands.items():

            # A. Level Check
            if diff_level not in constraints.allowed_difficulty_levels:
                continue

            # B. Modality Check (If defined)
            # If primary_modality is present, check if allowed in this band
            if (
                constraints.allowed_modalities
                and primary_modality not in constraints.allowed_modalities
            ):
                # Be careful: mismatching modality is a hard fail for that band
                continue

            # C. Metric Ranges
            # We allow 'soft' failures but for now let's be strict
            if not in_range(fk_grade, constraints.readability_range):
                continue
            if not in_range(diff_score, constraints.difficulty_score_range):
                continue
            if not in_range(entropy, constraints.entropy_range):
                continue
            if not in_range(diversity, constraints.diversity_range):
                continue
            if not in_range(structural_density, constraints.structural_density_range):
                continue

            # D. Tokenizer Level Check
            if (
                constraints.allowed_tokenizer_levels
                and tokenizer_level not in constraints.allowed_tokenizer_levels
            ):
                continue

            # C. COT Floor Check
            # If we detect COT, we disqualify B0, B1, B2 implicitly
            # (Or explicitly: constraints must allow reasoning?)
            # Simplified: If has_cot, we only accept >= B3
            if has_cot_trace and band_name in ["B0", "B1", "B2"]:
                continue

            candidates.append(band_name)

        # 4. Policy Resolution
        if not candidates:
            # Fallback: Nearest based on difficulty level?
            # Or safe default B0.
            # If we have COT but no candidates (e.g. extremely low complexity COT?), force B3
            if has_cot_trace:
                return self._result("B3", "Fallback (COT detected)")
            return self._result("B0", "Fallback (No constraints met)")

        # Sort candidates B0..B5
        band_order = ["B0", "B1", "B2", "B3", "B4", "B5"]
        sorted_candidates = sorted(candidates, key=lambda b: band_order.index(b))

        if self.logic_config.overlap_policy == "highest":
            selected = sorted_candidates[-1]
        elif self.logic_config.overlap_policy == "lowest":
            selected = sorted_candidates[0]
        else:
            selected = sorted_candidates[-1]  # Default highest

        return self._result(selected, f"Constraints met: {candidates}")

    def _result(self, band: str, reason: str) -> Dict[str, Any]:
        """Format the output result."""
        return {"band": band, "reason": reason}
