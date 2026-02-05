"""Metric for final band assignment based on aggregated signals."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from curriculum_extractor.core.plugin import MetricPlugin


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
                    print(f"DEBUG: Loaded band config from {yaml_path}")
                except Exception as e:
                    print(f"DEBUG: Error loading band_assignment.yaml: {e}")
                    pass
            else:
                print(f"DEBUG: band_assignment.yaml not found at {yaml_path}")

        # 2. Allow programmatic override (e.g. from tests) doesn't easily map to dataclass yet
        # unless strict mapping.
        # Since tests rely on DEFAULT behavior mostly, the YAML load should be fine.
        # The FAILURES in tests are likely due to missing implicit behavior or strict constraints?

        self.logic_config = logic_config

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Compute the final band assignment."""
        print("DEBUG: BandAssignmentMetric.compute called")
        # text = sample.get("text", "")
        tags = sample.get("curriculum_tags", {})
        print(f"DEBUG: Curriculum Tags Keys: {list(tags.keys())}")
        if tags:
            print(f"DEBUG: Difficulty Tag Sample: {tags.get('difficulty')}")
            print(f"DEBUG: Modality Tag Sample: {tags.get('modality')}")

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

        # Extract Domain Signals
        domain_tags = tags.get("domain", {})
        primary_domain = domain_tags.get("primary_domain", "Unknown")

        # Legacy Modality Signals (for compatibility)
        has_agentic = modality_tags.get("has_agentic", False)
        has_math = modality_tags.get("has_math", False)
        has_code = modality_tags.get("has_code", False)
        has_research = modality_tags.get("has_research_paper", False)

        # Selection state
        selected_band = None
        selection_reason = "default"

        # 2. Hard Overrides (Modality & Safety)
        if has_agentic or has_agentic_trace or primary_modality == "agentic_traces":
            selected_band = "B5"
            selection_reason = "Agentic Override"

        elif has_research or primary_modality == "research_papers":
            if fk_grade > 16.0 or diff_score > 0.8:
                selected_band = "B5"
                selection_reason = "Complex Research Paper"
            else:
                selected_band = "B4"
                selection_reason = "Research Paper"

        # 3. Code & Math Logic (Special Progression)
        elif has_code or has_math:
            if diff_score > 0.8 or diversity > 0.4:
                selected_band = "B5"
                selection_reason = "Advanced technical content"
            elif diff_score > 0.6:
                selected_band = "B4"
                selection_reason = "Technical content"
            elif diff_score > 0.4:
                selected_band = "B3"
                selection_reason = "Standard code/math"
            else:
                selected_band = "B2"
                selection_reason = "Introductory code/math"

        # 3b. Domain Precedence (Math/Science)
        # Prioritize domain tags over strict difficulty thresholds to avoid gaps
        elif primary_domain == "math_science":
            if diff_score > 0.8:
                selected_band = "B5"
                selection_reason = "Advanced Science (Domain Override)"
            elif diff_score > 0.7:
                selected_band = "B4"
                selection_reason = "Graduate Science (Domain Override)"
            elif diff_score > 0.45:  # Relaxed from 0.60
                selected_band = "B3"
                selection_reason = "Undergraduate Science (Domain Override)"
            else:
                selected_band = "B2"
                selection_reason = "Introductory Science (Domain Override)"

        # 4. Constraint-Based Classification
        if selected_band is None:
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
                if not in_range(
                    structural_density, constraints.structural_density_range
                ):
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
                    selected_band = "B3"
                    selection_reason = "Fallback (COT detected)"
                else:
                    selected_band = "B0"
                    selection_reason = "Fallback (No constraints met)"
            else:
                # Sort candidates B0..B5
                band_order = ["B0", "B1", "B2", "B3", "B4", "B5"]
                sorted_candidates = sorted(
                    candidates, key=lambda b: band_order.index(b)
                )

                if self.logic_config.overlap_policy == "highest":
                    selected = sorted_candidates[-1]
                elif self.logic_config.overlap_policy == "lowest":
                    selected = sorted_candidates[0]
                else:
                    selected = sorted_candidates[-1]  # Default highest

                selected_band = selected
                selection_reason = f"Constraints met: {candidates}"

        # 5. NCERT Adjustment (Post-Processing)
        # Apply strict grade capping even if overridden by Modality/Domain logic
        grade = self._get_ncert_grade(sample)
        if grade is not None:
            # 5a. Early Education Force (Lower Bound)
            # Force B0 for Grade <= 2 (Nursery/Early Primary)
            if grade <= 2:
                selected_band = "B0"
                selection_reason = "Forced B0 (Grade <= 2)"

            # Force max B1 for Grade <= 5 (Primary)
            elif grade <= 5:
                # If current selection is higher than B1, cap it
                # If it's already B0, leave it
                current_idx = ["B0", "B1", "B2", "B3", "B4", "B5"].index(selected_band)
                if current_idx > 1:  # Higher than B1
                    selected_band = "B1"
                    selection_reason = "Capped at B1 (Grade <= 5)"

            # 5b. Advanced Content Promotion (Upper Stratification)
            # Stratify B3 -> B4/B5 based on internal metadata
            elif selected_band == "B3" and primary_domain == "math_science":
                try:
                    meta_str = sample.get("metadata", "{}")
                    # Metadata comes as stringified JSON often in this dataset
                    import json

                    if isinstance(meta_str, str):
                        meta_obj = json.loads(meta_str)
                    else:
                        meta_obj = meta_str if isinstance(meta_str, dict) else {}

                    difficulty = meta_obj.get("difficulty", "")
                    student_level = meta_obj.get("student_level", "")

                    # Promote to B4
                    if difficulty == "Hard" or student_level == "Advanced":
                        # Default promotion to B4
                        selected_band = "B4"
                        selection_reason = "Promoted to B4 (Metadata: Hard/Advanced)"

                        # Aggressive promotion to B5 (Top Tier)
                        # If explicitly marked Advanced/Hard AND is Grade 11 or 12 (Higher Secondary)
                        # AND has high question complexity OR is a complex question type
                        q_complexity = float(meta_obj.get("question_complexity", 0.0))
                        q_type = meta_obj.get("question_type", "")

                        is_complex = q_complexity >= 0.5 or q_type in [
                            "Numerical",
                            "Conceptual",
                        ]

                        if (
                            grade >= 11
                            and (student_level == "Advanced" or difficulty == "Hard")
                            and is_complex
                        ):
                            selected_band = "B5"
                            selection_reason = "Promoted to B5 (Metadata: Advanced + Grade 11+ + Complex)"

                except Exception as e:
                    print(f"DEBUG: Metadata parsing failed for promotion: {e}")

            # 5c. Standard Capping (Upper Bound)
            else:
                selected_band = self.adjust_band_for_ncert(selected_band, grade)

        result = self._result(selected_band, selection_reason)
        # print(f"DEBUG: Band Result: {result}")
        return result

    def _get_ncert_grade(self, sample: Dict[str, Any]) -> int:
        """Extract grade from metadata if available."""
        if "metadata" in sample:
            meta = sample["metadata"]
            if isinstance(meta, str):
                import json

                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    meta = {}

            if isinstance(meta, dict):
                # Check for NCERT explicitly?
                # The user requirement implies applying this logic if "dataset is ncert"
                dataset = sample.get("dataset", "").lower()
                source_type = meta.get("source_type", "")

                is_ncert = (
                    dataset == "ncert"
                    or "ncert" in str(sample.get("id", "")).lower()
                    or source_type == "textbook"
                )

                if is_ncert and "grade" in meta:
                    try:
                        return int(meta["grade"])
                    except (ValueError, TypeError):
                        pass
        return None

    def adjust_band_for_ncert(self, band: str, grade: int) -> str:
        """
        Adjust band assignment based on NCERT grade level.
        Lower grades -> easier bands, higher grades -> harder bands.
        """
        if grade is None:
            return band

        # Grade 6-8: cap at B2
        if grade <= 8:
            if band in ["B3", "B4", "B5"]:
                return "B2"

        # Grade 9-10: cap at B3
        elif grade <= 10:
            if band in ["B4", "B5"]:
                return "B3"

        # Grade 11-12: allow up to B4 (rarely B5)
        elif grade <= 12:
            if band == "B5":
                return "B4"

        return band

    def _result(self, band: str, reason: str) -> Dict[str, Any]:
        """Format the output result."""
        return {"band": band, "reason": reason}
