"""
Curriculum validation and loading module.
Ensures strict compliance with curriculum YAML specifications.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from ..core.types import (
    BandDistribution,
    DifficultyBand,
    DomainDistribution,
    difficulty_band_order,
)


@dataclass
class CurriculumGuard:
    """
    Hard constraints that must be satisfied.
    Violations trigger immediate rejection or halt.
    """

    deterministic_sampling: bool = True
    deterministic_batch_content: bool = True
    deterministic_data_order: bool = True
    seed_required: bool = True
    reproducibility_enforced: bool = True


@dataclass
class GlobalContract:
    """Global contract and safety defaults from curriculum"""

    guarantees: Dict[str, Any]  # determinism settings
    safety_defaults: Dict[str, Any]  # downgrade_on_uncertainty, etc.
    enforcement: Dict[str, Any]  # violation actions
    rejection_reasons: List[str]  # why samples get rejected


@dataclass
class LanguagePolicy:
    """Language constraints"""

    primary_languages: Dict[str, float]  # code -> max_token_share
    secondary_languages: Dict[str, float]
    explicitly_excluded: Set[str]
    violation_action: str = "DROP_SAMPLE"  # DROP_SAMPLE or REJECT_STAGE
    earliest_stage: Optional[str] = None  # For secondary languages


@dataclass
class DifficultySystem:
    """Difficulty band system configuration"""

    definition_method: str  # heuristic, learning_model, etc.
    tokenizer_proxy: Dict[str, Any]  # Team 6 interface
    difficulty_centroids: Dict[str, float]  # B0-B5 centroids
    floors: Dict[str, float]  # Minimum required proportions
    model_capacity_config: Dict[str, Any]  # min/max params


@dataclass
class PerplexityRule:
    """Perplexity filtering rule for a band or globally"""

    ppl_min: float
    ppl_max: float
    band: Optional[str] = None  # None for global


@dataclass
class BandDefinition:
    """Definition of a difficulty band"""

    band: DifficultyBand
    name: str
    intent: str  # NEW: Band intent/purpose
    allowed_modalities: List[
        str
    ]  # NEW: What modalities are allowed (general_text, code, math, etc)
    allowed_domains: List[str]
    reasoning_policy: Dict[str, Any]  # NEW: CoT and agentic policies
    constraints: Dict[str, Any]  # NEW: Tokenizer constraints
    # Legacy fields for backward compatibility
    code_allowed: str = "false"  # false, true, light
    cot_allowed: str = "false"  # false, true, gated, allowed_capped
    agentic_allowed: str = "false"  # false, true, limited
    max_rare_token_percent: Optional[float] = None
    max_tail_token_percent: Optional[float] = None
    min_rare_token_percent: Optional[float] = None
    min_tail_token_percent: Optional[float] = None


@dataclass
class StageSpec:
    """Specification for a training stage"""

    stage_name: str
    total_tokens: int
    band_ratios: BandDistribution
    domain_ratios: Optional[DomainDistribution] = None
    # NEW: Stage profile support (base, harder_shift_1, etc)
    profile: Optional[str] = None
    band_weights: Optional[Dict[str, float]] = None
    modality_weights: Optional[Dict[str, float]] = None
    params: Optional[float] = None  # Model parameters for this stage


@dataclass
class GrowthSchedule:
    """Training growth schedule with stages and profiles"""

    stages: List[Dict[str, Any]]  # List of stage definitions
    stage_profiles: Dict[str, Dict[str, Any]]  # Profile name -> band/modality weights
    adaptive_knobs: Optional[Dict[str, Any]] = None


@dataclass
class RollingWindowSpec:
    """Rolling window constraints for smoothness"""

    window_tokens: int
    max_band_delta: float
    max_domain_delta: float
    enforcement: str = "HARD_REJECT"


@dataclass
class Guardrails:
    """Guardrails and constraints for training"""

    rolling_window: Optional[RollingWindowSpec] = None
    caps: Optional[Dict[str, Any]] = None  # Global caps on CoT, agentic, etc.


@dataclass
class DomainGrouping:
    """Domain definitions and policies"""

    definition_method: str  # How domains are defined
    domain_groups: List[Dict[str, Any]]  # Domain definitions
    band_domain_policy: Dict[str, List[str]]  # Band -> allowed domains


class CurriculumLoader:
    """Load and validate curriculum YAML"""

    def __init__(self, curriculum_path: str):
        self.curriculum_path = Path(curriculum_path)
        self.raw_curriculum: Dict = {}
        self.version: str = ""
        self.status: str = ""
        self.frozen_at: Optional[str] = None
        self.config_hash: str = ""

        # Loaded components
        self.guards: Optional[CurriculumGuard] = None
        self.global_contract: Optional[GlobalContract] = None
        self.language_policy: Optional[LanguagePolicy] = None
        self.difficulty_system: Optional[DifficultySystem] = None
        self.perplexity_rules: Dict[str, PerplexityRule] = {}
        self.bands: Dict[DifficultyBand, BandDefinition] = {}
        self.stages: Dict[str, StageSpec] = {}
        self.growth_schedule: Optional[GrowthSchedule] = None
        self.rolling_window: Optional[RollingWindowSpec] = None
        self.guardrails: Optional[Guardrails] = None
        self.domain_grouping: Optional[DomainGrouping] = None

    def load(self) -> Tuple[bool, List[str]]:
        """Load curriculum from YAML file - supports both old and new schema"""
        errors = []

        if not self.curriculum_path.exists():
            errors.append(f"Curriculum file not found: {self.curriculum_path}")
            return False, errors

        try:
            with open(self.curriculum_path, "r") as f:
                self.raw_curriculum = yaml.safe_load(f)
        except Exception as e:
            errors.append(f"Failed to parse curriculum YAML: {e}")
            return False, errors

        # Extract metadata
        self.version = self.raw_curriculum.get("version", "unknown")
        self.status = self.raw_curriculum.get("status", "DRAFT")
        self.frozen_at = self.raw_curriculum.get("frozen_on", None)

        # Compute hash of curriculum (for reproducibility)
        curriculum_str = yaml.dump(self.raw_curriculum)
        self.config_hash = hashlib.sha256(curriculum_str.encode()).hexdigest()

        # Detect schema version (old vs new)
        is_new_schema = (
            "global_contract" in self.raw_curriculum
            or "language_and_context" in self.raw_curriculum
        )

        if is_new_schema:
            self._load_new_schema(errors)
        else:
            self._load_old_schema(errors)

        return len(errors) == 0, errors

    def _load_old_schema(self, errors: List[str]) -> None:
        """Load old curriculum schema (v0.0.1)"""
        # Parse guarantees
        try:
            guarantees_dict = self.raw_curriculum.get("guarantees", {})
            self.guards = CurriculumGuard(
                deterministic_sampling=guarantees_dict.get(
                    "deterministic_sampling", True
                ),
                deterministic_batch_content=guarantees_dict.get(
                    "deterministic_batch_content", True
                ),
                deterministic_data_order=guarantees_dict.get(
                    "deterministic_data_order", True
                ),
                seed_required=guarantees_dict.get("seed_required", True),
                reproducibility_enforced=guarantees_dict.get(
                    "reproducibility_enforced_by", ""
                )
                != "",
            )
        except Exception as e:
            errors.append(f"Failed to parse guarantees: {e}")

        # Parse language policy
        try:
            lang_dict = self.raw_curriculum.get("languages", {})
            primary = {
                lang["code"]: lang.get("max_token_share", 1.0)
                for lang in lang_dict.get("primary", [])
            }
            secondary = {
                lang["code"]: lang.get("max_token_share", 0.1)
                for lang in lang_dict.get("secondary", [])
            }
            excluded = set(lang_dict.get("explicitly_excluded", []))

            self.language_policy = LanguagePolicy(
                primary_languages=primary,
                secondary_languages=secondary,
                explicitly_excluded=excluded,
                violation_action=lang_dict.get("violation_action", "DROP_SAMPLE"),
            )
        except Exception as e:
            errors.append(f"Failed to parse language policy: {e}")

        # Parse perplexity rules (old schema)
        try:
            ppl_dict = self.raw_curriculum.get("perplexity_filters", {})

            # Global rules
            global_reject = ppl_dict.get("global_reject", {})
            self.perplexity_rules["global"] = PerplexityRule(
                ppl_min=global_reject.get("ppl_min", 1.1),
                ppl_max=global_reject.get("ppl_max", 2000),
                band=None,
            )

            # Band-specific rules
            band_specific = ppl_dict.get("band_specific", {})
            for band_name, rule_dict in band_specific.items():
                ppl_range = rule_dict.get("ppl_range", [1.0, 1000.0])
                self.perplexity_rules[band_name] = PerplexityRule(
                    ppl_min=ppl_range[0],
                    ppl_max=ppl_range[1],
                    band=band_name,
                )
        except Exception as e:
            errors.append(f"Failed to parse perplexity rules: {e}")

        # Parse difficulty bands (old schema)
        try:
            bands_dict = self.raw_curriculum.get("difficulty_bands", {})
            for band_name, band_spec in bands_dict.items():
                band = DifficultyBand(band_name)
                self.bands[band] = BandDefinition(
                    band=band,
                    name=band_spec.get("name", band_name),
                    intent="",  # Not in old schema
                    allowed_modalities=[],  # Not in old schema
                    allowed_domains=band_spec.get("allowed_domains", []),
                    reasoning_policy={},  # Not in old schema
                    constraints={},  # Not in old schema
                    code_allowed=str(band_spec.get("code_allowed", "false")).lower(),
                    cot_allowed=str(band_spec.get("cot_allowed", "false")).lower(),
                    agentic_allowed=str(
                        band_spec.get("agentic_allowed", "false")
                    ).lower(),
                    max_rare_token_percent=band_spec.get(
                        "tokenizer_constraints", {}
                    ).get("max_rare_token_percent"),
                    max_tail_token_percent=band_spec.get(
                        "tokenizer_constraints", {}
                    ).get("max_tail_token_percent"),
                    min_rare_token_percent=band_spec.get(
                        "tokenizer_constraints", {}
                    ).get("min_rare_token_percent"),
                    min_tail_token_percent=band_spec.get(
                        "tokenizer_constraints", {}
                    ).get("min_tail_token_percent"),
                )
        except Exception as e:
            errors.append(f"Failed to parse difficulty bands: {e}")

        # Parse stages (old schema)
        try:
            stages_dict = self.raw_curriculum.get("stages", {})
            for stage_name, stage_spec in stages_dict.items():
                band_ratios_dict = stage_spec.get("band_ratios", {})
                band_dist = BandDistribution(
                    B0=band_ratios_dict.get("B0", 0.0),
                    B1=band_ratios_dict.get("B1", 0.0),
                    B2=band_ratios_dict.get("B2", 0.0),
                    B3=band_ratios_dict.get("B3", 0.0),
                    B4=band_ratios_dict.get("B4", 0.0),
                    B5=band_ratios_dict.get("B5", 0.0),
                    B6=band_ratios_dict.get("B6", 0.0),
                )

                self.stages[stage_name] = StageSpec(
                    stage_name=stage_name,
                    total_tokens=stage_spec.get("total_tokens", 0),
                    band_ratios=band_dist,
                )
        except Exception as e:
            errors.append(f"Failed to parse stages: {e}")

        # Parse rolling window (old schema)
        try:
            rw_dict = self.raw_curriculum.get("rolling_window", {})
            self.rolling_window = RollingWindowSpec(
                window_tokens=rw_dict.get("window_tokens", 1_000_000),
                max_band_delta=rw_dict.get("max_band_delta", 0.03),
                max_domain_delta=rw_dict.get("max_domain_delta", 0.05),
                enforcement=rw_dict.get("enforcement", "HARD_REJECT"),
            )
        except Exception as e:
            errors.append(f"Failed to parse rolling window: {e}")

    def _load_new_schema(self, errors: List[str]) -> None:
        """Load new curriculum schema (v0.4)"""
        # Parse global contract
        try:
            global_contract_dict = self.raw_curriculum.get("global_contract", {})
            self.global_contract = GlobalContract(
                guarantees=global_contract_dict.get("guarantees", {}),
                safety_defaults=global_contract_dict.get("safety_defaults", {}),
                enforcement=global_contract_dict.get("enforcement", {}),
                rejection_reasons=global_contract_dict.get("rejection_reasons", []),
            )

            # Extract guards from global contract
            guarantees = global_contract_dict.get("guarantees", {})
            determinism = guarantees.get("determinism", {})
            self.guards = CurriculumGuard(
                deterministic_sampling=determinism.get("sampling", True),
                deterministic_batch_content=determinism.get("batch_content", True),
                deterministic_data_order=determinism.get("data_order", True),
                seed_required=determinism.get("fixed_seed_required", True),
                reproducibility_enforced=True,
            )
        except Exception as e:
            errors.append(f"Failed to parse global_contract: {e}")

        # Parse language and context
        try:
            lang_context_dict = self.raw_curriculum.get("language_and_context", {})
            lang_policy_dict = lang_context_dict.get("language_policy", {})

            primary = {
                lang["lang"]: lang.get("max_share", 1.0)
                for lang in lang_policy_dict.get("primary_languages", [])
            }
            # v0.6 supports secondary languages specified as either:
            #   - lang: "hi"
            #   - lang: ["as", "bn", ...]
            secondary: Dict[str, float] = {}
            for entry in lang_policy_dict.get("secondary_languages", []) or []:
                langs = entry.get("lang")
                max_share = float(entry.get("max_share", 0.1))
                if not langs:
                    continue
                if isinstance(langs, (list, tuple)):
                    for code in langs:
                        if code:
                            secondary[str(code)] = max_share
                else:
                    secondary[str(langs)] = max_share
            excluded = set(lang_policy_dict.get("excluded_languages", []))

            self.language_policy = LanguagePolicy(
                primary_languages=primary,
                secondary_languages=secondary,
                explicitly_excluded=excluded,
                violation_action=lang_policy_dict.get(
                    "violation_action", "DROP_SAMPLE"
                ),
                earliest_stage=None,
            )

            # Extract earliest_stage if present
            for lang_entry in lang_policy_dict.get("secondary_languages", []):
                if lang_entry.get("earliest_stage"):
                    self.language_policy.earliest_stage = lang_entry["earliest_stage"]
                    break
        except Exception as e:
            errors.append(f"Failed to parse language_and_context: {e}")

        # Parse difficulty system
        try:
            diff_system_dict = self.raw_curriculum.get("difficulty_system", {})
            self.difficulty_system = DifficultySystem(
                definition_method=diff_system_dict.get("definition_method", {}).get(
                    "primary", "heuristic"
                ),
                tokenizer_proxy=diff_system_dict.get("tokenizer_proxy", {}),
                difficulty_centroids=diff_system_dict.get("difficulty_centroids", {}),
                floors=diff_system_dict.get("floors", {}),
                model_capacity_config=diff_system_dict.get("model_capacity_config", {}),
            )
        except Exception as e:
            errors.append(f"Failed to parse difficulty_system: {e}")

        # Parse difficulty bands (new schema)
        try:
            diff_system_dict = self.raw_curriculum.get("difficulty_system", {})
            bands_dict = diff_system_dict.get("bands", {})
            for band_name, band_spec in bands_dict.items():
                band = DifficultyBand(band_name)

                # Extract tokenizer constraints from new schema
                constraints = band_spec.get("constraints", {})
                constraints.get("tokenizer", {})

                self.bands[band] = BandDefinition(
                    band=band,
                    name=band_spec.get("name", band_name),
                    intent=band_spec.get("intent", ""),
                    allowed_modalities=band_spec.get("allowed_modalities", []),
                    allowed_domains=band_spec.get("allowed_domains", []),
                    reasoning_policy=band_spec.get("reasoning_policy", {}),
                    constraints=constraints,
                    # Legacy fields
                    code_allowed=(
                        "true"
                        if "code" in band_spec.get("allowed_modalities", [])
                        else "false"
                    ),
                    cot_allowed=str(
                        band_spec.get("reasoning_policy", {}).get("cot", "false")
                    ).lower(),
                    agentic_allowed=str(
                        band_spec.get("reasoning_policy", {}).get("agentic", "false")
                    ).lower(),
                )
        except Exception as e:
            errors.append(
                f"Failed to parse difficulty bands from difficulty_system: {e}"
            )

        # Parse growth schedule (new schema for stages)
        try:
            growth_schedule_dict = self.raw_curriculum.get("growth_schedule", {})

            # Parse stage_profiles
            stage_profiles = growth_schedule_dict.get("stage_profiles", {})
            self.growth_schedule = GrowthSchedule(
                stages=growth_schedule_dict.get("stages", []),
                stage_profiles=stage_profiles,
                adaptive_knobs=growth_schedule_dict.get("adaptive_knobs"),
            )

            # Create StageSpec entries from growth_schedule
            for stage_info in growth_schedule_dict.get("stages", []):
                stage_name = stage_info.get("name")
                profile_name = stage_info.get("profile")

                if not stage_name:
                    continue

                # Get band/modality weights from profile
                band_weights = {}
                modality_weights = {}
                if profile_name and profile_name in stage_profiles:
                    profile = stage_profiles[profile_name]
                    band_weights = profile.get("band_weights", {})
                    modality_weights = profile.get("modality_weights", {})

                # Convert band_weights to BandDistribution
                band_dist = BandDistribution(
                    B0=band_weights.get("B0", 0.0),
                    B1=band_weights.get("B1", 0.0),
                    B2=band_weights.get("B2", 0.0),
                    B3=band_weights.get("B3", 0.0),
                    B4=band_weights.get("B4", 0.0),
                    B5=band_weights.get("B5", 0.0),
                    B6=band_weights.get("B6", 0.0),
                )

                self.stages[stage_name] = StageSpec(
                    stage_name=stage_name,
                    total_tokens=0,  # Not specified in new schema at this level
                    band_ratios=band_dist,
                    profile=profile_name,
                    band_weights=band_weights,
                    modality_weights=modality_weights,
                    params=stage_info.get("params"),
                )
        except Exception as e:
            errors.append(f"Failed to parse growth_schedule: {e}")

        # Parse guardrails
        try:
            guardrails_dict = self.raw_curriculum.get("guardrails", {})
            rw_dict = guardrails_dict.get("rolling_window", {})

            self.rolling_window = RollingWindowSpec(
                window_tokens=rw_dict.get("window_tokens", 2_000_000),
                max_band_delta=rw_dict.get("max_band_delta", 0.03),
                max_domain_delta=rw_dict.get("max_domain_delta", 0.05),
                enforcement=rw_dict.get("enforcement", "HARD_REJECT"),
            )

            self.guardrails = Guardrails(
                rolling_window=self.rolling_window,
                caps=guardrails_dict.get("caps", {}),
            )
        except Exception as e:
            errors.append(f"Failed to parse guardrails: {e}")

        # Parse domain grouping
        try:
            domains_dict = self.raw_curriculum.get("domains", {})
            self.domain_grouping = DomainGrouping(
                definition_method=domains_dict.get("definition_method", ""),
                domain_groups=domains_dict.get("domain_groups", []),
                band_domain_policy=domains_dict.get("band_domain_policy", {}),
            )
        except Exception as e:
            errors.append(f"Failed to parse domains: {e}")

    def validate_curriculum_frozen(self) -> bool:
        """Check if curriculum is frozen"""
        return self.status == "FROZEN"

    def validate_deterministic_guarantees(self) -> Tuple[bool, List[str]]:
        """Validate deterministic guarantees are met"""
        errors = []

        if not self.guards:
            return False, ["Guards not loaded"]

        if not self.guards.deterministic_sampling:
            errors.append("Deterministic sampling is disabled")
        if not self.guards.seed_required:
            errors.append("Seed requirement is disabled")

        return len(errors) == 0, errors

    def validate_band_ratios(
        self, stage_name: str, actual: BandDistribution
    ) -> Tuple[bool, List[str]]:
        """Validate band distribution matches curriculum for a stage"""
        errors = []

        if stage_name not in self.stages:
            errors.append(f"Stage {stage_name} not found in curriculum")
            return False, errors

        expected = self.stages[stage_name].band_ratios
        tolerance = 0.01  # 1% tolerance

        for band_name in difficulty_band_order():
            expected_val = getattr(expected, band_name)
            actual_val = getattr(actual, band_name)

            if abs(expected_val - actual_val) > tolerance:
                errors.append(
                    f"{band_name}: expected {expected_val:.2%}, got {actual_val:.2%}"
                )

        return len(errors) == 0, errors

    def validate_language_constraints(
        self, lang: str, token_share: float
    ) -> Tuple[bool, str]:
        """Validate language doesn't violate constraints"""
        if not self.language_policy:
            return True, ""

        # Check excluded languages
        if lang in self.language_policy.explicitly_excluded:
            return False, f"Language {lang} is explicitly excluded"

        # Check share for primary languages
        if lang in self.language_policy.primary_languages:
            max_share = self.language_policy.primary_languages[lang]
            if token_share > max_share:
                return (
                    False,
                    f"{lang} share {token_share:.2%} exceeds max {max_share:.2%}",
                )

        # Check share for secondary languages
        if lang in self.language_policy.secondary_languages:
            max_share = self.language_policy.secondary_languages[lang]
            if token_share > max_share:
                return (
                    False,
                    f"{lang} share {token_share:.2%} exceeds max {max_share:.2%}",
                )

        return True, ""

    def validate_perplexity(self, band_name: str, ppl: float) -> Tuple[bool, str]:
        """Validate perplexity within band constraints"""

        # Check band-specific rule
        if band_name in self.perplexity_rules:
            rule = self.perplexity_rules[band_name]
            if not (rule.ppl_min <= ppl <= rule.ppl_max):
                return (
                    False,
                    f"{band_name}: PPL {ppl:.2f} not in [{rule.ppl_min:.2f}, {rule.ppl_max:.2f}]",
                )

        # Check global rule
        if "global" in self.perplexity_rules:
            rule = self.perplexity_rules["global"]
            if not (rule.ppl_min <= ppl <= rule.ppl_max):
                return (
                    False,
                    f"Global: PPL {ppl:.2f} not in [{rule.ppl_min:.2f}, {rule.ppl_max:.2f}]",
                )

        return True, ""

    def get_stage_config(self, stage_name: str) -> Optional[StageSpec]:
        """Get configuration for a specific stage"""
        return self.stages.get(stage_name)

    def get_band_definition(self, band: DifficultyBand) -> Optional[BandDefinition]:
        """Get definition for a difficulty band"""
        return self.bands.get(band)

    def get_allowed_domains_for_band(self, band: DifficultyBand) -> List[str]:
        """Get allowed domains for a band, supporting both old and new schemas"""
        band_def = self.bands.get(band)
        if band_def and band_def.allowed_domains:
            return band_def.allowed_domains

        # Fall back to band_domain_policy from domain_grouping (new schema)
        if self.domain_grouping:
            band_name = band.value if isinstance(band, DifficultyBand) else str(band)
            return self.domain_grouping.band_domain_policy.get(band_name, [])

        return []

    def get_allowed_languages_for_stage(self, stage_name: str) -> Set[str]:
        """Return the set of languages allowed at a given stage.

        Supports both old and new curriculum schemas, including v0.6 secondary language lists.
        """
        if not self.language_policy:
            return set()

        allowed: Set[str] = set(self.language_policy.primary_languages.keys())
        stage_order = ["1B", "3B", "8B", "70B", "SFT", "ALIGNMENT"]
        current_stage_idx = (
            stage_order.index(stage_name)
            if stage_name in stage_order
            else len(stage_order)
        )

        raw = getattr(self, "raw_curriculum", {}) or {}

        # New schema: language_and_context.language_policy.secondary_languages
        secondary_specs = (
            raw.get("language_and_context", {})
            .get("language_policy", {})
            .get("secondary_languages", [])
        )
        if isinstance(secondary_specs, list) and secondary_specs:
            for spec in secondary_specs:
                langs = spec.get("lang")
                earliest = spec.get("earliest_stage")
                if not langs:
                    continue
                if earliest:
                    try:
                        if stage_order.index(str(earliest)) > current_stage_idx:
                            continue
                    except ValueError:
                        # Unknown earliest_stage: be permissive
                        pass
                if isinstance(langs, (list, tuple)):
                    for code in langs:
                        if code:
                            allowed.add(str(code))
                else:
                    allowed.add(str(langs))
            return allowed

        # Old schema: languages.secondary list
        secondary_old = raw.get("languages", {}).get("secondary", [])
        if isinstance(secondary_old, list):
            for spec in secondary_old:
                lang = spec.get("code")
                earliest = spec.get("earliest_stage")
                if not lang:
                    continue
                if earliest:
                    try:
                        if stage_order.index(str(earliest)) > current_stage_idx:
                            continue
                    except ValueError:
                        pass
                allowed.add(str(lang))

        # Fallback: whatever loader parsed into language_policy.secondary_languages
        allowed.update(self.language_policy.secondary_languages.keys())
        return allowed
