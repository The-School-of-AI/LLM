"""
Curriculum validation and loading module.
Ensures strict compliance with curriculum YAML specifications.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import yaml
from pathlib import Path
from datetime import datetime
import hashlib

from ..core.types import DifficultyBand, BandDistribution, DomainDistribution


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
class LanguagePolicy:
    """Language constraints"""
    primary_languages: Dict[str, float]  # code -> max_token_share
    secondary_languages: Dict[str, float]
    explicitly_excluded: Set[str]
    violation_action: str = "DROP_SAMPLE"  # DROP_SAMPLE or REJECT_STAGE


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
    allowed_domains: List[str]
    code_allowed: str  # false, true, light
    cot_allowed: str  # false, true, gated, allowed_capped
    agentic_allowed: str  # false, true, limited
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


@dataclass
class RollingWindowSpec:
    """Rolling window constraints for smoothness"""
    window_tokens: int
    max_band_delta: float
    max_domain_delta: float
    enforcement: str = "HARD_REJECT"


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
        self.language_policy: Optional[LanguagePolicy] = None
        self.perplexity_rules: Dict[str, PerplexityRule] = {}
        self.bands: Dict[DifficultyBand, BandDefinition] = {}
        self.stages: Dict[str, StageSpec] = {}
        self.rolling_window: Optional[RollingWindowSpec] = None
        
    def load(self) -> Tuple[bool, List[str]]:
        """Load curriculum from YAML file"""
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
        
        # Parse guarantees
        try:
            guarantees_dict = self.raw_curriculum.get("guarantees", {})
            self.guards = CurriculumGuard(
                deterministic_sampling=guarantees_dict.get("deterministic_sampling", True),
                deterministic_batch_content=guarantees_dict.get("deterministic_batch_content", True),
                deterministic_data_order=guarantees_dict.get("deterministic_data_order", True),
                seed_required=guarantees_dict.get("seed_required", True),
                reproducibility_enforced=guarantees_dict.get("reproducibility_enforced_by", "") != "",
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
        
        # Parse perplexity rules
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
        
        # Parse difficulty bands
        try:
            bands_dict = self.raw_curriculum.get("difficulty_bands", {})
            for band_name, band_spec in bands_dict.items():
                band = DifficultyBand(band_name)
                self.bands[band] = BandDefinition(
                    band=band,
                    name=band_spec.get("name", band_name),
                    allowed_domains=band_spec.get("allowed_domains", []),
                    code_allowed=str(band_spec.get("code_allowed", "false")).lower(),
                    cot_allowed=str(band_spec.get("cot_allowed", "false")).lower(),
                    agentic_allowed=str(band_spec.get("agentic_allowed", "false")).lower(),
                    max_rare_token_percent=band_spec.get("tokenizer_constraints", {}).get("max_rare_token_percent"),
                    max_tail_token_percent=band_spec.get("tokenizer_constraints", {}).get("max_tail_token_percent"),
                    min_rare_token_percent=band_spec.get("tokenizer_constraints", {}).get("min_rare_token_percent"),
                    min_tail_token_percent=band_spec.get("tokenizer_constraints", {}).get("min_tail_token_percent"),
                )
        except Exception as e:
            errors.append(f"Failed to parse difficulty bands: {e}")
        
        # Parse stages
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
                )
                
                self.stages[stage_name] = StageSpec(
                    stage_name=stage_name,
                    total_tokens=stage_spec.get("total_tokens", 0),
                    band_ratios=band_dist,
                )
        except Exception as e:
            errors.append(f"Failed to parse stages: {e}")
        
        # Parse rolling window
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
        
        return len(errors) == 0, errors
    
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
    
    def validate_band_ratios(self, stage_name: str, actual: BandDistribution) -> Tuple[bool, List[str]]:
        """Validate band distribution matches curriculum for a stage"""
        errors = []
        
        if stage_name not in self.stages:
            errors.append(f"Stage {stage_name} not found in curriculum")
            return False, errors
        
        expected = self.stages[stage_name].band_ratios
        tolerance = 0.01  # 1% tolerance
        
        for band_name in ["B0", "B1", "B2", "B3", "B4", "B5"]:
            expected_val = getattr(expected, band_name)
            actual_val = getattr(actual, band_name)
            
            if abs(expected_val - actual_val) > tolerance:
                errors.append(
                    f"{band_name}: expected {expected_val:.2%}, got {actual_val:.2%}"
                )
        
        return len(errors) == 0, errors
    
    def validate_language_constraints(self, lang: str, token_share: float) -> Tuple[bool, str]:
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
                return False, f"{lang} share {token_share:.2%} exceeds max {max_share:.2%}"
        
        # Check share for secondary languages
        if lang in self.language_policy.secondary_languages:
            max_share = self.language_policy.secondary_languages[lang]
            if token_share > max_share:
                return False, f"{lang} share {token_share:.2%} exceeds max {max_share:.2%}"
        
        return True, ""
    
    def validate_perplexity(self, band_name: str, ppl: float) -> Tuple[bool, str]:
        """Validate perplexity within band constraints"""
        
        # Check band-specific rule
        if band_name in self.perplexity_rules:
            rule = self.perplexity_rules[band_name]
            if not (rule.ppl_min <= ppl <= rule.ppl_max):
                return False, f"{band_name}: PPL {ppl:.2f} not in [{rule.ppl_min:.2f}, {rule.ppl_max:.2f}]"
        
        # Check global rule
        if "global" in self.perplexity_rules:
            rule = self.perplexity_rules["global"]
            if not (rule.ppl_min <= ppl <= rule.ppl_max):
                return False, f"Global: PPL {ppl:.2f} not in [{rule.ppl_min:.2f}, {rule.ppl_max:.2f}]"
        
        return True, ""
    
    def get_stage_config(self, stage_name: str) -> Optional[StageSpec]:
        """Get configuration for a specific stage"""
        return self.stages.get(stage_name)
    
    def get_band_definition(self, band: DifficultyBand) -> Optional[BandDefinition]:
        """Get definition for a difficulty band"""
        return self.bands.get(band)
