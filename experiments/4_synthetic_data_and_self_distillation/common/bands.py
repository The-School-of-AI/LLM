"""
bands.py — Band Definitions & COT Policy (Authoritative)

This is the SINGLE SOURCE OF TRUTH for:
  • Band definitions (B0-B5)
  • COT policy per band
  • Language constraints
  • Stage alignment

All other modules import from here.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


# ================================================================
# BAND DEFINITIONS
# ================================================================

class Band(Enum):
    """Training difficulty bands."""
    B0 = "B0"  # Nursery
    B1 = "B1"  # Primary
    B2 = "B2"  # High School
    B3 = "B3"  # Undergraduate
    B4 = "B4"  # Graduate
    B5 = "B5"  # PhD


@dataclass
class BandSpec:
    """Specification for a training band."""
    band: Band
    name: str
    description: str
    
    # COT Policy
    cot_allowed: bool
    cot_frequency: Literal["never", "rare", "allowed"]
    cot_max_tokens: int | None  # None = no COT allowed
    
    # Content constraints
    reasoning_explicit: bool
    code_complexity: Literal["none", "trivial", "introductory", "non-trivial", "advanced", "system-level"]
    
    # Language
    hindi_allowed: bool
    hindi_cap_pct: float  # max % of samples in Hindi


# Band specifications (LOCKED - do not modify without escalation)
BAND_SPECS: dict[Band, BandSpec] = {
    Band.B0: BandSpec(
        band=Band.B0,
        name="Nursery",
        description="Language fundamentals: grammar, syntax, high-frequency constructions",
        cot_allowed=False,
        cot_frequency="never",
        cot_max_tokens=None,
        reasoning_explicit=False,
        code_complexity="none",
        hindi_allowed=True,
        hindi_cap_pct=5.0,
    ),
    Band.B1: BandSpec(
        band=Band.B1,
        name="Primary",
        description="Fluent everyday language, common knowledge, clean narrative",
        cot_allowed=False,
        cot_frequency="never",
        cot_max_tokens=None,
        reasoning_explicit=False,
        code_complexity="trivial",
        hindi_allowed=True,
        hindi_cap_pct=10.0,
    ),
    Band.B2: BandSpec(
        band=Band.B2,
        name="High School",
        description="Structured knowledge, richer topics, implicit reasoning allowed",
        cot_allowed=False,
        cot_frequency="never",
        cot_max_tokens=None,
        reasoning_explicit=False,  # implicit only
        code_complexity="introductory",
        hindi_allowed=True,
        hindi_cap_pct=10.0,
    ),
    Band.B3: BandSpec(
        band=Band.B3,
        name="Undergraduate",
        description="Reasoning begins, multi-step explanations, non-trivial code",
        cot_allowed=True,
        cot_frequency="rare",
        cot_max_tokens=1024,  # increased from 512 to prevent truncation
        reasoning_explicit=True,
        code_complexity="non-trivial",
        hindi_allowed=True,
        hindi_cap_pct=5.0,
    ),
    Band.B4: BandSpec(
        band=Band.B4,
        name="Graduate",
        description="Explicit reasoning, math/algorithms/proofs, controlled COT",
        cot_allowed=True,
        cot_frequency="allowed",
        cot_max_tokens=1024,  # capped
        reasoning_explicit=True,
        code_complexity="advanced",
        hindi_allowed=False,
        hindi_cap_pct=0.0,
    ),
    Band.B5: BandSpec(
        band=Band.B5,
        name="PhD",
        description="Maximum trusted complexity, hardest reasoning, limited agentic",
        cot_allowed=True,
        cot_frequency="allowed",
        cot_max_tokens=2048,  # still capped, never dominant
        reasoning_explicit=True,
        code_complexity="system-level",
        hindi_allowed=False,
        hindi_cap_pct=0.0,
    ),
}


def get_band_spec(band: Band | str) -> BandSpec:
    """Get specification for a band."""
    if isinstance(band, str):
        band = Band(band)
    return BAND_SPECS[band]


def cot_allowed_for_band(band: Band | str) -> bool:
    """Check if COT is allowed for a band."""
    return get_band_spec(band).cot_allowed


def get_cot_max_tokens(band: Band | str) -> int:
    """Get max COT tokens for a band (0 if not allowed)."""
    spec = get_band_spec(band)
    return spec.cot_max_tokens or 0


# ================================================================
# TRAINING STAGES
# ================================================================

class Stage(Enum):
    """Training stages where synthetic data can be injected."""
    PRE = "PRE"    # Pretraining
    SFT = "SFT"    # Supervised Fine-Tuning
    DPO = "DPO"    # Direct Preference Optimization
    RLHF = "RLHF"  # Reinforcement Learning from Human Feedback


@dataclass
class StageConfig:
    """Configuration for a training stage."""
    stage: Stage
    synthetic_cap_pct: float  # max % of synthetic data
    bands_allowed: list[Band]
    cot_sampling_weight: float  # weight for COT view vs distilled view


STAGE_CONFIGS: dict[Stage, StageConfig] = {
    Stage.PRE: StageConfig(
        stage=Stage.PRE,
        synthetic_cap_pct=5.0,
        bands_allowed=[Band.B0, Band.B1, Band.B2, Band.B3, Band.B4, Band.B5],
        cot_sampling_weight=0.2,  # 20% COT, 80% distilled
    ),
    Stage.SFT: StageConfig(
        stage=Stage.SFT,
        synthetic_cap_pct=10.0,
        bands_allowed=[Band.B3, Band.B4, Band.B5],
        cot_sampling_weight=0.4,  # 40% COT in SFT
    ),
    Stage.DPO: StageConfig(
        stage=Stage.DPO,
        synthetic_cap_pct=10.0,
        bands_allowed=[Band.B3, Band.B4, Band.B5],
        cot_sampling_weight=0.3,
    ),
    Stage.RLHF: StageConfig(
        stage=Stage.RLHF,
        synthetic_cap_pct=5.0,
        bands_allowed=[Band.B4, Band.B5],
        cot_sampling_weight=0.3,
    ),
}


# ================================================================
# SPECIAL TOKENS
# ================================================================

# COT gating tokens (used in dual-view format)
THINK_START = "<think>"
THINK_END = "</think>"
DIST_START = "<dist>"
DIST_END = "</dist>"

# For training, these determine which view to use
COT_GATE_TOKEN = "<|think|>"  # Presence triggers COT mode


# ================================================================
# LANGUAGE CONSTRAINTS
# ================================================================

ALLOWED_LANGUAGES = ["en", "hi"]  # English primary, Hindi capped
PRIMARY_LANGUAGE = "en"
SECONDARY_LANGUAGE = "hi"

# Hindi constraints per band (enforced during generation)
HINDI_CONSTRAINTS = {
    Band.B0: {"allowed": True, "max_pct": 5.0},
    Band.B1: {"allowed": True, "max_pct": 10.0},
    Band.B2: {"allowed": True, "max_pct": 10.0},
    Band.B3: {"allowed": True, "max_pct": 5.0},
    Band.B4: {"allowed": False, "max_pct": 0.0},
    Band.B5: {"allowed": False, "max_pct": 0.0},
}


# ================================================================
# INJECTION CAPS
# ================================================================

# Synthetic data injection caps per stage (5-10% rule)
INJECTION_CAPS = {
    Stage.PRE: {
        Band.B0: 0.02,  # 2%
        Band.B1: 0.03,  # 3%
        Band.B2: 0.05,  # 5%
        Band.B3: 0.07,  # 7%
        Band.B4: 0.10,  # 10%
        Band.B5: 0.10,  # 10%
    },
    Stage.SFT: {
        Band.B3: 0.10,
        Band.B4: 0.10,
        Band.B5: 0.10,
    },
    Stage.DPO: {
        Band.B3: 0.10,
        Band.B4: 0.10,
        Band.B5: 0.10,
    },
}


def get_injection_cap(stage: Stage, band: Band) -> float:
    """Get max injection % for a stage+band combo."""
    stage_caps = INJECTION_CAPS.get(stage, {})
    return stage_caps.get(band, 0.0)
