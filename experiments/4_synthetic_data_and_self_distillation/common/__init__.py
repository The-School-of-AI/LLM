"""
common — Shared definitions for synthetic data pipeline.

Exports:
  • Band definitions and COT policy
  • Skill buckets and failure modes
  • Special tokens
"""

from .bands import (
    Band,
    BandSpec,
    Stage,
    StageConfig,
    BAND_SPECS,
    STAGE_CONFIGS,
    THINK_START,
    THINK_END,
    DIST_START,
    DIST_END,
    COT_GATE_TOKEN,
    get_band_spec,
    cot_allowed_for_band,
    get_cot_max_tokens,
    get_injection_cap,
    ALLOWED_LANGUAGES,
)

from .skills import (
    SkillCategory,
    SkillBucket,
    FailureMode,
    SKILL_BUCKETS,
    FAILURE_MODES,
    get_skill_bucket,
    get_skills_for_band,
    get_skills_for_stage,
    get_failure_mode,
    get_skill_failure_modes,
)

__all__ = [
    # Bands
    "Band",
    "BandSpec", 
    "Stage",
    "StageConfig",
    "BAND_SPECS",
    "STAGE_CONFIGS",
    "get_band_spec",
    "cot_allowed_for_band",
    "get_cot_max_tokens",
    "get_injection_cap",
    # Tokens
    "THINK_START",
    "THINK_END",
    "DIST_START",
    "DIST_END",
    "COT_GATE_TOKEN",
    # Skills
    "SkillCategory",
    "SkillBucket",
    "FailureMode",
    "SKILL_BUCKETS",
    "FAILURE_MODES",
    "get_skill_bucket",
    "get_skills_for_band",
    "get_skills_for_stage",
    "get_failure_mode",
    "get_skill_failure_modes",
    # Languages
    "ALLOWED_LANGUAGES",
]
