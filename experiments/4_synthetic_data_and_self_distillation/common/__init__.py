"""
common — Shared definitions for synthetic data pipeline.

Exports:
  • Band definitions and COT policy
  • Skill buckets and failure modes
  • Special tokens
"""

from .bands import (
    ALLOWED_LANGUAGES,
    BAND_SPECS,
    COT_GATE_TOKEN,
    DIST_END,
    DIST_START,
    STAGE_CONFIGS,
    THINK_END,
    THINK_START,
    Band,
    BandSpec,
    Stage,
    StageConfig,
    cot_allowed_for_band,
    get_band_spec,
    get_cot_max_tokens,
    get_injection_cap,
)
from .skills import (
    FAILURE_MODES,
    SKILL_BUCKETS,
    FailureMode,
    SkillBucket,
    SkillCategory,
    get_failure_mode,
    get_skill_bucket,
    get_skill_failure_modes,
    get_skills_for_band,
    get_skills_for_stage,
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
