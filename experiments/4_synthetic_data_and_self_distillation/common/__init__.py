"""
common — Shared definitions for synthetic data pipeline.

Exports:
  • Band definitions and COT policy
  • Skill buckets and failure modes
  • Special tokens
  • Ollama client with seed support
  • Run tracker for reproducibility
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
from .ollama_client import check_ollama as check_ollama
from .ollama_client import get_ollama_seed as get_ollama_seed
from .ollama_client import get_ollama_version as get_ollama_version
from .ollama_client import ollama_chat as ollama_chat
from .ollama_client import ollama_generate as ollama_generate
from .ollama_client import set_ollama_seed as set_ollama_seed
from .run_tracker import RunTracker as RunTracker
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
    # Ollama client
    "ollama_chat",
    "ollama_generate",
    "check_ollama",
    "set_ollama_seed",
    "get_ollama_seed",
    "get_ollama_version",
    # Run tracker
    "RunTracker",
]
