
"""
curriculum_yaml_generator.py

Generate a -schema-aligned curriculum.yaml from:
- base_distribution over difficulty bands (B0..B5)
- optional manual stage_profiles (if already decided)
- or compute stage profiles using capacity targets + KL regularization (no proxy training)

Requires: curriculum_tools.py (provided).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import datetime
import yaml

from curriculum_tools import (
    BANDS,
    compute_stage_band_weights,
    apply_floors_caps,
)

DEFAULT_OWNER = "Team 2: Curriculum Architects"

@dataclass
class LanguagePolicy:
    primary: Tuple[str, ...] = ("en",)
    secondary_hi_max_token_share: float = 0.08
    secondary_hi_earliest_stage: str = "3B"
    excluded_langs: Tuple[str, ...] = ("zh","ja","ko")
    language_id_provider: str = "dataset_claimed_clean"
    min_confidence: float = 0.90

@dataclass
class ContextPolicy:
    min_context_tokens: int = 4096
    pretrain_context_tokens: int = 4096
    long_context_target_tokens: int = 262144
    note: str = "Team 2 ensures curriculum is compatible with later context extension; does not implement extension."

def default_dataset_interface() -> dict:
    return {
        "input_from_team1": {
            "required_fields": ["dataset_id", "license", "estimated_tokens", "domain_tags", "language_tags", "cleaning_claims"]
        },
        "segmentation_requests_allowed": True,
        "partial_rejection_allowed": True,
        "rejection_reasons": [
            "language_scope_violation",
            "insufficient_cleaning_claims",
            "duplication_or_boilerplate_risk",
            "benchmark_contamination_risk",
            "unknown_license_or_restrictions",
        ],
    }

def default_guardrails() -> dict:
    return {
        "non_negotiables": [
            "prevent_overfitting_before_capacity_unlocks",
            "long_context_capability_must_appear",
            "reasoning_capability_must_appear",
        ],
        "anti_domain_spike": {
            "rolling_window_tokens": 2_000_000,
            "max_domain_share_in_window": 0.25,
            "smoothing": {"type": "ema", "max_weight_delta_per_update": 0.02},
        },
        "cot_caps": {
            "max_share_global": 0.06,
            "max_share_in_band": {"B3": 0.05, "B4": 0.08, "B5": 0.10},
        },
        "agentic_caps": {"max_share_global": 0.03, "earliest_band": "B4"},
        "hindi_caps": {
            "max_share_global": 0.08,
            "earliest_stage": "3B",
            "note": "Hindi must not leak into B0/B1 as a byproduct of difficulty heuristics.",
        },
        "quality_gates": {
            "enforcement_mode": "dataset_claimed",
            "must_have_claims": [
                "language_filtering_en_hi_only_or_equivalent",
                "boilerplate_and_spam_reduction",
                "deduplication_or_near-dedup",
            ],
            "optional_claims": ["pii_reduction", "toxicity_filtering"],
            "fail_action": "reject_or_request_resegmentation",
        },
    }

def default_stage_profiles() -> dict:
    # -provided sample numbers (can be overridden)
    return {
        "base": {
            "band_weights": {"B0": 0.30, "B1": 0.28, "B2": 0.20, "B3": 0.14, "B4": 0.06, "B5": 0.02},
            "modality_weights": {"general_text": 0.86, "code": 0.12, "cot_reasoning": 0.02, "agentic_traces": 0.00},
            "notes": [
                "Strong language foundation; introduce code early but controlled.",
                "Reasoning seed exists (B4/B5 small) to avoid 'never appears'.",
            ],
        },
        "harder_shift_1": {
            "band_weights": {"B0": 0.22, "B1": 0.24, "B2": 0.22, "B3": 0.18, "B4": 0.10, "B5": 0.04},
            "modality_weights": {"general_text": 0.78, "code": 0.18, "cot_reasoning": 0.03, "agentic_traces": 0.01},
            "notes": [
                "Shift toward higher difficulty and more code.",
                "Hindi can be introduced within caps, staged mainly into B2–B4.",
            ],
        },
        "harder_shift_2": {
            "band_weights": {"B0": 0.14, "B1": 0.18, "B2": 0.22, "B3": 0.22, "B4": 0.16, "B5": 0.08},
            "modality_weights": {"general_text": 0.68, "code": 0.24, "cot_reasoning": 0.05, "agentic_traces": 0.03},
            "notes": ["Ensure reasoning and long-context eval start improving measurably here."],
        },
        "final_adaptive_knobs": {
            "band_weights": {"B0": 0.10, "B1": 0.14, "B2": 0.20, "B3": 0.22, "B4": 0.20, "B5": 0.14},
            "modality_weights": {"general_text": 0.62, "code": 0.28, "cot_reasoning": 0.07, "agentic_traces": 0.03},
            "notes": ["Structure frozen; weights may only shift within bounded knobs (see adaptive_knobs)."],
        },
    }

def growth_schedule() -> dict:
    return {
        "stages": [
            {"name": "1B", "order": 1, "curriculum_profile": "base"},
            {"name": "3B", "order": 2, "curriculum_profile": "harder_shift_1"},
            {"name": "8B", "order": 3, "curriculum_profile": "harder_shift_2"},
            {"name": "70B", "order": 4, "curriculum_profile": "final_adaptive_knobs"},
        ]
    }

def build_curriculum_yaml(
    *,
    frozen_on: str,
    version: str = "0.2",
    owner_team: str = DEFAULT_OWNER,
    language_policy: Optional[LanguagePolicy] = None,
    context_policy: Optional[ContextPolicy] = None,
    stage_profiles: Optional[dict] = None,
    compute_profiles_from_base: bool = False,
    base_distribution: Optional[Dict[str, float]] = None,
) -> dict:
    """
    If compute_profiles_from_base=True, you must pass base_distribution, and we will generate
    band_weights for each profile using capacity targets + KL regularization and minimal floors/caps.
    """
    language_policy = language_policy or LanguagePolicy()
    context_policy = context_policy or ContextPolicy()
    stage_profiles = stage_profiles or default_stage_profiles()

    if compute_profiles_from_base:
        if base_distribution is None:
            raise ValueError("base_distribution is required when compute_profiles_from_base=True")

        # Minimal floors/caps consistent with  guardrails (tune as needed)
        profiles_to_params = {"base": 1e9, "harder_shift_1": 3e9, "harder_shift_2": 8e9, "final_adaptive_knobs": 70e9}
        floors_caps = {
            "base": ({"B0": 0.45, "B1": 0.25}, {"B4": 0.01, "B5": 0.00}),
            "harder_shift_1": ({"B0": 0.15}, {"B5": 0.01}),
            "harder_shift_2": ({"B1": 0.10}, {"B5": 0.03}),
            "final_adaptive_knobs": ({"B0": 0.10}, {}),
        }
        for prof, params in profiles_to_params.items():
            floors, caps = floors_caps.get(prof, ({}, {}))
            bw = compute_stage_band_weights(
                base_distribution,
                params=params,
                floors=floors,
                caps=caps,
                lambda_kl=0.4,
                q_anchors=(0.50, 0.75),
                sharpness=10.0,
            )
            stage_profiles[prof]["band_weights"] = {b: float(round(bw.get(b, 0.0), 6)) for b in BANDS}

    doc = {
        "version": version,
        "owner_team": owner_team,
        "frozen_on": frozen_on,
        "languages": {
            "primary": list(language_policy.primary),
            "secondary": [
                {
                    "lang": "hi",
                    "max_token_share": language_policy.secondary_hi_max_token_share,
                    "earliest_stage": language_policy.secondary_hi_earliest_stage,
                }
            ],
            "excluded_langs": list(language_policy.excluded_langs),
            "language_id_requirement": {
                "provider": language_policy.language_id_provider,
                "min_confidence": language_policy.min_confidence,
            },
        },
        "context_policy": {
            "min_context_tokens": context_policy.min_context_tokens,
            "pretrain_context_tokens": context_policy.pretrain_context_tokens,
            "long_context_target_tokens": context_policy.long_context_target_tokens,
            "note": context_policy.note,
        },
        "growth_schedule": growth_schedule(),
        "dataset_interface": default_dataset_interface(),
        "guardrails": default_guardrails(),
        "stage_profiles": stage_profiles,
        # Optional section: bounded knobs for 70B
        "adaptive_knobs": {
            "enabled_for_profile": "final_adaptive_knobs",
            "max_band_weight_delta": 0.02,
            "max_modality_weight_delta": 0.02,
            "notes": [
                "Knobs are bounded and must preserve guardrails.",
                "Any change requires logging (who/why) and passes the simulator checks.",
            ],
        },
    }
    return doc

def dump_yaml(doc: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)

if __name__ == "__main__":
    # Example usage:
    today = datetime.date.today().isoformat()
    curriculum = build_curriculum_yaml(frozen_on=today)
    dump_yaml(curriculum, "curriculum.yaml")
