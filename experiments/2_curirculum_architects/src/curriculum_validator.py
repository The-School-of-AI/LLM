
"""
curriculum_validator.py

Validates a curriculum.yaml against defined guardrails.

Checks:
- required top-level keys
- language scope enforcement
- context policy (4K locked)
- stage profile presence & normalization
- guardrails: cot caps, agentic caps, hindi caps, anti-domain spike config sanity
- policy consistency: earliest_stage / earliest_band constraints

Usage:
  python curriculum_validator.py curriculum.yaml
"""

from __future__ import annotations

import sys
import math
import yaml

BANDS = ["B0","B1","B2","B3","B4","B5"]

def _sum_close_to_one(d: dict, tol: float = 1e-6) -> bool:
    s = sum(float(v) for v in d.values())
    return abs(s - 1.0) <= tol

def _normalize(d: dict) -> dict:
    s = sum(float(v) for v in d.values())
    if s <= 0:
        return d
    return {k: float(v)/s for k,v in d.items()}

def _require_keys(obj: dict, keys: list, where: str):
    missing = [k for k in keys if k not in obj]
    if missing:
        raise ValueError(f"Missing keys at {where}: {missing}")

def validate_language_policy(doc: dict):
    lang = doc["languages"]
    primary = set(lang.get("primary", []))
    secondary = lang.get("secondary", [])
    excluded = set(lang.get("excluded_langs", []))

    if "en" not in primary:
        raise ValueError("languages.primary must include 'en'")

    # exclude zh/ja/ko (can be customized but should exist)
    for ex in ("zh","ja","ko"):
        if ex not in excluded:
            raise ValueError(f"languages.excluded_langs must include '{ex}' ( constraint)")

    # Hindi allowed as secondary with max share and earliest stage
    sec_langs = {x.get("lang") for x in secondary if isinstance(x, dict)}
    if "hi" not in sec_langs:
        raise ValueError("languages.secondary must include Hindi entry {'lang':'hi', ...}")

    hi_entry = [x for x in secondary if isinstance(x, dict) and x.get("lang") == "hi"][0]
    if not (0.0 < float(hi_entry.get("max_token_share", 0.0)) <= 0.20):
        raise ValueError("Hindi max_token_share must be within (0, 0.20] (recommend <=0.08)")
    if hi_entry.get("earliest_stage") not in ("3B","8B","70B"):
        raise ValueError("Hindi earliest_stage must be one of: 3B, 8B, 70B")

def validate_context_policy(doc: dict):
    cp = doc["context_policy"]
    # : 4K context from day one
    if int(cp.get("min_context_tokens", 0)) != 4096 or int(cp.get("pretrain_context_tokens", 0)) != 4096:
        raise ValueError("context_policy must lock min_context_tokens and pretrain_context_tokens to 4096")

def validate_stage_profiles(doc: dict):
    profiles = doc["stage_profiles"]
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("stage_profiles must be a non-empty mapping")

    for name, prof in profiles.items():
        if "band_weights" not in prof or "modality_weights" not in prof:
            raise ValueError(f"stage_profiles.{name} must include band_weights and modality_weights")

        bw = prof["band_weights"]
        mw = prof["modality_weights"]

        # Ensure all bands exist
        for b in BANDS:
            if b not in bw:
                raise ValueError(f"stage_profiles.{name}.band_weights missing {b}")

        # normalization
        if not _sum_close_to_one(bw, tol=1e-4):
            raise ValueError(f"stage_profiles.{name}.band_weights must sum to 1.0 (now {sum(bw.values())})")
        if not _sum_close_to_one(mw, tol=1e-4):
            raise ValueError(f"stage_profiles.{name}.modality_weights must sum to 1.0 (now {sum(mw.values())})")

        # Non-negative
        if any(float(v) < -1e-12 for v in bw.values()):
            raise ValueError(f"stage_profiles.{name}.band_weights has negative values")
        if any(float(v) < -1e-12 for v in mw.values()):
            raise ValueError(f"stage_profiles.{name}.modality_weights has negative values")

def validate_growth_schedule(doc: dict):
    sched = doc["growth_schedule"]["stages"]
    if not sched or not isinstance(sched, list):
        raise ValueError("growth_schedule.stages must be a non-empty list")
    profiles = set(doc["stage_profiles"].keys())
    for s in sched:
        _require_keys(s, ["name","order","curriculum_profile"], "growth_schedule.stages[]")
        if s["curriculum_profile"] not in profiles:
            raise ValueError(f"growth_schedule references missing profile: {s['curriculum_profile']}")

def validate_guardrails(doc: dict):
    g = doc["guardrails"]

    # Anti-domain spike sanity
    ads = g["anti_domain_spike"]
    if int(ads.get("rolling_window_tokens", 0)) <= 0:
        raise ValueError("guardrails.anti_domain_spike.rolling_window_tokens must be positive")
    max_share = float(ads.get("max_domain_share_in_window", 0.0))
    if not (0.0 < max_share <= 0.50):
        raise ValueError("guardrails.anti_domain_spike.max_domain_share_in_window must be within (0, 0.5]")

    # CoT caps
    cot = g["cot_caps"]
    if not (0.0 <= float(cot.get("max_share_global", 0.0)) <= 0.20):
        raise ValueError("guardrails.cot_caps.max_share_global must be within [0, 0.2]")
    per_band = cot.get("max_share_in_band", {})
    for b, cap in per_band.items():
        if b not in BANDS:
            raise ValueError(f"guardrails.cot_caps.max_share_in_band has unknown band {b}")
        if not (0.0 <= float(cap) <= 0.25):
            raise ValueError(f"CoT cap for {b} must be within [0, 0.25]")

    # Agentic
    ag = g["agentic_caps"]
    if not (0.0 <= float(ag.get("max_share_global", 0.0)) <= 0.20):
        raise ValueError("guardrails.agentic_caps.max_share_global must be within [0, 0.2]")
    earliest_band = ag.get("earliest_band")
    if earliest_band not in ("B3","B4","B5"):
        raise ValueError("guardrails.agentic_caps.earliest_band must be one of B3/B4/B5")
    if earliest_band != "B4":
        #  expectation: B4 earliest; warn instead of hard error
        print("WARNING:  expectation is earliest_band='B4' for agentic traces")

    # Hindi caps
    hi = g["hindi_caps"]
    if not (0.0 <= float(hi.get("max_share_global", 0.0)) <= 0.20):
        raise ValueError("guardrails.hindi_caps.max_share_global must be within [0, 0.2]")
    if hi.get("earliest_stage") not in ("3B","8B","70B"):
        raise ValueError("guardrails.hindi_caps.earliest_stage must be 3B/8B/70B")

def validate(doc: dict):
    _require_keys(doc, ["version","owner_team","frozen_on","languages","context_policy","growth_schedule","dataset_interface","guardrails","stage_profiles"], "root")
    validate_language_policy(doc)
    validate_context_policy(doc)
    validate_growth_schedule(doc)
    validate_stage_profiles(doc)
    validate_guardrails(doc)

def main():
    if len(sys.argv) < 2:
        print("Usage: python curriculum_validator.py curriculum.yaml")
        sys.exit(2)
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    validate(doc)
    print("OK: curriculum.yaml passed validation.")

if __name__ == "__main__":
    main()
