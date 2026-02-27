"""Generate a curriculum-aligned JSONL chunk sample.

Purpose
- Produce an input file that is *feasible* for the curriculum constraints at a given
  stage-target scale, so the streaming pipeline can hit scaled token targets.
- Avoids excluded languages and disallowed band/domain combinations.

This script intentionally keeps records lightweight (no token_ids/text) to keep
file size manageable.

Usage:
  python tools/generate_curriculum_aligned_sample.py \
    --curriculum config/curriculum.yaml \
    --out data/datasets/large_sample_chunks.jsonl \
    --stage-target-scale 0.00005 \
    --slack 0.15 \
    --chunk-tokens 512

Prints the computed total token count to use as --total-input-tokens-estimate.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

BANDS: List[str] = ["B0", "B1", "B2", "B3", "B4", "B5"]
STAGES: List[str] = ["1B", "3B", "8B", "70B"]


@dataclass(frozen=True)
class StageProfile:
    total_tokens: int
    band_weights: Dict[str, float]


def _load_stage_profiles(
    curriculum_path: Path,
) -> Tuple[Dict[str, str], Dict[str, StageProfile], Dict[str, List[str]]]:
    raw = yaml.safe_load(curriculum_path.read_text(encoding="utf-8"))

    growth = (raw or {}).get("growth_schedule", {})
    stages = growth.get("stages", []) or []
    stage_profiles_raw = growth.get("stage_profiles", {}) or {}

    stage_to_profile: Dict[str, str] = {}
    for s in stages:
        name = str(s.get("name"))
        profile = str(s.get("profile"))
        if name and profile:
            stage_to_profile[name] = profile

    profiles: Dict[str, StageProfile] = {}
    for profile_name, profile in stage_profiles_raw.items():
        total_tokens = int(profile.get("total_tokens", 0) or 0)
        band_weights = profile.get("band_weights", {}) or {}
        profiles[str(profile_name)] = StageProfile(
            total_tokens=total_tokens,
            band_weights={str(k): float(v) for k, v in band_weights.items()},
        )

    # Band -> allowed domains
    bands = (raw or {}).get("difficulty_system", {}).get("bands", {}) or {}
    allowed_domains_by_band: Dict[str, List[str]] = {}
    for band_name in BANDS:
        allowed = (bands.get(band_name, {}) or {}).get("allowed_domains", []) or []
        allowed_domains_by_band[band_name] = [str(x) for x in allowed]

    return stage_to_profile, profiles, allowed_domains_by_band


def compute_required_band_tokens(
    *,
    stage_to_profile: Dict[str, str],
    profiles: Dict[str, StageProfile],
    stage_target_scale: float,
    slack: float,
) -> Dict[str, int]:
    required: Dict[str, int] = {b: 0 for b in BANDS}

    for stage in STAGES:
        prof_name = stage_to_profile.get(stage)
        if not prof_name:
            continue
        prof = profiles.get(prof_name)
        if not prof:
            continue

        stage_target = max(0, int(prof.total_tokens * float(stage_target_scale)))
        for b in BANDS:
            ratio = float(prof.band_weights.get(b, 0.0) or 0.0)
            need = int(stage_target * ratio)
            need = int(ceil(need * (1.0 + float(slack))))
            required[b] = max(required[b], need)

    return required


def generate(
    *,
    out_path: Path,
    required_band_tokens: Dict[str, int],
    allowed_domains_by_band: Dict[str, List[str]],
    chunk_tokens: int,
    hi_share: float,
    seed: int,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Decide per-band counts
    band_counts: Dict[str, int] = {}
    for b in BANDS:
        req = int(required_band_tokens.get(b, 0) or 0)
        band_counts[b] = int(ceil(req / float(chunk_tokens))) if req > 0 else 0

    sum(band_counts.values())

    # Deterministic language assignment spread across the file.
    # Use a 100-step cycle to approximate hi_share; this keeps each input batch
    # close to the intended language mix, preventing large post-hoc removals.
    hi_per_100 = int(round(max(0.0, min(1.0, float(hi_share))) * 100.0))
    hi_per_100 = max(0, min(100, hi_per_100))

    # Deterministically shuffle band labels so smaller bands are spread across the file.
    band_labels: List[str] = []
    for b in BANDS:
        band_labels.extend([b] * int(band_counts[b]))

    rng = random.Random(int(seed))
    rng.shuffle(band_labels)

    total_tokens = 0

    with out_path.open("w", encoding="utf-8") as f:
        for i, b in enumerate(band_labels):

            allowed = allowed_domains_by_band.get(b) or []
            if not allowed:
                domain = "clean_web"
            else:
                # Choose a single allowed domain per band to keep combos valid and simple.
                # This is sufficient for passing curriculum feasibility checks.
                domain = allowed[0]

            language = "hi" if (hi_per_100 > 0 and (i % 100) < hi_per_100) else "en"
            token_count = int(chunk_tokens)
            total_tokens += token_count

            row = {
                "chunk_id": f"chunk_{i:07d}",
                "dataset_id": f"ds_{domain}",
                "token_count_estimate": token_count,
                "byte_length": token_count * 4,
                "domain": domain,
                "language": language,
                "band": b,
                "source_doc_id": f"doc_{i // 4:07d}",
                "source_url": f"http://example.com/{i}",
                "quality_flags": [],
                "sensitive_markers": [],
                "start_offset": 0,
            }
            f.write(json.dumps(row) + "\n")

    return total_tokens


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a curriculum-aligned JSONL sample"
    )
    ap.add_argument("--curriculum", type=str, default="config/curriculum.yaml")
    ap.add_argument(
        "--out", type=str, default="data/datasets/large_sample_chunks.jsonl"
    )
    ap.add_argument("--stage-target-scale", type=float, default=0.00005)
    ap.add_argument(
        "--slack",
        type=float,
        default=0.15,
        help="Extra supply beyond required tokens per band",
    )
    ap.add_argument(
        "--chunk-tokens", type=int, default=512, help="Token count per chunk"
    )
    ap.add_argument(
        "--hi-share", type=float, default=0.075, help="Fraction of chunks labeled hi"
    )
    ap.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed")

    args = ap.parse_args()

    curriculum_path = Path(args.curriculum)
    out_path = Path(args.out)

    stage_to_profile, profiles, allowed_domains_by_band = _load_stage_profiles(
        curriculum_path
    )
    required = compute_required_band_tokens(
        stage_to_profile=stage_to_profile,
        profiles=profiles,
        stage_target_scale=float(args.stage_target_scale),
        slack=float(args.slack),
    )

    total_tokens = generate(
        out_path=out_path,
        required_band_tokens=required,
        allowed_domains_by_band=allowed_domains_by_band,
        chunk_tokens=int(args.chunk_tokens),
        hi_share=float(args.hi_share),
        seed=int(args.seed),
    )

    print("Generated curriculum-aligned sample")
    print(f"  out: {out_path}")
    print(f"  total_tokens: {total_tokens}")
    print(f"  recommended --total-input-tokens-estimate: {total_tokens}")
    print(f"  band_required_tokens: {required}")


if __name__ == "__main__":
    main()
