"""Calculate curriculum band proportions based on model capacity."""

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Optional

import pyarrow.parquet as pq
import yaml

# Ensure project root is in path
sys.path.append(str(Path(__file__).parent.parent))

from curriculum_tags.metrics.band_assignment import BandAssignmentMetric


def load_curriculum_config(path: str | Path) -> Dict:
    """Load config from curriculum.yaml."""
    with open(path) as f:
        return yaml.safe_load(f)


def model_capacity(
    params: float, min_params: float = 1e9, max_params: float = 70e9
) -> float:
    """Normalized model capacity based on log(parameter count)."""
    # Clamp params to range to avoid domain errors or out-of-bounds
    params = max(min_params, min(params, max_params))

    return (math.log(params) - math.log(min_params)) / (
        math.log(max_params) - math.log(min_params)
    )


def alignment_weight(
    difficulty: float, capacity: float, lambda_align: float = 3.0
) -> float:
    """Smooth alignment between band difficulty and model capacity."""
    return math.exp(-lambda_align * abs(difficulty - capacity))


def raw_band_weights(
    base_distribution: Dict[str, float],
    params: float,
    band_diffs: Dict[str, float],
    lambda_align: float = 3.0,
) -> Dict[str, float]:
    """Compute raw (unnormalized) band weights."""
    capacity = model_capacity(params)
    print(f"Model Capacity Score: {capacity:.4f} (Params: {params/1e9:.1f}B)")

    weights = {}
    for band, base_w in base_distribution.items():
        if band not in band_diffs:
            print(f"Warning: Band {band} missing in difficulty centroids, skipping.")
            continue

        d_b = band_diffs[band]
        align = alignment_weight(d_b, capacity, lambda_align)
        weights[band] = base_w * align
        # print(f"  {band}: Base={base_w:.3f} Diff={d_b:.2f} Align={align:.3f} -> Raw={weights[band]:.4f}")

    return weights


def apply_floors_and_caps(
    weights: Dict[str, float],
    floors: Dict[str, float],
    caps: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Enforce curriculum floors and caps."""
    caps = caps or {}
    constrained = {}

    for band, w in weights.items():
        w_floor = max(w, floors.get(band, 0.0))
        w_cap = min(w_floor, caps.get(band, float("inf")))
        constrained[band] = w_cap

    return constrained


def renormalize(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize weights to sum to 1.0."""
    total = sum(weights.values())
    if total == 0:
        raise ValueError("Cannot renormalize: total weight is zero.")
    return {b: w / total for b, w in weights.items()}


def compute_band_proportions(
    base_distribution: Dict[str, float],
    params: float,
    band_diffs: Dict[str, float],
    floors: Dict[str, float],
    caps: Dict[str, float],
    lambda_align: float = 3.0,
) -> Dict[str, float]:
    """Full curriculum band proportion computation for one training stage."""
    # Ensure all bands present in base
    for band in band_diffs.keys():
        if band not in base_distribution:
            base_distribution[band] = 0.0

    raw = raw_band_weights(
        base_distribution=base_distribution,
        params=params,
        band_diffs=band_diffs,
        lambda_align=lambda_align,
    )

    constrained = apply_floors_and_caps(raw, floors=floors, caps=caps)

    final = renormalize(constrained)
    return final


def sample_metadata(
    metadata_path: str,
    sample_rate: float,
    recompute: bool = False,
    curriculum_path: str = "curriculum.yaml",
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Sample metadata and calculate base distribution of bands."""
    print(f"Reading metadata from: {metadata_path}")
    print(f"Sampling rate: {sample_rate*100}%")
    if recompute:
        print(
            "Re-computing band assignments using current band_assignment.yaml logic..."
        )
        # Create metric instance
        fake_config = type(
            "Config", (), {"path": str(Path(curriculum_path).absolute())}
        )()
        metric = BandAssignmentMetric(fake_config)

    if not Path(metadata_path).exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # pq.read_table handles single files or directories
    table = pq.read_table(metadata_path)
    total_rows = table.num_rows
    print(f"Total rows found across files: {total_rows}")

    # Set random seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)

    # Simple random sampling indices
    sample_size = int(total_rows * sample_rate)
    indices = sorted(random.sample(range(total_rows), sample_size))

    # Take rows
    samples = table.take(indices).to_pylist()

    # Count bands
    counts = {}
    valid_samples = 0

    for row in samples:
        tags = row.get("curriculum_tags", {})
        band = None

        if recompute:
            # IMPORTANT: Re-run logic on the raw tags
            # Construct sample dict expected by compute()
            # compute() expects {"curriculum_tags": ...}
            # row is flat, but contains "curriculum_tags" column which is a dict (struct in parquet)
            # wait, table.to_pylist() converts struct columns to dicts.
            # So tags is indeed the dictionary of tags.
            sample_input = {"curriculum_tags": tags}
            result = metric.compute(sample_input)
            band = result["band"]
        else:
            # Priority 1: New Band Assignment Metric
            if "band_assignment" in tags and "band" in tags["band_assignment"]:
                band = tags["band_assignment"]["band"]

            # Priority 2: Legacy Difficulty Metric (Fallback)
            elif "difficulty" in tags and "band" in tags["difficulty"]:
                band = tags["difficulty"]["band"]

        if band:
            counts[band] = counts.get(band, 0) + 1
            valid_samples += 1

    if valid_samples == 0:
        raise ValueError(
            "No valid band assignment or difficulty tags found in sampled data."
        )

    # Convert to proportions
    return {k: v / valid_samples for k, v in counts.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Calculate curriculum band proportions."
    )
    parser.add_argument(
        "metadata_path",
        help="Path to .metadata.parquet file or directory containing them",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=0.005,
        help="Sampling rate (default 0.005)",
    )
    parser.add_argument(
        "--curriculum-path", default="curriculum.yaml", help="Path to curriculum.yaml"
    )
    parser.add_argument("--output-json", help="Path to save output proportions as JSON")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Re-run band assignment logic on signals instead of using stored tags",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (default: None)",
    )

    args = parser.parse_args()

    # Load config
    config = load_curriculum_config(args.curriculum_path)
    diff_system = config.get("difficulty_system", {})
    growth_schedule = config.get("growth_schedule", {})

    band_diffs = diff_system.get("difficulty_centroids", {})
    floors = diff_system.get("floors", {})
    cap_config = diff_system.get("model_capacity_config", {})
    stages = growth_schedule.get("stages", [])

    # Validation
    if not band_diffs:
        print("Error: 'difficulty_centroids' missing in curriculum.yaml")
        return
    if not floors:
        print("Error: 'floors' missing in curriculum.yaml")
        return
    if not stages:
        print("Error: 'growth_schedule.stages' missing in curriculum.yaml")
        return

    min_params = cap_config.get("min_params", 1e9)
    max_params = cap_config.get("max_params", 70e9)

    # 1. Sample Distribution
    try:
        base_dist = sample_metadata(
            args.metadata_path,
            args.sampling_rate,
            args.recompute,
            args.curriculum_path,
            args.seed,
        )
        print("\nBase Distribution (from Data):")
        for b, p in sorted(base_dist.items()):
            print(f"  {b}: {p:.4f}")
    except Exception as e:
        print(f"Error sampling data: {e}")
        return

    # 2. Compute Proportions for Each Stage
    print(f"\nComputing Target Proportions for {len(stages)} stages...")

    all_stage_results = {}

    for stage in stages:
        stage_name = stage["name"]
        params = stage.get("params")

        if params is None:
            print(f"Warning: Stage '{stage_name}' missing 'params', skipping.")
            continue

        params = float(params)
        capacity = model_capacity(params, min_params=min_params, max_params=max_params)

        print(f"\nStage: {stage_name} (Params: {params/1e9:.1f}B, Cap: {capacity:.4f})")
        print("-" * 40)

        weights = {}
        for band, base_w in base_dist.items():
            if band not in band_diffs:
                continue
            d_b = band_diffs[band]
            align = alignment_weight(d_b, capacity, lambda_align=3.0)
            weights[band] = base_w * align

        constrained = apply_floors_and_caps(weights, floors=floors, caps={})
        final_props = renormalize(constrained)

        all_stage_results[stage_name] = final_props

        for band in sorted(band_diffs.keys()):
            prop = final_props.get(band, 0.0)
            print(f"  {band}: {prop:.4f}")
        print(f"  Sum: {sum(final_props.values()):.4f}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_stage_results, f, indent=2)
        print(f"\nSaved all stage proportions to {args.output_json}")


if __name__ == "__main__":
    main()
