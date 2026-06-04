#!/usr/bin/env python3
"""
Preview which shards the curriculum dataloader would use during training.

Shows 10% of each pool's shards (rank 0 slice) so you can see the
manifest-driven distribution without downloading any data.

Usage:
    python scripts/preview_curriculum_shards.py [--stage 1B] [--world-size 8]
"""

import argparse
import json
import os
import sys

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_manifest_shards(manifest_dir: str, filename: str, rank: int, world_size: int):
    path = os.path.join(manifest_dir, filename)
    with open(path, "r") as f:
        all_shards = [line.strip() for line in f if line.strip()]
    rank_shards = all_shards[rank::world_size]
    return all_shards, rank_shards


def main():
    parser = argparse.ArgumentParser(
        description="Preview curriculum shard distribution"
    )
    parser.add_argument("--stage", default="1B", help="Training stage (default: 1B)")
    parser.add_argument(
        "--world-size", type=int, default=8, help="Number of GPUs (default: 8)"
    )
    parser.add_argument(
        "--rank", type=int, default=0, help="Rank to preview (default: 0)"
    )
    parser.add_argument(
        "--manifest-dir", default="manifests", help="Manifest directory"
    )
    parser.add_argument(
        "--curriculum", default="configs/curriculum_v2.yaml", help="Curriculum YAML"
    )
    parser.add_argument(
        "--sample-pct",
        type=float,
        default=0.10,
        help="Fraction of shards to show (default: 0.10)",
    )
    args = parser.parse_args()

    # Load manifest JSON
    manifest_path = os.path.join(args.manifest_dir, "curriculum_v2_manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Load stage weights
    stage_cfg = manifest["stages"].get(args.stage)
    if not stage_cfg:
        print(
            f"Unknown stage '{args.stage}'. Available: {list(manifest['stages'].keys())}"
        )
        sys.exit(1)

    print(f"{'='*70}")
    print(f"  Curriculum Shard Preview — Stage: {args.stage}")
    print(f"  World size: {args.world_size}, Rank: {args.rank}")
    print(f"  Showing {args.sample_pct*100:.0f}% of each pool's shards")
    print(f"{'='*70}\n")

    pools_info = []

    # D1-D4
    for pool_name in ("D1", "D2", "D3", "D4"):
        pool_def = manifest["pools"][pool_name]
        weight = stage_cfg.get(pool_name, 0)
        all_shards, rank_shards = load_manifest_shards(
            args.manifest_dir,
            pool_def["shard_list_file"],
            args.rank,
            args.world_size,
        )
        pools_info.append(
            (pool_name, pool_def["name"], weight, all_shards, rank_shards)
        )

    # AON sub-pools
    aon_def = manifest["pools"]["AON"]
    aon_weight = stage_cfg.get("AON", 0.08)
    for sub_name, sub_def in aon_def["sub_pools"].items():
        display = f"AON_{sub_name}"
        all_shards, rank_shards = load_manifest_shards(
            args.manifest_dir,
            sub_def["shard_list_file"],
            args.rank,
            args.world_size,
        )
        sub_weight = aon_weight * aon_def["internal_split"][sub_name]
        pools_info.append((display, sub_name, sub_weight, all_shards, rank_shards))

    total_all = 0
    total_rank = 0

    for pool_name, description, weight, all_shards, rank_shards in pools_info:
        total_all += len(all_shards)
        total_rank += len(rank_shards)
        sample_n = max(1, int(len(rank_shards) * args.sample_pct))

        print(f"── {pool_name} ({description}) ──")
        print(f"   Weight: {weight:.2%}")
        print(f"   Total shards: {len(all_shards)}")
        print(f"   Rank {args.rank} shards: {len(rank_shards)}")
        print(f"   Sample ({sample_n} shards):")
        for s in rank_shards[:sample_n]:
            print(
                f"     s3://t1-dataacquisition-datasets-2/shards_reordered/{s}/tokens.bin"
            )
        if len(rank_shards) > sample_n:
            print(f"     ... and {len(rank_shards) - sample_n} more")
        print()

    print(f"{'='*70}")
    print(f"  SUMMARY — Stage {args.stage}")
    print(f"{'='*70}")
    print(f"  Total training shards (all ranks): {total_all}")
    print(f"  Shards for rank {args.rank}: {total_rank}")
    print()
    print(f"  Pool weights for stage {args.stage}:")
    for pool_name, _, weight, _, rank_shards in pools_info:
        print(
            f"    {pool_name:12s}  {weight:6.2%}  ({len(rank_shards)} shards on this rank)"
        )
    print()
    print(
        "  Checkpoint state per pool would be just: current_shard_index + completed_count"
    )
    print(
        "  (shard order is deterministic from manifests, so index is enough to resume)"
    )
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
