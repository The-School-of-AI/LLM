#!/usr/bin/env python3
"""
Create synthetic test shards for curriculum_v2 checkpoint testing.

Generates 2 shards per pool (D1-D4 + AON_bench + AON_indic = 12 shards total)
using the first 2 entries from each manifest file. Each shard is a directory
containing a tokens.bin file with random uint32 token IDs.

This avoids downloading real shards from S3 while exercising the full
curriculum dataloader → spot checkpoint → resume pipeline.

Usage:
    python scripts/create_curriculum_test_shards.py \
        --output-dir /tmp/curriculum_test_shards \
        --manifest-dir manifests \
        --shards-per-pool 2
"""

import argparse
import os
import struct

# 4096 tokens per block, uint32 = 4 bytes each
BLOCK_SIZE = 4096
BYTES_PER_TOKEN = 4
# Generate 3 blocks per shard (enough for a few sequences)
BLOCKS_PER_SHARD = 3


POOL_MANIFEST_FILES = {
    "D1": "D1_shards.txt",
    "D2": "D2_shards.txt",
    "D3": "D3_shards.txt",
    "D4": "D4_shards.txt",
    "AON_bench": "AON_bench_train_shards.txt",
    "AON_indic": "AON_indic_shards.txt",
}


def create_synthetic_shard(shard_dir: str, vocab_size: int = 131072, seed: int = 0):
    """Write a tokens.bin with random token IDs."""
    import random

    rng = random.Random(seed)

    os.makedirs(shard_dir, exist_ok=True)
    bin_path = os.path.join(shard_dir, "tokens.bin")

    total_tokens = BLOCK_SIZE * BLOCKS_PER_SHARD
    with open(bin_path, "wb") as f:
        for _ in range(total_tokens):
            token_id = rng.randint(0, vocab_size - 1)
            f.write(struct.pack("<I", token_id))  # uint32 little-endian

    file_size = os.path.getsize(bin_path)
    expected = total_tokens * BYTES_PER_TOKEN
    assert file_size == expected, f"Size mismatch: {file_size} != {expected}"
    return bin_path


def main():
    parser = argparse.ArgumentParser(
        description="Create synthetic curriculum test shards"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Root directory for test shards"
    )
    parser.add_argument(
        "--manifest-dir", default="manifests", help="Directory with manifest .txt files"
    )
    parser.add_argument(
        "--shards-per-pool", type=int, default=2, help="Number of shards per pool"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=131072,
        help="Vocabulary size for random tokens",
    )
    args = parser.parse_args()

    manifest_dir = args.manifest_dir
    output_dir = args.output_dir
    n = args.shards_per_pool

    print(f"Creating {n} synthetic shards per pool in: {output_dir}")
    print(f"Manifest dir: {manifest_dir}")

    total_created = 0
    seed_counter = 0

    for pool_name, manifest_file in POOL_MANIFEST_FILES.items():
        manifest_path = os.path.join(manifest_dir, manifest_file)
        if not os.path.exists(manifest_path):
            print(f"  WARNING: {manifest_path} not found, skipping pool {pool_name}")
            continue

        with open(manifest_path, "r") as f:
            all_shards = [line.strip() for line in f if line.strip()]

        # Take first N shards from manifest
        selected = all_shards[:n]
        print(f"  {pool_name}: {len(selected)} shards (from {len(all_shards)} total)")

        for rel_path in selected:
            shard_dir = os.path.join(output_dir, rel_path)
            create_synthetic_shard(
                shard_dir, vocab_size=args.vocab_size, seed=seed_counter
            )
            seed_counter += 1
            total_created += 1

    print(f"\nDone: {total_created} shards created in {output_dir}")
    print(
        f"Total size: ~{total_created * BLOCK_SIZE * BLOCKS_PER_SHARD * BYTES_PER_TOKEN / 1024:.0f} KB"
    )

    # Also create trimmed manifest files for testing (only the shards we created)
    test_manifest_dir = os.path.join(output_dir, "_test_manifests")
    os.makedirs(test_manifest_dir, exist_ok=True)

    for pool_name, manifest_file in POOL_MANIFEST_FILES.items():
        manifest_path = os.path.join(manifest_dir, manifest_file)
        if not os.path.exists(manifest_path):
            continue
        with open(manifest_path, "r") as f:
            all_shards = [line.strip() for line in f if line.strip()]
        selected = all_shards[:n]
        out_path = os.path.join(test_manifest_dir, manifest_file)
        with open(out_path, "w") as f:
            for s in selected:
                f.write(s + "\n")

    # Copy curriculum_v2_manifest.json (unchanged — pool structure is the same)
    import shutil

    src_json = os.path.join(manifest_dir, "curriculum_v2_manifest.json")
    if os.path.exists(src_json):
        shutil.copy2(
            src_json, os.path.join(test_manifest_dir, "curriculum_v2_manifest.json")
        )

    print(f"Test manifests written to: {test_manifest_dir}")
    print(f"\nTo use: set manifest_dir to {test_manifest_dir} in your test config")


if __name__ == "__main__":
    main()
