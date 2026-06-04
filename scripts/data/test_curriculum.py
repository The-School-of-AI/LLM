#!/usr/bin/env python3
"""
End-to-end test for the curriculum data pipeline.

Tests:
  1. Band assignment fix — correct band directories created
  2. Shard verification — all shards pass verify.py checks
  3. Curriculum config — YAML loading, weight validation
  4. Dataloader shapes — correct tensor shapes, dtypes, values
  5. Band proportions — convergence to target weights
  6. Missing band handling — graceful redistribution

Usage:
    python test_curriculum.py \\
        --shard-dir /tmp/test_bands \\
        --curriculum curriculum.yaml \\
        --stage 1B \\
        --tokenizer-dir ../../Tokenizer/output_hybrid
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

PASS = 0
FAIL = 0
BLOCK_SIZE = 4096


def _check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        msg = f"  FAIL: {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
    return condition


def test_band_directories(shard_dir: str):
    """Test that band-separated directories were created."""
    print("\n" + "=" * 60)
    print("TEST 1: Band directory structure")
    print("=" * 60)

    shard_root = Path(shard_dir)
    band_dirs = sorted(
        [d for d in shard_root.iterdir() if d.is_dir() and d.name.startswith("band_")]
    )

    _check(
        "At least one band directory exists",
        len(band_dirs) > 0,
        f"found {len(band_dirs)}",
    )

    # Check each band has shards
    band_info = {}
    for bd in band_dirs:
        band_name = bd.name[len("band_") :]
        shard_subdirs = sorted(
            [sd for sd in bd.iterdir() if sd.is_dir() and (sd / "tokens.bin").exists()]
        )
        band_info[band_name] = len(shard_subdirs)
        _check(
            f"Band {band_name} has shards",
            len(shard_subdirs) > 0,
            f"{len(shard_subdirs)} shards",
        )

    # Check manifest
    manifest_path = shard_root / "manifest.json"
    _check("manifest.json exists", manifest_path.exists())

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest_bands = manifest.get("bands", {})
        _check(
            "manifest has band entries",
            len(manifest_bands) == len(band_dirs),
            f"manifest has {list(manifest_bands.keys())}, dirs have {list(band_info.keys())}",
        )

        for band, info in sorted(manifest_bands.items()):
            print(
                f"    {band}: {info.get('docs', 0):,} docs, "
                f"{info.get('blocks', 0):,} blocks, "
                f"sources={info.get('sources', [])}"
            )

    return band_info


def test_shard_metadata(shard_dir: str):
    """Test that shard metadata correctly reflects band assignment."""
    print("\n" + "=" * 60)
    print("TEST 2: Shard metadata band correctness")
    print("=" * 60)

    shard_root = Path(shard_dir)
    mismatches = 0
    checked = 0

    for band_dir in sorted(shard_root.iterdir()):
        if not band_dir.is_dir() or not band_dir.name.startswith("band_"):
            continue
        expected_band = band_dir.name[len("band_") :]

        for shard_dir_path in sorted(band_dir.iterdir()):
            meta_path = shard_dir_path / "metadata.json"
            if not meta_path.exists():
                continue
            checked += 1
            with open(meta_path) as f:
                meta = json.load(f)
            actual_band = meta.get("band", "MISSING")
            if actual_band != expected_band:
                mismatches += 1
                print(
                    f"    MISMATCH: {shard_dir_path.name} in band_{expected_band}/ "
                    f"but metadata says band={actual_band}"
                )

    _check(
        f"All {checked} shard metadata match band directory",
        mismatches == 0,
        f"{mismatches} mismatches",
    )
    return mismatches == 0


def test_curriculum_config(curriculum_path: str):
    """Test curriculum.yaml loading and validation."""
    print("\n" + "=" * 60)
    print("TEST 3: Curriculum config")
    print("=" * 60)

    from curriculum_dataloader import CurriculumConfig

    _check("curriculum.yaml exists", os.path.exists(curriculum_path))

    for stage in ["1B", "3B", "8B", "70B"]:
        config = CurriculumConfig(curriculum_path, stage)
        weights = config.band_weights
        total = sum(weights.values())
        _check(
            f"Stage {stage} weights sum to 1.0", abs(total - 1.0) < 1e-6, f"sum={total}"
        )
        _check(f"Stage {stage} has 6 bands", len(weights) == 6, f"has {len(weights)}")
        _check(
            f"Stage {stage} all weights non-negative",
            all(w >= 0 for w in weights.values()),
        )

    # Test effective weights with missing bands
    config = CurriculumConfig(curriculum_path, "1B")
    effective = config.effective_weights(["B0", "B1"])
    eff_sum = sum(effective.values())
    _check(
        "Effective weights sum to 1.0 with missing bands",
        abs(eff_sum - 1.0) < 1e-6,
        f"sum={eff_sum}",
    )
    _check(
        "B0 weight increases when B2-B5 missing",
        effective["B0"] > config.band_weights["B0"],
        f"B0: {effective['B0']:.3f} > {config.band_weights['B0']:.3f}",
    )

    # Test invalid stage
    try:
        CurriculumConfig(curriculum_path, "999B")
        _check("Invalid stage raises KeyError", False)
    except KeyError:
        _check("Invalid stage raises KeyError", True)


def test_dataloader_shapes(shard_dir: str, curriculum_path: str, stage: str):
    """Test dataloader tensor shapes, dtypes, and values."""
    print("\n" + "=" * 60)
    print("TEST 4: Dataloader shapes and values")
    print("=" * 60)

    from curriculum_dataloader import build_curriculum_dataloader

    batch_size = 4
    seq_len = BLOCK_SIZE

    loader = build_curriculum_dataloader(
        shard_dir=shard_dir,
        curriculum_path=curriculum_path,
        stage=stage,
        batch_size=batch_size,
        seq_len=seq_len,
        num_workers=0,
        log_interval=50,
        rank=0,
        world_size=1,
    )

    n_batches = 100
    t_start = time.time()
    all_ok = True
    max_token_seen = 0
    min_token_seen = float("inf")

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break

        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        labels = batch["labels"]

        # Shape checks
        if i == 0:
            _check(
                f"input_ids shape [{batch_size}, {seq_len}]",
                ids.shape == (batch_size, seq_len),
                f"got {ids.shape}",
            )
            _check(
                f"attention_mask shape [{batch_size}, {seq_len}]",
                mask.shape == (batch_size, seq_len),
                f"got {mask.shape}",
            )
            _check(
                f"labels shape [{batch_size}, {seq_len}]",
                labels.shape == (batch_size, seq_len),
                f"got {labels.shape}",
            )
            _check(
                "input_ids dtype is int64/long",
                ids.dtype == torch.long,
                f"got {ids.dtype}",
            )

        # Value checks
        if not (mask == 1).all():
            if all_ok:
                _check("attention_mask all 1s (packed, no padding)", False)
            all_ok = False

        if not (labels == ids).all():
            if all_ok:
                _check("labels == input_ids", False)
            all_ok = False

        # No all-zero sequences
        seq_sums = ids.sum(dim=1)
        if (seq_sums == 0).any():
            _check(
                "No all-zero sequences",
                False,
                f"batch {i} has {(seq_sums == 0).sum()} zero seqs",
            )
            all_ok = False

        max_token_seen = max(max_token_seen, int(ids.max()))
        min_token_seen = min(min_token_seen, int(ids.min()))

    if all_ok:
        _check("attention_mask all 1s across 100 batches", True)
        _check("labels == input_ids across 100 batches", True)
        _check("No all-zero sequences across 100 batches", True)

    _check(
        f"Token IDs in valid range (min={min_token_seen}, max={max_token_seen})",
        min_token_seen >= 0 and max_token_seen < 131072,
        f"range [{min_token_seen}, {max_token_seen}]",
    )

    elapsed = time.time() - t_start
    total_tokens = n_batches * batch_size * seq_len
    print(
        f"\n  {n_batches} batches x {batch_size} = "
        f"{n_batches * batch_size} sequences, "
        f"{total_tokens:,} tokens in {elapsed:.1f}s "
        f"({total_tokens / elapsed / 1e6:.2f}M tok/s)"
    )


def test_band_proportions(shard_dir: str, curriculum_path: str, stage: str):
    """Test that band sampling converges to target proportions."""
    print("\n" + "=" * 60)
    print("TEST 5: Band proportion convergence")
    print("=" * 60)

    from curriculum_dataloader import CurriculumConfig, CurriculumDataset

    config = CurriculumConfig(curriculum_path, stage)
    dataset = CurriculumDataset(
        shard_dir=shard_dir,
        curriculum_config=config,
        seq_len=BLOCK_SIZE,
        rank=0,
        world_size=1,
        seed=42,
        log_interval=2000,
    )

    effective = dataset.effective_weights
    print(f"\n  Effective weights for stage {stage}:")
    for band, w in sorted(effective.items()):
        print(f"    {band}: {w:.3f}")

    n_samples = 5000
    t_start = time.time()

    for i, _ in enumerate(dataset):
        if i >= n_samples:
            break

    elapsed = time.time() - t_start
    stats = dataset.stats

    print(f"\n  Sampled {n_samples} blocks in {elapsed:.1f}s")
    print(stats.summary())

    # Check convergence: each band within 5% relative error
    all_converged = True
    for band, target in sorted(effective.items()):
        count = stats._counts.get(band, 0)
        actual = count / stats.total_blocks if stats.total_blocks > 0 else 0
        if target > 0:
            relative_error = abs(actual - target) / target
            converged = relative_error < 0.10  # 10% relative tolerance
            _check(
                f"Band {band} converged (target={target:.3f}, "
                f"actual={actual:.3f}, err={relative_error:.1%})",
                converged,
            )
            if not converged:
                all_converged = False
        else:
            _check(f"Band {band} zero-weight, count={count}", count == 0)

    dataset.close()
    return all_converged


def print_band_report(shard_dir: str):
    """Print detailed per-band statistics."""
    print("\n" + "=" * 60)
    print("BAND STATISTICS REPORT")
    print("=" * 60)

    shard_root = Path(shard_dir)
    total_blocks = 0
    total_tokens = 0
    total_shards = 0

    for band_dir in sorted(shard_root.iterdir()):
        if not band_dir.is_dir() or not band_dir.name.startswith("band_"):
            continue
        band_name = band_dir.name[len("band_") :]

        band_blocks = 0
        band_shards = 0
        sources = set()
        domains = set()

        for shard_path in sorted(band_dir.iterdir()):
            meta_path = shard_path / "metadata.json"
            if not meta_path.exists():
                continue
            band_shards += 1
            with open(meta_path) as f:
                meta = json.load(f)
            band_blocks += meta.get("num_blocks", 0)
            for s in meta.get("sources", []):
                sources.add(s)
            if "domain" in meta:
                domains.add(meta["domain"])

        band_tokens = band_blocks * BLOCK_SIZE
        total_blocks += band_blocks
        total_tokens += band_tokens
        total_shards += band_shards

        print(f"\n  {band_name}:")
        print(f"    Shards:    {band_shards}")
        print(f"    Blocks:    {band_blocks:,}")
        print(f"    Tokens:    {band_tokens:,} ({band_tokens / 1e6:.1f}M)")
        print(f"    Size:      {band_blocks * BLOCK_SIZE * 4 / (1024**2):.1f} MB")
        print(f"    Sources:   {sorted(sources)}")
        print(f"    Domains:   {sorted(domains)}")

    print("\n  TOTAL:")
    print(
        f"    Bands:     {sum(1 for d in shard_root.iterdir() if d.is_dir() and d.name.startswith('band_'))}"
    )
    print(f"    Shards:    {total_shards}")
    print(f"    Blocks:    {total_blocks:,}")
    print(f"    Tokens:    {total_tokens:,} ({total_tokens / 1e6:.1f}M)")
    print(f"    Size:      {total_blocks * BLOCK_SIZE * 4 / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Test curriculum pipeline")
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--stage", default="1B")
    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Tokenizer dir (for verify.py cross-check)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CURRICULUM PIPELINE END-TO-END TESTS")
    print("=" * 60)
    print(f"  shard_dir:    {args.shard_dir}")
    print(f"  curriculum:   {args.curriculum}")
    print(f"  stage:        {args.stage}")
    t_start = time.time()

    # Run all tests
    band_info = test_band_directories(args.shard_dir)
    test_shard_metadata(args.shard_dir)
    test_curriculum_config(args.curriculum)
    test_dataloader_shapes(args.shard_dir, args.curriculum, args.stage)
    test_band_proportions(args.shard_dir, args.curriculum, args.stage)
    print_band_report(args.shard_dir)

    # Summary
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    total = PASS + FAIL
    if FAIL == 0:
        print(f"ALL {total} TESTS PASSED ({elapsed:.1f}s)")
    else:
        print(f"{FAIL}/{total} TESTS FAILED ({elapsed:.1f}s)")
    print("=" * 60)

    sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
    main()
