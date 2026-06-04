#!/usr/bin/env python3
"""
Generate deterministic shard manifest for curriculum v2 (D1-D4 + AON + GP).

Connects to S3, enumerates all shards per band, builds pool assignments,
generates exclude lists, and writes the manifest JSON + per-pool shard lists.

Output:
  manifests/
    curriculum_v2_manifest.json   — master manifest with pool→shard mappings
    D1_shards.txt                 — one shard path per line
    D2_shards.txt
    D3_shards.txt
    D4_shards.txt
    AON_bench_train_shards.txt
    AON_indic_shards.txt
    GP_shards.txt
    indic_numerals_exclude.txt    — B1 shards to exclude from D2
    DROPPED_B2_shards.txt         — for reference only
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np

S3_BASE = "s3://t1-dataacquisition-datasets-2/shards_reordered"
SEED = 42
OUTPUT_DIR = Path(__file__).parent / "manifests"


def list_s3_shards(s3_prefix: str) -> list[str]:
    """List all shard subdirectories under an S3 prefix."""
    cmd = ["aws", "s3", "ls", f"{s3_prefix}/"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    shards = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("PRE "):
            name = line[4:].rstrip("/")
            shards.append(name)
    return sorted(shards)


def list_s3_shards_recursive(s3_prefix: str) -> list[str]:
    """List shard dirs by finding tokens.bin files recursively."""
    cmd = ["aws", "s3", "ls", "--recursive", f"{s3_prefix}/"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    shards = set()
    for line in result.stdout.strip().split("\n"):
        if "tokens.bin" in line:
            # Extract shard dir from path like .../shard_000001/tokens.bin
            parts = line.strip().split()
            if len(parts) >= 4:
                path = parts[3]  # relative path
                shard_dir = "/".join(path.split("/")[:-1])
                shards.add(shard_dir)
    return sorted(shards)


def deterministic_shuffle(shard_list: list[str], seed: int) -> list[str]:
    """Shuffle a shard list deterministically with the given seed."""
    rng = np.random.RandomState(seed)
    shuffled = list(shard_list)
    rng.shuffle(shuffled)
    return shuffled


def write_shard_list(filepath: Path, shards: list[str]) -> None:
    """Write one shard name per line."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for s in shards:
            f.write(s + "\n")
    print(f"  Wrote {len(shards)} shards to {filepath.name}")


def main():
    t0 = time.time()
    print("=" * 60)
    print("MANIFEST GENERATOR v2 — D1/D2/D3/D4/AON/GP")
    print(f"S3 base: {S3_BASE}")
    print(f"Seed: {SEED}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Enumerate all bands on S3 ───────────────────────────────────
    print("\n[1/6] Enumerating S3 bands...")

    bands = {}
    band_names = [
        "band_B0",
        "band_B1",
        "band_B2",
        "band_B3",
        "band_B4",
        "band_B5",
        "band_B6",
        "band_code_tab",
        "band_code_crlf",
        "band_indic_numerals",
    ]

    for bn in band_names:
        t1 = time.time()
        shards = list_s3_shards(f"{S3_BASE}/{bn}")
        bands[bn] = shards
        print(f"  {bn}: {len(shards)} shards ({time.time()-t1:.1f}s)")

    # Golden proxy has nested structure
    t1 = time.time()
    gp_shards_raw = list_s3_shards_recursive(f"{S3_BASE}/golden_proxy")
    # Extract just the shard names (e.g., "band_golden_proxy/shard_000000")
    gp_shards = sorted(
        set("/".join(s.split("/")[-2:]) if "/" in s else s for s in gp_shards_raw)
    )
    bands["golden_proxy"] = gp_shards
    print(f"  golden_proxy: {len(gp_shards)} shards ({time.time()-t1:.1f}s)")

    total = sum(len(v) for v in bands.values())
    print(f"\n  Total enumerated: {total} shards")

    # ─── Build pool assignments ──────────────────────────────────────
    print("\n[2/6] Building pool assignments...")

    # D1 = band_B0
    d1_shards = bands["band_B0"]
    print(f"  D1 (Web Foundation): {len(d1_shards)} shards from band_B0")

    # Exclude list: indic_numerals shard names that overlap with B1
    indic_exclude = set(bands["band_indic_numerals"])
    print(f"  Indic exclude set: {len(indic_exclude)} shards")

    # D2 = band_B1 minus indic overlap
    d2_shards = [s for s in bands["band_B1"] if s not in indic_exclude]
    d2_excluded = len(bands["band_B1"]) - len(d2_shards)
    print(
        f"  D2 (Web Diverse): {len(d2_shards)} shards from band_B1 "
        f"({d2_excluded} excluded as indic overlap)"
    )

    # D3 = band_B3 + band_code_tab + band_code_crlf
    d3_shards = (
        [f"band_B3/{s}" for s in bands["band_B3"]]
        + [f"band_code_tab/{s}" for s in bands["band_code_tab"]]
        + [f"band_code_crlf/{s}" for s in bands["band_code_crlf"]]
    )
    # For D1, D2, D4 we use band_name/shard_name format too
    d1_shards_full = [f"band_B0/{s}" for s in d1_shards]
    d2_shards_full = [f"band_B1/{s}" for s in d2_shards]
    d3_shards_full = sorted(d3_shards)
    print(
        f"  D3 (Code): {len(d3_shards_full)} shards "
        f"(B3:{len(bands['band_B3'])} + tab:{len(bands['band_code_tab'])} "
        f"+ crlf:{len(bands['band_code_crlf'])})"
    )

    # D4 = band_B4 + band_B5
    d4_shards = [f"band_B4/{s}" for s in bands["band_B4"]] + [
        f"band_B5/{s}" for s in bands["band_B5"]
    ]
    d4_shards_full = sorted(d4_shards)
    print(
        f"  D4 (STEM): {len(d4_shards_full)} shards "
        f"(B4:{len(bands['band_B4'])} + B5:{len(bands['band_B5'])})"
    )

    # AON = band_B6 + band_indic_numerals
    aon_bench = [f"band_B6/{s}" for s in bands["band_B6"]]
    aon_indic = [f"band_indic_numerals/{s}" for s in bands["band_indic_numerals"]]
    print(
        f"  AON (Always-ON): {len(aon_bench) + len(aon_indic)} shards "
        f"(B6:{len(aon_bench)} + indic:{len(aon_indic)})"
    )

    # GP
    gp_shards_full = [f"golden_proxy/{s}" for s in gp_shards]
    print(f"  GP (Golden Proxy): {len(gp_shards_full)} shards")

    # DROPPED
    dropped = [f"band_B2/{s}" for s in bands["band_B2"]]
    print(f"  DROPPED (B2): {len(dropped)} shards")

    # ─── Deterministic shuffle ───────────────────────────────────────
    print(f"\n[3/6] Deterministic shuffle (seed={SEED})...")

    d1_shuffled = deterministic_shuffle(d1_shards_full, SEED + 1)
    d2_shuffled = deterministic_shuffle(d2_shards_full, SEED + 2)
    d3_shuffled = deterministic_shuffle(d3_shards_full, SEED + 3)
    d4_shuffled = deterministic_shuffle(d4_shards_full, SEED + 4)
    aon_bench_shuffled = deterministic_shuffle(aon_bench, SEED + 10)
    aon_indic_shuffled = deterministic_shuffle(aon_indic, SEED + 11)
    gp_ordered = gp_shards_full  # GP is not shuffled — always same order

    print("  All pools shuffled deterministically.")

    # ─── Write shard lists ───────────────────────────────────────────
    print(f"\n[4/6] Writing shard lists to {OUTPUT_DIR}/...")

    write_shard_list(OUTPUT_DIR / "D1_shards.txt", d1_shuffled)
    write_shard_list(OUTPUT_DIR / "D2_shards.txt", d2_shuffled)
    write_shard_list(OUTPUT_DIR / "D3_shards.txt", d3_shuffled)
    write_shard_list(OUTPUT_DIR / "D4_shards.txt", d4_shuffled)
    write_shard_list(OUTPUT_DIR / "AON_bench_train_shards.txt", aon_bench_shuffled)
    write_shard_list(OUTPUT_DIR / "AON_indic_shards.txt", aon_indic_shuffled)
    write_shard_list(OUTPUT_DIR / "GP_shards.txt", gp_ordered)
    write_shard_list(OUTPUT_DIR / "indic_numerals_exclude.txt", sorted(indic_exclude))
    write_shard_list(OUTPUT_DIR / "DROPPED_B2_shards.txt", sorted(dropped))

    # ─── Build master manifest ───────────────────────────────────────
    print("\n[5/6] Building master manifest...")

    manifest = {
        "version": "2.0",
        "created": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "seed": SEED,
        "s3_base": S3_BASE,
        "pools": {
            "D1": {
                "name": "Web Foundation",
                "shard_count": len(d1_shuffled),
                "s3_bands": ["band_B0"],
                "shard_list_file": "D1_shards.txt",
            },
            "D2": {
                "name": "Web Diverse",
                "shard_count": len(d2_shuffled),
                "s3_bands": ["band_B1"],
                "excluded_count": d2_excluded,
                "shard_list_file": "D2_shards.txt",
            },
            "D3": {
                "name": "Code",
                "shard_count": len(d3_shuffled),
                "s3_bands": ["band_B3", "band_code_tab", "band_code_crlf"],
                "shard_list_file": "D3_shards.txt",
            },
            "D4": {
                "name": "STEM",
                "shard_count": len(d4_shuffled),
                "s3_bands": ["band_B4", "band_B5"],
                "shard_list_file": "D4_shards.txt",
            },
            "AON": {
                "name": "Always-ON",
                "sub_pools": {
                    "bench_train": {
                        "shard_count": len(aon_bench_shuffled),
                        "s3_band": "band_B6",
                        "shard_list_file": "AON_bench_train_shards.txt",
                    },
                    "indic_guaranteed": {
                        "shard_count": len(aon_indic_shuffled),
                        "s3_band": "band_indic_numerals",
                        "shard_list_file": "AON_indic_shards.txt",
                    },
                },
                "total_shards": len(aon_bench_shuffled) + len(aon_indic_shuffled),
                "injection_rate": 0.08,
                "internal_split": {"bench_train": 0.50, "indic_guaranteed": 0.50},
            },
            "GP": {
                "name": "Golden Proxy",
                "shard_count": len(gp_ordered),
                "s3_path": "golden_proxy/band_golden_proxy",
                "shard_list_file": "GP_shards.txt",
                "trainable": False,
            },
        },
        "dropped": {
            "B2": {
                "reason": "18.7% cross-doc leakage, 9.14% garbage, 15.69% repetition",
                "shard_count": len(dropped),
                "shard_list_file": "DROPPED_B2_shards.txt",
            },
        },
        "stages": {
            "1B": {
                "budget": "50B",
                "D1": 0.42,
                "D2": 0.30,
                "D3": 0.13,
                "D4": 0.07,
                "AON": 0.08,
            },
            "WU_3B": {
                "budget": "3B",
                "D1": 0.34,
                "D2": 0.30,
                "D3": 0.17,
                "D4": 0.11,
                "AON": 0.08,
            },
            "3B": {
                "budget": "40B",
                "D1": 0.22,
                "D2": 0.28,
                "D3": 0.25,
                "D4": 0.17,
                "AON": 0.08,
            },
            "WU_8B": {
                "budget": "3B",
                "D1": 0.14,
                "D2": 0.22,
                "D3": 0.30,
                "D4": 0.26,
                "AON": 0.08,
            },
            "8B": {
                "budget": "80B",
                "D1": 0.09,
                "D2": 0.18,
                "D3": 0.33,
                "D4": 0.32,
                "AON": 0.08,
            },
            "WU_70B": {
                "budget": "3B",
                "D1": 0.06,
                "D2": 0.12,
                "D3": 0.35,
                "D4": 0.39,
                "AON": 0.08,
            },
            "70B": {
                "budget": "30B",
                "D1": 0.06,
                "D2": 0.12,
                "D3": 0.35,
                "D4": 0.39,
                "AON": 0.08,
            },
        },
        "summary": {
            "total_opus_eligible": len(d1_shuffled)
            + len(d2_shuffled)
            + len(d3_shuffled)
            + len(d4_shuffled),
            "total_always_on": len(aon_bench_shuffled) + len(aon_indic_shuffled),
            "total_golden_proxy": len(gp_ordered),
            "total_dropped": len(dropped),
            "total_training": (
                len(d1_shuffled)
                + len(d2_shuffled)
                + len(d3_shuffled)
                + len(d4_shuffled)
                + len(aon_bench_shuffled)
                + len(aon_indic_shuffled)
            ),
            "grand_total_incl_dropped": (
                len(d1_shuffled)
                + len(d2_shuffled)
                + len(d3_shuffled)
                + len(d4_shuffled)
                + len(aon_bench_shuffled)
                + len(aon_indic_shuffled)
                + len(gp_ordered)
                + len(dropped)
            ),
        },
    }

    manifest_path = OUTPUT_DIR / "curriculum_v2_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote {manifest_path.name}")

    # ─── Summary ─────────────────────────────────────────────────────
    print("\n[6/6] Summary:")
    print(
        f"  D1 (Web Foundation):   {manifest['pools']['D1']['shard_count']:>6,} shards"
    )
    print(
        f"  D2 (Web Diverse):      {manifest['pools']['D2']['shard_count']:>6,} shards"
    )
    print(
        f"  D3 (Code):             {manifest['pools']['D3']['shard_count']:>6,} shards"
    )
    print(
        f"  D4 (STEM):             {manifest['pools']['D4']['shard_count']:>6,} shards"
    )
    print(
        f"  AON (Always-ON):       {manifest['pools']['AON']['total_shards']:>6,} shards"
    )
    print(
        f"    bench_train:         {manifest['pools']['AON']['sub_pools']['bench_train']['shard_count']:>6,}"
    )
    print(
        f"    indic_guaranteed:    {manifest['pools']['AON']['sub_pools']['indic_guaranteed']['shard_count']:>6,}"
    )
    print(
        f"  GP (Golden Proxy):     {manifest['pools']['GP']['shard_count']:>6,} shards"
    )
    print(
        f"  DROPPED (B2):          {manifest['dropped']['B2']['shard_count']:>6,} shards"
    )
    print("  ──────────────────────────────")
    print(
        f"  Training total:        {manifest['summary']['total_training']:>6,} shards"
    )
    print(
        f"  Grand total:           {manifest['summary']['grand_total_incl_dropped']:>6,} shards"
    )
    print(f"\n  Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
