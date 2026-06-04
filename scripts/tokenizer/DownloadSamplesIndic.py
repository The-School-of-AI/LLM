#!/usr/bin/env python3
"""
Download ~1/10th (by **file count**) of each Indic normalized dataset for tokenizer testing.

For each source:
  - List all .parquet objects under:
      s3://t1-dataacquisition-datasets/processed_dataset/normalized_data/source=<SOURCE>/
  - Let N = total number of parquet files.
  - Choose sample_n = ceil(N / 10) files (first sample_n in the listing).
  - Download those sample_n files with progress tracking (speed, ETA, progress bar).

Requires:
  pip install boto3 tqdm
  AWS credentials configured (env/credentials/role).
"""

import math
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import boto3
from tqdm import tqdm

# ---------- CONFIG ----------

BUCKET = "t1-dataacquisition-datasets"
BASE_PREFIX = "processed_dataset/normalized_data"
DEST_ROOT = "indic_tokenizer_samples_by_size"

SOURCES = [
    "ai-bharath-BPCC_seed",
    "ai-bharath-comparable",
    "ai-bharath-daily",
    "ai-bharath-ilci",
    "ai-bharath-massive",
    "ai-bharath-nllb_filtered",
    "ai-bharath-samanantar",
    "ai-bharath-wiki",
    "erav4_lang_as",
    "erav4_lang_hi",
    "erav4_lang_kn",
    "erav4_lang_mr",
    "erav4_lang_pa",
    "erav4_lang_te",
    "samvaad_hi",
    "sangraha_as",
    "sangraha_bn",
    "sangraha_gu",
    "sangraha_hi",
    "sangraha_kn",
    "sangraha_ml",
    "sangraha_mr",
    "sangraha_or",
    "sangraha_pa",
    "sangraha_ta",
    "sangraha_te",
    "sarvamai_mmlu",
]

# ----------------------------


def human_size(bytes_val: int) -> str:
    if bytes_val >= 1024**3:
        return f"{bytes_val / (1024 ** 3):.2f} GB"
    return f"{bytes_val / (1024 ** 2):.2f} MB"


def format_eta(seconds: float) -> str:
    if seconds < 0 or seconds > 86400:
        return "??:??"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def list_parquet_keys_and_sizes(
    s3_client, bucket: str, prefix: str
) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                items.append((key, obj["Size"]))
    return items


class ProgressCallback:
    """Per-file progress callback for s3.download_file."""

    def __init__(self, tqdm_bar, file_size):
        self.bar = tqdm_bar
        self.file_size = file_size

    def __call__(self, bytes_transferred):
        self.bar.update(bytes_transferred)


def main():
    s3 = boto3.client("s3")
    Path(DEST_ROOT).mkdir(parents=True, exist_ok=True)

    # ── Phase 1: List all files across all sources ──
    print("Scanning S3 for all sources...", flush=True)
    all_jobs = []  # list of (source, key, size)
    source_stats = {}

    for src in SOURCES:
        s3_prefix = f"{BASE_PREFIX}/source={src}/"
        files = list_parquet_keys_and_sizes(s3, BUCKET, s3_prefix)
        if not files:
            print(f"  {src}: no .parquet files, skipping.")
            continue

        total_bytes = sum(sz for _, sz in files)
        n_files = len(files)
        sample_n = max(1, math.ceil(n_files / 10))
        chosen = files[:sample_n]
        sample_bytes = sum(sz for _, sz in chosen)

        source_stats[src] = {
            "total_files": n_files,
            "total_bytes": total_bytes,
            "sample_files": sample_n,
            "sample_bytes": sample_bytes,
        }

        for key, sz in chosen:
            all_jobs.append((src, key, sz))

        print(
            f"  {src}: {sample_n}/{n_files} files, {human_size(sample_bytes)} / {human_size(total_bytes)}"
        )

    total_download_bytes = sum(sz for _, _, sz in all_jobs)
    total_files = len(all_jobs)

    # Check what's already downloaded
    to_download = []
    skipped_bytes = 0
    for src, key, sz in all_jobs:
        dest_dir = Path(DEST_ROOT) / f"source={src}"
        filename = key.rsplit("/", 1)[-1]
        dest_path = dest_dir / filename
        if dest_path.exists():
            skipped_bytes += sz
        else:
            to_download.append((src, key, sz))

    remaining_bytes = sum(sz for _, _, sz in to_download)

    print(f"\n{'='*60}")
    print(
        f"Total to download:  {len(to_download)} files, {human_size(remaining_bytes)}"
    )
    if skipped_bytes > 0:
        print(
            f"Already exists:     {total_files - len(to_download)} files, {human_size(skipped_bytes)} (skipped)"
        )
    print(f"{'='*60}\n", flush=True)

    if not to_download:
        print("Nothing to download — all files already exist!")
        return

    # ── Phase 2: Download with global progress bar ──
    global_start = time.time()
    downloaded_bytes = 0

    # Global progress bar (bytes)
    pbar = tqdm(
        total=remaining_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Total progress",
        bar_format="{l_bar}{bar:30}{r_bar}",
        file=sys.stdout,
    )

    for i, (src, key, sz) in enumerate(to_download):
        dest_dir = Path(DEST_ROOT) / f"source={src}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        filename = key.rsplit("/", 1)[-1]
        dest_path = dest_dir / filename

        elapsed = time.time() - global_start
        speed = downloaded_bytes / elapsed if elapsed > 0 else 0
        eta = (remaining_bytes - downloaded_bytes) / speed if speed > 0 else 0

        pbar.set_postfix_str(
            f"[{i+1}/{len(to_download)}] {src}/{filename} ({human_size(sz)}) | "
            f"Speed: {human_size(int(speed))}/s | ETA: {format_eta(eta)}",
            refresh=True,
        )

        callback = ProgressCallback(pbar, sz)
        s3.download_file(BUCKET, key, str(dest_path), Callback=callback)
        downloaded_bytes += sz

    pbar.close()

    elapsed_total = time.time() - global_start
    avg_speed = downloaded_bytes / elapsed_total if elapsed_total > 0 else 0

    # ── Phase 3: Summary ──
    global_total = sum(s["total_bytes"] for s in source_stats.values())
    global_sample = sum(s["sample_bytes"] for s in source_stats.values())

    print(f"\n{'='*60}")
    print("  DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"  Downloaded:     {human_size(downloaded_bytes)}")
    print(f"  Time:           {format_eta(elapsed_total)}")
    print(f"  Avg speed:      {human_size(int(avg_speed))}/s")
    print(
        f"  Sample/Total:   {human_size(global_sample)} / {human_size(global_total)} ({global_sample/global_total*100:.1f}%)"
    )
    print(f"  Saved to:       {os.path.abspath(DEST_ROOT)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
