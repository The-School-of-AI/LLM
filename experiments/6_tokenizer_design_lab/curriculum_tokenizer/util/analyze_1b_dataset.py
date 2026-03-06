#!/usr/bin/env python3
"""
analyze_1b_dataset.py — T3 / T1 size analysis for 1B training planning
========================================================================

Reads every T3 coreset-index parquet file and extracts the unique T1 source
parquet files referenced by them.  Then reports file sizes so you can plan
storage / tokenisation capacity.

Reports
-------
  T3 summary  : file count, rows per file, total rows, total token budget
  T1 summary  : unique file count, avg / max / min size, total size

Modes
-----
  Local  — paths are ordinary directories on disk.
  S3     — paths begin with "s3://".  T1 sizes are fetched via HEAD requests
           (no data downloaded for T1), so AWS cost is negligible.

Usage
-----
  # Local
  python util/analyze_1b_dataset.py \\
      --t3-dir  dataset/source/t3_coresets \\
      --t1-base dataset/source/t1_rawdata/normalized_data

  # S3 (cross-account / IAM-role assumed)
  python util/analyze_1b_dataset.py \\
      --t3-dir  s3://my-bucket/coresets/1B/ \\
      --t1-base s3://my-bucket/t1_rawdata/normalized_data \\
      --region  us-east-1

AWS cost note
-------------
  * T3 files are downloaded to memory to read chunk_id / t1_file_path columns.
    These are small index files (~1–5 MB each) — cost is S3 GET + egress.
  * T1 sizes are obtained via S3 HeadObject (no data download).
    Cost: ~$0.0004 per 1 000 HEAD requests (effectively free).
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd

# ── Helpers: detect local vs S3 ───────────────────────────────────────────────

def is_s3(uri: str) -> bool:
    return uri.startswith("s3://")


def parse_s3(uri: str) -> Tuple[str, str]:
    """Return (bucket, key_prefix) from an s3:// URI."""
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")


# ── Local backend ─────────────────────────────────────────────────────────────

def list_local_parquets(directory: str) -> List[str]:
    d = Path(directory)
    if not d.is_dir():
        sys.exit(f"ERROR: directory not found: {directory}")
    files = sorted(d.glob("**/*.parquet"))
    return [str(f) for f in files]


def read_local_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def local_file_size(path: str) -> Optional[int]:
    """Return file size in bytes, or None if the file doesn't exist."""
    p = Path(path)
    if p.exists():
        return p.stat().st_size
    return None


# ── S3 backend ────────────────────────────────────────────────────────────────

def _boto3_client(region: Optional[str]):
    """Lazy import boto3 so local-only usage doesn't require it."""
    try:
        import boto3  # noqa: PLC0415
    except ImportError:
        sys.exit("ERROR: boto3 is required for S3 mode.  pip install boto3")
    kwargs = {}
    if region:
        kwargs["region_name"] = region
    return boto3.client("s3", **kwargs)


def list_s3_parquets(s3_uri: str, region: Optional[str]) -> List[str]:
    """List all .parquet objects under the given S3 prefix; returns s3:// URIs."""
    s3 = _boto3_client(region)
    bucket, prefix = parse_s3(s3_uri)
    paginator = s3.get_paginator("list_objects_v2")
    results: List[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                results.append(f"s3://{bucket}/{key}")
    return sorted(results)


def read_s3_parquet(s3_uri: str, region: Optional[str]) -> pd.DataFrame:
    """Download a parquet object from S3 into memory and return a DataFrame."""
    s3 = _boto3_client(region)
    bucket, key = parse_s3(s3_uri)
    response = s3.get_object(Bucket=bucket, Key=key)
    data = response["Body"].read()
    return pd.read_parquet(io.BytesIO(data))


def s3_object_size(s3_uri: str, region: Optional[str]) -> Optional[int]:
    """Return byte size of an S3 object via HeadObject (no data transfer)."""
    s3 = _boto3_client(region)
    bucket, key = parse_s3(s3_uri)
    try:
        meta = s3.head_object(Bucket=bucket, Key=key)
        return meta["ContentLength"]
    except s3.exceptions.ClientError:
        return None


# ── CSV export ────────────────────────────────────────────────────────────────

def write_csvs(
    out_prefix: str,
    t3_dir: str,
    t1_base: str,
    t3_stats: List[Dict],
    t1_sizes: Dict[str, Optional[int]],
    t1_refs: Dict[str, int],
    total_rows: int,
    total_tokens: int,
    total_size: int,
    avg_size: float,
) -> None:
    """Write three CSV files: _t3.csv, _t1.csv, _summary.csv."""
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── t3_report.csv ─────────────────────────────────────────────────────────
    t3_path = f"{out_prefix}_t3.csv"
    n_t3 = t3_stats[0]["total"] if t3_stats else 0
    with open(t3_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# T3 Coreset File Report", f"generated={ts}"])
        w.writerow(["# t3_dir", t3_dir])
        w.writerow([])
        w.writerow(["seq", "total_files", "t3_file", "rows", "tokens"])
        for s in t3_stats:
            w.writerow([f"{s['seq']}/{n_t3}", n_t3, s["file"], s["rows"], s["tokens"]])
        w.writerow([])
        w.writerow(["TOTAL", "", "", total_rows, total_tokens])
    print(f"  Written: {t3_path}")

    # ── t1_report.csv ─────────────────────────────────────────────────────────
    t1_path = f"{out_prefix}_t1.csv"
    n_t1 = len(t1_sizes)
    with open(t1_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# T1 Source File Report", f"generated={ts}"])
        w.writerow(["# t1_base", t1_base])
        w.writerow([])
        w.writerow(["seq", "total_files", "t1_file_path", "size_bytes", "size_mb", "size_human", "status"])
        for j, rel_path in enumerate(sorted(t1_sizes.keys()), 1):
            size = t1_sizes[rel_path]
            if size is not None:
                w.writerow([f"{j}/{n_t1}", n_t1, rel_path, size,
                            f"{size / BYTES_PER_MB:.4f}", format_size(size), "found"])
            else:
                w.writerow([f"{j}/{n_t1}", n_t1, rel_path, "", "", "", "not_found"])
    print(f"  Written: {t1_path}")

    # ── summary.csv ───────────────────────────────────────────────────────────
    n_found   = sum(1 for s in t1_sizes.values() if s is not None)
    n_missing = len(t1_refs) - n_found
    est_total = int(avg_size * len(t1_refs)) if avg_size > 0 else 0
    blocks_4k = total_tokens // 4096 if total_tokens > 0 else 0
    shards_512 = int(blocks_4k * 4096 * 2 / BYTES_PER_MB / 512) if blocks_4k > 0 else 0

    summary_path = f"{out_prefix}_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# Analysis Summary", f"generated={ts}"])
        w.writerow(["# t3_dir", t3_dir])
        w.writerow(["# t1_base", t1_base])
        w.writerow([])
        w.writerow(["metric", "value", "note"])
        w.writerow(["t3_file_count",         len(t3_stats),       ""])
        w.writerow(["t3_total_rows",          total_rows,          ""])
        w.writerow(["t3_total_tokens",        total_tokens,        ""])
        w.writerow(["t3_total_tokens_B",      f"{total_tokens/1e9:.6f}", "billion tokens"])
        w.writerow(["t1_unique_files",        len(t1_refs),        ""])
        w.writerow(["t1_files_found",         n_found,             ""])
        w.writerow(["t1_files_not_found",     n_missing,           "only relevant for local runs"])
        w.writerow(["t1_avg_size_bytes",      int(avg_size),       "of found files"])
        w.writerow(["t1_avg_size_mb",         f"{avg_size/BYTES_PER_MB:.4f}", "of found files"])
        w.writerow(["t1_total_size_found_bytes", total_size,       "sum of found files"])
        w.writerow(["t1_total_size_found_mb", f"{total_size/BYTES_PER_MB:.4f}", ""])
        w.writerow(["t1_total_size_found_gb", f"{total_size/BYTES_PER_GB:.4f}", ""])
        if n_missing > 0:
            w.writerow(["t1_est_total_size_bytes", est_total,      f"avg x {len(t1_refs)} files"])
            w.writerow(["t1_est_total_size_gb",    f"{est_total/BYTES_PER_GB:.4f}", "estimated"])
        w.writerow(["blocks_4096_tokens",     blocks_4k,           "floor(total_tokens / 4096)"])
        w.writerow(["shards_512mb_uint16",    shards_512,          "blocks * 4096 * 2B / 512MB"])
    print(f"  Written: {summary_path}")


# ── Core analysis ─────────────────────────────────────────────────────────────

T3_CHUNK_COL   = "chunk_id"
T3_T1_PATH_COL = "t1_file_path"
T3_TOKEN_COL   = "token_count"

BYTES_PER_MB = 1024 ** 2
BYTES_PER_GB = 1024 ** 3


def format_size(n_bytes: int) -> str:
    if n_bytes >= BYTES_PER_GB:
        return f"{n_bytes / BYTES_PER_GB:.2f} GB"
    if n_bytes >= BYTES_PER_MB:
        return f"{n_bytes / BYTES_PER_MB:.2f} MB"
    return f"{n_bytes / 1024:.2f} KB"


def analyze(t3_dir: str, t1_base: str, region: Optional[str], out_prefix: str) -> None:
    use_s3 = is_s3(t3_dir)
    t1_s3  = is_s3(t1_base)

    # ── 1. Discover T3 files ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Step 1 — Discovering T3 parquet files ...")
    if use_s3:
        t3_files = list_s3_parquets(t3_dir, region)
    else:
        t3_files = list_local_parquets(t3_dir)

    if not t3_files:
        sys.exit(f"ERROR: no .parquet files found under: {t3_dir}")
    print(f"  Found {len(t3_files)} T3 file(s)")

    # ── 2. Read each T3 file: collect rows & unique T1 paths ─────────────────
    print("\nStep 2 — Reading T3 files to extract T1 references ...")

    t3_stats: List[Dict] = []           # per-file stats
    t1_refs: Dict[str, int] = {}        # relative_t1_path -> appearance count

    total_tokens = 0
    missing_col_warned = False

    for i, t3_path in enumerate(t3_files, 1):
        short = t3_path.split("/")[-1] if "/" in t3_path else os.path.basename(t3_path)
        print(f"  [{i:>4}/{len(t3_files)}] {short}", end=" ... ", flush=True)

        if use_s3:
            df = read_s3_parquet(t3_path, region)
        else:
            df = read_local_parquet(t3_path)

        n_rows = len(df)
        tokens = int(df[T3_TOKEN_COL].sum()) if T3_TOKEN_COL in df.columns else 0
        total_tokens += tokens

        # Collect unique T1 paths from this batch
        if T3_T1_PATH_COL in df.columns:
            for rel_path in df[T3_T1_PATH_COL].dropna().unique():
                t1_refs[rel_path] = t1_refs.get(rel_path, 0) + 1
        elif not missing_col_warned:
            print(f"\n  [WARN] Column '{T3_T1_PATH_COL}' not found in T3 files.")
            missing_col_warned = True

        t3_stats.append({"seq": i, "total": len(t3_files), "file": short, "rows": n_rows, "tokens": tokens})
        print(f"{n_rows:>7,} rows | {tokens:>12,} tokens")

    # ── 3. Resolve T1 full paths and get sizes ───────────────────────────────
    print(f"\nStep 3 — Fetching T1 file sizes ({len(t1_refs)} unique file(s)) ...")

    t1_sizes: Dict[str, Optional[int]] = {}

    for j, rel_path in enumerate(sorted(t1_refs.keys()), 1):
        # Build absolute / full URI
        if t1_s3:
            full_path = t1_base.rstrip("/") + "/" + rel_path.lstrip("/")
        else:
            full_path = str(Path(t1_base) / rel_path)

        short_t1 = rel_path.split("/")[-1] if "/" in rel_path else rel_path
        print(f"  [{j:>5}/{len(t1_refs)}] {short_t1}", end=" ... ", flush=True)

        if t1_s3:
            size = s3_object_size(full_path, region)
        else:
            size = local_file_size(full_path)

        t1_sizes[rel_path] = size

        if size is not None:
            print(f"{format_size(size)}")
        else:
            print("NOT FOUND")

    # ── 4. Print T3 report ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("T3 FILE REPORT")
    print(f"{'='*70}")
    print(f"{'File':<55} {'Rows':>8}  {'Tokens':>14}")
    print(f"{'-'*55} {'-'*8}  {'-'*14}")
    for stat in t3_stats:
        print(f"  {stat['file']:<53} {stat['rows']:>8,}  {stat['tokens']:>14,}")
    print(f"{'-'*55} {'-'*8}  {'-'*14}")
    total_rows = sum(s["rows"] for s in t3_stats)
    print(f"  {'TOTAL':<53} {total_rows:>8,}  {total_tokens:>14,}")
    print(f"\n  T3 file count : {len(t3_files):,}")
    print(f"  Total rows    : {total_rows:,}")
    print(f"  Total tokens  : {total_tokens:,}  (~{total_tokens/1e9:.3f} B tokens)")

    # ── 5. Print T1 report ───────────────────────────────────────────────────
    valid_sizes = [s for s in t1_sizes.values() if s is not None]
    missing     = [p for p, s in t1_sizes.items() if s is None]

    print(f"\n{'='*70}")
    print("T1 FILE REPORT (sizes only — no content downloaded)")
    print(f"{'='*70}")
    print(f"\n  Unique T1 file count : {len(t1_refs):,}")

    total_size = 0
    avg_size   = 0.0
    if valid_sizes:
        avg_size   = sum(valid_sizes) / len(valid_sizes)
        max_size   = max(valid_sizes)
        min_size   = min(valid_sizes)
        total_size = sum(valid_sizes)
        max_path   = max(t1_sizes, key=lambda p: t1_sizes[p] or 0)
        min_path   = min((p for p, s in t1_sizes.items() if s is not None),
                         key=lambda p: t1_sizes[p])

        print(f"  Files with size found    : {len(valid_sizes):,} / {len(t1_refs):,}")
        print(f"  Average size             : {format_size(int(avg_size))}")
        print(f"  Largest file             : {format_size(max_size)}")
        print(f"    -> {max_path}")
        print(f"  Smallest file            : {format_size(min_size)}")
        print(f"    -> {min_path}")
        print(f"  Total T1 size (found)    : {format_size(total_size)}")

        # Estimate total when some files were not reachable locally
        n_missing = len(t1_refs) - len(valid_sizes)
        if n_missing > 0:
            est_total = int(avg_size * len(t1_refs))
            print(f"  Est. total T1 size (all) : {format_size(est_total)}"
                  f"  [{n_missing:,} file(s) not found; using avg size]")

    if missing:
        print(f"\n  [WARN] {len(missing)} T1 file(s) NOT FOUND at the given base URI:")
        for p in missing[:10]:
            print(f"    {p}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")

    # ── 6. Planning summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TOKENISATION PLANNING SUMMARY")
    print(f"{'='*70}")
    print(f"  T3 coreset files          : {len(t3_files):,}")
    print(f"  Total coreset rows        : {total_rows:,}")
    print(f"  Total token budget        : {total_tokens:,}  ({total_tokens/1e9:.3f} B)")
    print(f"  Unique T1 source files    : {len(t1_refs):,}")
    if valid_sizes:
        n_missing = len(t1_refs) - len(valid_sizes)
        if n_missing == 0:
            print(f"  Total T1 data size        : {format_size(total_size)}")
        else:
            est_total = int(avg_size * len(t1_refs))
            print(f"  Total T1 size (found)     : {format_size(total_size)}"
                  f"  ({len(valid_sizes):,}/{len(t1_refs):,} files)")
            print(f"  Est. total T1 size (all)  : {format_size(est_total)}"
                  f"  [avg {format_size(int(avg_size))} x {len(t1_refs):,}]")
    if total_tokens > 0:
        print(f"\n  Estimated at block_size=4096 tokens:")
        blocks_4k = total_tokens // 4096
        print(f"    Blocks                 : ~{blocks_4k:,}")
        print(f"    (at 512 MB shards)     : ~{blocks_4k * 4096 * 2 / BYTES_PER_MB / 512:.0f} shard(s)"
              f"  [uint16, 2 bytes/token]")
    print(f"{'='*70}\n")

    # ── 7. Write CSVs ─────────────────────────────────────────────────────────
    print("Writing CSV reports ...")
    write_csvs(
        out_prefix=out_prefix,
        t3_dir=t3_dir,
        t1_base=t1_base,
        t3_stats=t3_stats,
        t1_sizes=t1_sizes,
        t1_refs=t1_refs,
        total_rows=total_rows,
        total_tokens=total_tokens,
        total_size=total_size,
        avg_size=avg_size,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Analyse T3 coreset files and T1 source sizes for 1B training planning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--t3-dir", required=True,
        help="Directory (local) or S3 prefix (s3://) containing T3 coreset parquets",
    )
    p.add_argument(
        "--t1-base", required=True,
        help="Base directory (local) or S3 prefix (s3://) for T1 raw parquets. "
             "t1_file_path values in T3 are appended to this base.",
    )
    p.add_argument(
        "--region", default=None,
        help="AWS region (optional, for S3 mode). Defaults to AWS_DEFAULT_REGION env var.",
    )
    p.add_argument(
        "--out", default="analyze_1b_report",
        help="Output prefix for CSV files. Three files are written: "
             "<prefix>_t3.csv, <prefix>_t1.csv, <prefix>_summary.csv. "
             "(default: analyze_1b_report)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    print(f"T3 source : {args.t3_dir}")
    print(f"T1 base   : {args.t1_base}")
    print(f"CSV out   : {args.out}_{{t3,t1,summary}}.csv")
    if is_s3(args.t3_dir) or is_s3(args.t1_base):
        print(f"Mode      : S3  (region={args.region or 'from env/profile'})")
    else:
        print(f"Mode      : Local")
    analyze(args.t3_dir, args.t1_base, args.region, args.out)


if __name__ == "__main__":
    main()
