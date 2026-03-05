#!/usr/bin/env python3
"""
Flatten Shards — Post-Processing Step for Curriculum Tokenizer
==============================================================

Converts the per-batch staging layout produced by tokenize_curriculum.py
into a flat, globally-numbered shard directory suitable for training pipelines.

Staging layout (written by tokenize_curriculum.py):
  <src-dir>/
    selected_indices_small_batch000000/
      shard_001/ -> tokens.bin, tokens.idx, metadata.json
      shard_002/ -> ...
    selected_indices_small_batch000001/
      shard_001/ -> ...
      shard_002/ -> ...
    ...

Output layout (written by this script):
  <dst-dir>/
    shards/
      shard_001/ -> tokens.bin, tokens.idx, metadata.json  (shard_name patched)
      shard_002/ -> ...
      shard_003/ -> ...        <- first shard of second batch, renumbered
      ...
    flatten_manifest.json      <- batch -> global shard range mapping

Design decisions:
  - Resumable: shards already present in <dst-dir>/shards/ are skipped.
  - Streaming: copy shard N -> delete staging shard N before moving to N+1,
    so disk/S3 never holds more than one extra shard at a time.
  - metadata.json patched in-place: only shard_name changes; all other fields
    (tokenizer_hash, source_file, band, domain, ...) are preserved.
  - Deterministic ordering: batches sorted lexicographically; shards within
    each batch sorted by their original shard_NNN name.
  - Local runs use os.rename (O(1) inode update, zero data movement).
  - S3 runs use CopyObject + DeleteObject (server-side, no egress cost).

Usage:
  # Local
  python flatten_shards.py \\
      --src-dir ./tok_staging \\
      --dst-dir ./output

  # S3
  python flatten_shards.py \\
      --src-uri s3://my-bucket/tok_staging \\
      --dst-uri s3://my-bucket/output

  # Dry-run (print plan, no writes)
  python flatten_shards.py --src-dir ./tok_staging --dst-dir ./output --dry-run

  # Resume (skip already-present shards in dst)
  python flatten_shards.py --src-dir ./tok_staging --dst-dir ./output
  # (resumable by default — safe to re-run)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import tempfile
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHARD_FILES = ("tokens.bin", "tokens.idx", "metadata.json")


# ---------------------------------------------------------------------------
# URI helpers
# ---------------------------------------------------------------------------


def is_s3_uri(uri: str) -> bool:
    return uri.startswith("s3://")


def parse_s3_url(url: str) -> Tuple[str, str]:
    parsed = urlparse(url)
    if parsed.scheme != "s3":
        raise ValueError(f"Not an S3 URI: {url}")
    return parsed.netloc, parsed.path.lstrip("/")


def _s3_key_join(prefix: str, *parts: str) -> str:
    """Join S3 key segments, normalising slashes."""
    result = prefix.rstrip("/")
    for part in parts:
        result = result.rstrip("/") + "/" + part.lstrip("/")
    return result


def _s3_uri_join(base_uri: str, *parts: str) -> str:
    """Join an S3 URI with path segments."""
    bucket, prefix = parse_s3_url(base_uri)
    key = _s3_key_join(prefix, *parts)
    return f"s3://{bucket}/{key}"


# ---------------------------------------------------------------------------
# S3 operations
# ---------------------------------------------------------------------------


def _get_s3():
    import boto3
    return boto3.client("s3")


def s3_key_exists(s3, bucket: str, key: str) -> bool:
    from botocore.exceptions import ClientError
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def s3_list_immediate_subdirs(s3, bucket: str, prefix: str) -> List[str]:
    """Return sorted list of immediate sub-prefixes under prefix (one level deep)."""
    prefix = prefix.rstrip("/") + "/"
    paginator = s3.get_paginator("list_objects_v2")
    subdirs = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            # cp["Prefix"] looks like "prefix/subdir/"
            rel = cp["Prefix"][len(prefix):].rstrip("/")
            if rel:
                subdirs.add(rel)
    return sorted(subdirs)


def s3_list_objects_under(s3, bucket: str, prefix: str) -> List[str]:
    """Return all object keys under prefix."""
    prefix = prefix.rstrip("/") + "/"
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def s3_copy_object(s3, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str) -> None:
    s3.copy_object(
        CopySource={"Bucket": src_bucket, "Key": src_key},
        Bucket=dst_bucket,
        Key=dst_key,
    )


def s3_delete_object(s3, bucket: str, key: str) -> None:
    s3.delete_object(Bucket=bucket, Key=key)


def s3_read_json(s3, bucket: str, key: str) -> dict:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(resp["Body"].read().decode("utf-8"))


def s3_put_json(s3, bucket: str, key: str, data: dict) -> None:
    body = json.dumps(data, indent=2).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")


# ---------------------------------------------------------------------------
# Staging layout discovery
# ---------------------------------------------------------------------------


def discover_staging_batches_local(src_dir: str) -> List[Tuple[str, str]]:
    """
    Walk src_dir and return (batch_name, batch_path) pairs sorted by batch_name.
    A batch directory is any immediate subdir of src_dir that itself contains
    at least one shard_NNN/ subdirectory.
    """
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    batches = []
    for entry in sorted(os.listdir(src_dir)):
        batch_path = os.path.join(src_dir, entry)
        if not os.path.isdir(batch_path):
            continue
        # Check it has at least one shard_NNN subdir
        has_shards = any(
            sub.startswith("shard_") and os.path.isdir(os.path.join(batch_path, sub))
            for sub in os.listdir(batch_path)
        )
        if has_shards:
            batches.append((entry, batch_path))
    return batches


def discover_staging_batches_s3(s3, src_bucket: str, src_prefix: str) -> List[Tuple[str, str]]:
    """
    Return (batch_name, full_s3_prefix) pairs from S3, sorted by batch_name.
    """
    batch_names = s3_list_immediate_subdirs(s3, src_bucket, src_prefix)
    result = []
    for name in batch_names:
        prefix = _s3_key_join(src_prefix, name)
        # Verify it contains shard subdirs
        shard_subdirs = [
            d for d in s3_list_immediate_subdirs(s3, src_bucket, prefix)
            if d.startswith("shard_")
        ]
        if shard_subdirs:
            result.append((name, f"s3://{src_bucket}/{prefix}"))
    return result


def list_shard_dirs_in_batch_local(batch_path: str) -> List[Tuple[str, str]]:
    """Return sorted (shard_name, shard_path) pairs inside a local batch dir."""
    shards = []
    for sub in sorted(os.listdir(batch_path)):
        if sub.startswith("shard_") and os.path.isdir(os.path.join(batch_path, sub)):
            shards.append((sub, os.path.join(batch_path, sub)))
    return shards


def list_shard_dirs_in_batch_s3(s3, src_bucket: str, batch_prefix: str) -> List[Tuple[str, str]]:
    """Return sorted (shard_name, s3_prefix) pairs inside an S3 batch prefix."""
    shard_names = sorted([
        d for d in s3_list_immediate_subdirs(s3, src_bucket, batch_prefix)
        if d.startswith("shard_")
    ])
    return [
        (name, f"s3://{src_bucket}/{_s3_key_join(batch_prefix, name)}")
        for name in shard_names
    ]


# ---------------------------------------------------------------------------
# Core flatten operations
# ---------------------------------------------------------------------------


def _patch_metadata_shard_name(meta: dict, new_shard_name: str) -> dict:
    """Return a copy of metadata with shard_name updated to new_shard_name."""
    patched = dict(meta)
    patched["shard_name"] = new_shard_name
    return patched


def flatten_shard_local(
    src_shard_dir: str,
    dst_shard_dir: str,
    new_shard_name: str,
    dry_run: bool = False,
) -> None:
    """
    Move a shard from staging to flat output directory (local).

    Strategy: rename the directory (O(1) if on same filesystem), then patch
    metadata.json in-place. Falls back to copy+delete if src and dst are on
    different filesystems.
    """
    if dry_run:
        print(f"    [DRY-RUN] {src_shard_dir} -> {dst_shard_dir} (shard_name={new_shard_name})")
        return

    os.makedirs(os.path.dirname(dst_shard_dir), exist_ok=True)

    try:
        os.rename(src_shard_dir, dst_shard_dir)
    except OSError:
        # Cross-device: fall back to copy then delete
        shutil.copytree(src_shard_dir, dst_shard_dir)
        shutil.rmtree(src_shard_dir)

    # Patch metadata.json in-place
    meta_path = os.path.join(dst_shard_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta = _patch_metadata_shard_name(meta, new_shard_name)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def flatten_shard_s3(
    s3,
    src_bucket: str,
    src_prefix: str,
    dst_bucket: str,
    dst_prefix: str,
    new_shard_name: str,
    dry_run: bool = False,
) -> None:
    """
    Move a shard from staging to flat output on S3 (CopyObject + DeleteObject).

    Streaming strategy: copy each file, then immediately delete its source.
    metadata.json is patched (GET -> modify -> PUT) before the source is deleted.
    """
    src_prefix = src_prefix.rstrip("/")
    dst_prefix = dst_prefix.rstrip("/")

    for fname in SHARD_FILES:
        src_key = f"{src_prefix}/{fname}"
        dst_key = f"{dst_prefix}/{fname}"

        if dry_run:
            print(f"    [DRY-RUN] s3://{src_bucket}/{src_key} -> s3://{dst_bucket}/{dst_key}")
            continue

        if fname == "metadata.json":
            # Patch shard_name before writing to destination
            meta = s3_read_json(s3, src_bucket, src_key)
            meta = _patch_metadata_shard_name(meta, new_shard_name)
            s3_put_json(s3, dst_bucket, dst_key, meta)
        else:
            s3_copy_object(s3, src_bucket, src_key, dst_bucket, dst_key)

        # Delete source immediately (streaming — never hold both at once)
        s3_delete_object(s3, src_bucket, src_key)


# ---------------------------------------------------------------------------
# Flatten manifest
# ---------------------------------------------------------------------------


def write_flatten_manifest_local(dst_dir: str, manifest: dict) -> None:
    path = os.path.join(dst_dir, "flatten_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved flatten_manifest.json -> {path}")


def write_flatten_manifest_s3(s3, dst_bucket: str, dst_prefix: str, manifest: dict) -> None:
    key = _s3_key_join(dst_prefix, "flatten_manifest.json")
    s3_put_json(s3, dst_bucket, key, manifest)
    print(f"\nSaved flatten_manifest.json -> s3://{dst_bucket}/{key}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_flatten(args: argparse.Namespace) -> None:
    use_s3 = is_s3_uri(args.src_uri or "") or is_s3_uri(args.dst_uri or "")

    s3 = None
    if use_s3:
        s3 = _get_s3()

    # ------------------------------------------------------------------
    # Resolve source / destination
    # ------------------------------------------------------------------
    src_is_s3 = is_s3_uri(args.src_uri or "")
    dst_is_s3 = is_s3_uri(args.dst_uri or "")

    if src_is_s3:
        src_bucket, src_prefix = parse_s3_url(args.src_uri)
    else:
        src_dir = args.src_dir

    if dst_is_s3:
        dst_bucket, dst_prefix = parse_s3_url(args.dst_uri)
        # Output shards go under <dst>/shards/
        shards_prefix = _s3_key_join(dst_prefix, "shards")
    else:
        dst_dir = args.dst_dir
        shards_dir = os.path.join(dst_dir, "shards")
        os.makedirs(shards_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Discover staging batches
    # ------------------------------------------------------------------
    print("Discovering staging batches...")
    if src_is_s3:
        batches = discover_staging_batches_s3(s3, src_bucket, src_prefix)
    else:
        batches = discover_staging_batches_local(src_dir)

    if not batches:
        print("No batch directories with shards found. Nothing to do.")
        return

    print(f"Found {len(batches)} batch(es).")

    # ------------------------------------------------------------------
    # Determine start index (resume support)
    # ------------------------------------------------------------------
    # Count how many shards already exist in the destination
    start_global_idx = 1
    if dst_is_s3:
        existing = sorted([
            d for d in s3_list_immediate_subdirs(s3, dst_bucket, shards_prefix)
            if d.startswith("shard_")
        ])
    else:
        if os.path.isdir(shards_dir):
            existing = sorted([
                d for d in os.listdir(shards_dir)
                if d.startswith("shard_") and os.path.isdir(os.path.join(shards_dir, d))
            ])
        else:
            existing = []

    if existing:
        # Find the highest existing global index
        last_idx = max(int(d.split("_")[1]) for d in existing)
        start_global_idx = last_idx + 1
        print(f"Resuming: {len(existing)} shard(s) already in destination. "
              f"Next global index: {start_global_idx}.")

    # ------------------------------------------------------------------
    # Walk all batches and build the plan
    # ------------------------------------------------------------------
    # plan: list of (batch_name, old_shard_name, src_shard_path_or_prefix, global_idx)
    plan: List[Tuple[str, str, str, int]] = []
    global_idx = start_global_idx

    for batch_name, batch_path_or_prefix in batches:
        if src_is_s3:
            batch_prefix_key = parse_s3_url(batch_path_or_prefix)[1]
            shards_in_batch = list_shard_dirs_in_batch_s3(s3, src_bucket, batch_prefix_key)
        else:
            shards_in_batch = list_shard_dirs_in_batch_local(batch_path_or_prefix)

        for old_shard_name, shard_path_or_prefix in shards_in_batch:
            plan.append((batch_name, old_shard_name, shard_path_or_prefix, global_idx))
            global_idx += 1

    total_shards = len(plan)
    print(f"Total shards to move: {total_shards}")
    if total_shards == 0:
        print("Nothing to flatten (all shards may already be in destination).")
        return

    if args.dry_run:
        print("\n[DRY-RUN MODE] No files will be written.\n")

    # ------------------------------------------------------------------
    # Execute flatten
    # ------------------------------------------------------------------
    # manifest_entries: per-batch shard range info
    manifest_entries: Dict[str, dict] = {}
    processed = 0
    failed = 0

    for batch_name, old_shard_name, src_path, gidx in plan:
        new_shard_name = f"shard_{gidx:03d}"

        if dst_is_s3:
            dst_shard_prefix = _s3_key_join(shards_prefix, new_shard_name)
        else:
            dst_shard_dir = os.path.join(shards_dir, new_shard_name)

        print(f"  [{processed+1}/{total_shards}] {batch_name}/{old_shard_name} -> {new_shard_name}")

        try:
            if src_is_s3 and dst_is_s3:
                src_shard_prefix_key = parse_s3_url(src_path)[1]
                dst_shard_prefix_key = parse_s3_url(dst_shard_prefix)[1] if dst_is_s3 else None
                flatten_shard_s3(
                    s3,
                    src_bucket, src_shard_prefix_key,
                    dst_bucket, _s3_key_join(shards_prefix, new_shard_name),
                    new_shard_name,
                    dry_run=args.dry_run,
                )
            elif not src_is_s3 and not dst_is_s3:
                flatten_shard_local(src_path, dst_shard_dir, new_shard_name, dry_run=args.dry_run)
            else:
                # Mixed local/S3 — not a typical use case, raise clearly
                raise NotImplementedError(
                    "Mixed local-source and S3-destination (or vice versa) is not supported. "
                    "Use either both local or both S3."
                )

            # Track per-batch shard range for manifest
            if batch_name not in manifest_entries:
                manifest_entries[batch_name] = {"shard_start": gidx, "shard_end": gidx}
            else:
                manifest_entries[batch_name]["shard_end"] = gidx

            processed += 1

        except Exception as e:
            print(f"    ERROR: {e}")
            failed += 1
            if args.fail_fast:
                print("Stopping due to --fail-fast.")
                break

    # ------------------------------------------------------------------
    # Write flatten_manifest.json
    # ------------------------------------------------------------------
    manifest = {
        "format": "flatten_manifest_v1",
        "src": args.src_uri or args.src_dir,
        "dst": args.dst_uri or args.dst_dir,
        "total_shards_moved": processed,
        "batches": [
            {
                "batch_name": name,
                "shard_start": info["shard_start"],
                "shard_end": info["shard_end"],
                "num_shards": info["shard_end"] - info["shard_start"] + 1,
            }
            for name, info in manifest_entries.items()
        ],
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    if not args.dry_run:
        if dst_is_s3:
            write_flatten_manifest_s3(s3, dst_bucket, dst_prefix, manifest)
        else:
            write_flatten_manifest_local(dst_dir, manifest)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Flatten complete: {processed}/{total_shards} shards moved.")
    if failed:
        print(f"  WARNING: {failed} shard(s) failed.")
    if args.dry_run:
        print("  (DRY-RUN — no files were written)")
    print("=" * 60)

    if failed and not args.dry_run:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Flatten per-batch staging shards into a global flat shards/ directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src_group = p.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--src-dir",
        default=None,
        help="Local path to the staging directory (output of tokenize_curriculum.py)",
    )
    src_group.add_argument(
        "--src-uri",
        default=None,
        help="S3 URI of the staging directory (s3://bucket/prefix)",
    )

    dst_group = p.add_mutually_exclusive_group(required=True)
    dst_group.add_argument(
        "--dst-dir",
        default=None,
        help="Local path where shards/ output directory will be created",
    )
    dst_group.add_argument(
        "--dst-uri",
        default=None,
        help="S3 URI of the output prefix (s3://bucket/prefix); shards/ is created under it",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the flatten plan without making any changes",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on the first shard error",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Basic validation: src and dst must both be local or both be S3
    src_is_s3 = is_s3_uri(args.src_uri or "")
    dst_is_s3 = is_s3_uri(args.dst_uri or "")
    if src_is_s3 != dst_is_s3:
        print(
            "ERROR: --src-uri/--src-dir and --dst-uri/--dst-dir must both be S3 "
            "or both be local paths.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 60)
    print("Flatten Shards")
    print(f"  Source : {args.src_uri or args.src_dir}")
    print(f"  Dest   : {args.dst_uri or args.dst_dir}")
    if args.dry_run:
        print("  Mode   : DRY-RUN")
    print("=" * 60)

    run_flatten(args)


if __name__ == "__main__":
    main()
