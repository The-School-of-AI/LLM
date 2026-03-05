#!/usr/bin/env python3
"""
Shard Validation Script
=======================

Validates tokenized shard directories against the 8-point checklist defined in
TOKENIZER_TEAM_RECOMMENDATIONS.md §4.

Checks per shard:
  1. Required files present: tokens.bin, tokens.idx, metadata.json
  2. tokenizer_hash matches the current canonical tokenizer
  3. eos_token_id, pad_token_id, vocab_size match live tokenizer
  4. metadata.total_tokens == tokens.bin size / 4
  5. len(idx_offsets) - 1 == metadata.num_blocks
  6. rows_dropped + rows_with_eos == rows_input
  7. max(token_ids) < vocab_size
  8. band, domain, stage are non-empty in metadata.json

Expected layout (output of flatten_shards.py):
  <shards-dir>/
    shards/
      shard_001/ -> tokens.bin, tokens.idx, metadata.json
      shard_002/ -> ...
      ...

Usage:
  python validate_shards.py --shards-dir /tmp/output --tokenizer-path ./tsai_131k_tokenizer
  python validate_shards.py --shards-dir /tmp/output --tokenizer-path ./tsai_131k_tokenizer --verbose
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np


# ---------------------------------------------------------------------------
# Helpers (duplicated from tokenize_curriculum.py to keep this script standalone)
# ---------------------------------------------------------------------------

def is_s3_uri(uri: str) -> bool:
    return uri.startswith("s3://")


def parse_s3_url(url: str) -> Tuple[str, str]:
    parsed = urlparse(url)
    return parsed.netloc, parsed.path.lstrip("/")


def _make_s3():
    import boto3
    return boto3.client("s3")


def key_exists_local(path: str) -> bool:
    return os.path.exists(path)


def read_bytes_local(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def read_json_local(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_shard_dirs_local(base_dir: str) -> List[str]:
    """Return sorted list of shard_NNN subdirectories under <base_dir>/shards/.

    Layout: <base_dir>/shards/shard_NNN/
    """
    shards_dir = os.path.join(base_dir, "shards")
    if not os.path.isdir(shards_dir):
        return []
    return sorted([
        os.path.join(shards_dir, d)
        for d in os.listdir(shards_dir)
        if d.startswith("shard_") and os.path.isdir(os.path.join(shards_dir, d))
    ])


def compute_tokenizer_hash(tokenizer_dir: str) -> str:
    files = sorted(["special_tokens_map.json", "tokenizer.json"])
    h = hashlib.sha256()
    for fname in files:
        fpath = os.path.join(tokenizer_dir, fname)
        with open(fpath, "rb") as f:
            h.update(fname.encode())
            h.update(f.read())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Shard Validator
# ---------------------------------------------------------------------------

SPDL_IDX_HEADER_FMT = "<II"
TOKEN_DTYPE = np.uint32
BYTES_PER_TOKEN = 4


def validate_shard(
    shard_dir: str,
    expected_tokenizer_hash: str,
    live_vocab_size: int,
    live_eos_token_id: int,
    live_pad_token_id: int,
    verbose: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Validate a single shard directory.

    Returns (passed: bool, errors: List[str]).
    """
    errors = []

    # ------------------------------------------------------------------ #
    # Check 1: Required files present
    # ------------------------------------------------------------------ #
    bin_path = os.path.join(shard_dir, "tokens.bin")
    idx_path = os.path.join(shard_dir, "tokens.idx")
    meta_path = os.path.join(shard_dir, "metadata.json")

    for path, label in [(bin_path, "tokens.bin"), (idx_path, "tokens.idx"), (meta_path, "metadata.json")]:
        if not os.path.exists(path):
            errors.append(f"[CHECK 1] Missing file: {label}")

    if errors:
        return False, errors

    # Load metadata
    try:
        meta = read_json_local(meta_path)
    except Exception as e:
        errors.append(f"[CHECK 1] Cannot parse metadata.json: {e}")
        return False, errors

    # ------------------------------------------------------------------ #
    # Check 2: tokenizer_hash matches canonical tokenizer
    # ------------------------------------------------------------------ #
    meta_hash = meta.get("tokenizer_hash", "")
    if not meta_hash:
        errors.append("[CHECK 2] tokenizer_hash is missing from metadata.json")
    elif meta_hash != expected_tokenizer_hash:
        errors.append(
            f"[CHECK 2] tokenizer_hash mismatch: "
            f"metadata={meta_hash[:16]}... expected={expected_tokenizer_hash[:16]}..."
        )

    # ------------------------------------------------------------------ #
    # Check 3: eos_token_id, pad_token_id, vocab_size match live tokenizer
    # ------------------------------------------------------------------ #
    meta_eos = meta.get("eos_token_id")
    meta_pad = meta.get("pad_token_id")
    meta_vocab = meta.get("vocab_size")

    if meta_eos != live_eos_token_id:
        errors.append(f"[CHECK 3] eos_token_id mismatch: metadata={meta_eos} live={live_eos_token_id}")
    if meta_pad != live_pad_token_id:
        errors.append(f"[CHECK 3] pad_token_id mismatch: metadata={meta_pad} live={live_pad_token_id}")
    if meta_vocab != live_vocab_size:
        errors.append(f"[CHECK 3] vocab_size mismatch: metadata={meta_vocab} live={live_vocab_size}")

    # ------------------------------------------------------------------ #
    # Check 4: total_tokens == file_size / 4
    # ------------------------------------------------------------------ #
    bin_size = os.path.getsize(bin_path)
    expected_tokens = bin_size // BYTES_PER_TOKEN
    meta_total_tokens = meta.get("total_tokens", -1)

    if meta_total_tokens != expected_tokens:
        errors.append(
            f"[CHECK 4] total_tokens mismatch: metadata={meta_total_tokens} "
            f"computed_from_file={expected_tokens} (bin_size={bin_size})"
        )

    # ------------------------------------------------------------------ #
    # Check 5: len(idx_offsets) - 1 == num_blocks
    # ------------------------------------------------------------------ #
    try:
        idx_bytes = read_bytes_local(idx_path)
        header_size = struct.calcsize(SPDL_IDX_HEADER_FMT)
        ver, dtype_size = struct.unpack_from(SPDL_IDX_HEADER_FMT, idx_bytes, 0)
        offsets = np.frombuffer(idx_bytes[header_size:], dtype=np.uint64)
        num_blocks_from_idx = len(offsets) - 1
        meta_num_blocks = meta.get("num_blocks", -1)

        if num_blocks_from_idx != meta_num_blocks:
            errors.append(
                f"[CHECK 5] num_blocks mismatch: idx_file={num_blocks_from_idx} "
                f"metadata={meta_num_blocks}"
            )
    except Exception as e:
        errors.append(f"[CHECK 5] Cannot parse tokens.idx: {e}")

    # ------------------------------------------------------------------ #
    # Check 6: rows_dropped + rows_with_eos == rows_input
    # ------------------------------------------------------------------ #
    rows_input = meta.get("rows_input")
    rows_with_eos = meta.get("rows_with_eos")
    rows_dropped = meta.get("rows_dropped")

    if rows_input is None or rows_with_eos is None or rows_dropped is None:
        errors.append("[CHECK 6] Missing row audit fields: rows_input, rows_with_eos, or rows_dropped")
    else:
        if rows_with_eos + rows_dropped != rows_input:
            errors.append(
                f"[CHECK 6] Row count invariant broken: "
                f"rows_with_eos({rows_with_eos}) + rows_dropped({rows_dropped}) "
                f"!= rows_input({rows_input})"
            )

    # ------------------------------------------------------------------ #
    # Check 7: max(token_ids) < vocab_size
    # ------------------------------------------------------------------ #
    try:
        tokens = np.fromfile(bin_path, dtype=TOKEN_DTYPE)
        if len(tokens) == 0:
            errors.append("[CHECK 7] tokens.bin is empty")
        else:
            max_id = int(tokens.max())
            if max_id >= live_vocab_size:
                errors.append(
                    f"[CHECK 7] Token ID out of vocab range: max_id={max_id} vocab_size={live_vocab_size}"
                )
            elif verbose:
                print(f"        max_token_id={max_id}, eos_count={(tokens == live_eos_token_id).sum()}, "
                      f"pad_count={(tokens == live_pad_token_id).sum()}")
    except Exception as e:
        errors.append(f"[CHECK 7] Cannot read tokens.bin: {e}")

    # ------------------------------------------------------------------ #
    # Check 8: band, domain, stage are non-empty
    # ------------------------------------------------------------------ #
    for field in ("band", "domain", "stage"):
        val = meta.get(field)
        if val is None or val == "":
            errors.append(f"[CHECK 8] Required curriculum field '{field}' is missing or empty")

    passed = len(errors) == 0
    return passed, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Validate tokenized shard directories")
    p.add_argument(
        "--shards-dir",
        default=None,
        help="Local output directory produced by flatten_shards.py (must contain a shards/ subdir)",
    )
    p.add_argument(
        "--tokenizer-path",
        required=True,
        help="Path to tokenizer directory (used to compute expected hash)",
    )
    p.add_argument(
        "--verbose", action="store_true", help="Print per-check details even for passing shards"
    )
    p.add_argument(
        "--fail-fast", action="store_true", help="Stop after the first failing shard"
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.shards_dir:
        print("ERROR: --shards-dir is required.", file=sys.stderr)
        sys.exit(1)

    # Load live tokenizer for ground-truth token IDs
    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {args.tokenizer_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    live_vocab_size = len(tokenizer)
    live_eos_token_id = tokenizer.eos_token_id
    live_pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Compute expected hash
    expected_hash = compute_tokenizer_hash(args.tokenizer_path)
    print(f"Expected tokenizer_hash: {expected_hash}")
    print(f"Live vocab_size={live_vocab_size}, eos={live_eos_token_id}, pad={live_pad_token_id}")
    print()

    shard_dirs = list_shard_dirs_local(args.shards_dir)
    if not shard_dirs:
        print(f"No shard directories found under {args.shards_dir}/shards/")
        print("Run flatten_shards.py first to produce the flat shards/ layout.")
        sys.exit(1)

    print(f"Validating {len(shard_dirs)} shard(s)...\n")

    total = 0
    passed = 0
    failed = 0

    for shard_dir in shard_dirs:
        total += 1
        rel = os.path.relpath(shard_dir, args.shards_dir)
        ok, errors = validate_shard(
            shard_dir,
            expected_tokenizer_hash=expected_hash,
            live_vocab_size=live_vocab_size,
            live_eos_token_id=live_eos_token_id,
            live_pad_token_id=live_pad_token_id,
            verbose=args.verbose,
        )

        if ok:
            passed += 1
            if args.verbose:
                print(f"  [PASS] {rel}")
        else:
            failed += 1
            print(f"  [FAIL] {rel}")
            for err in errors:
                print(f"         {err}")
            if args.fail_fast:
                break

    print()
    print("=" * 60)
    print(f"Validation complete: {passed}/{total} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
