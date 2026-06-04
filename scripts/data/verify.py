#!/usr/bin/env python3
"""
Verify shards produced by process.py are compatible with bin_idx_dataloader.

Checks:
  1. Shard structure: tokens.bin + tokens.idx + metadata.json exist
  2. Metadata has required fields (tokenizer_hash, eos_token_id, band, domain)
  3. Tokenizer hash cross-check against loaded tokenizer
  4. .idx offsets are strictly monotonically increasing
  5. .idx offsets align with .bin byte boundaries (each block = BLOCK_SIZE × 4)
  6. Token IDs are in valid vocab range
  7. Blocks are exactly BLOCK_SIZE tokens (via mmap, zero syscalls per block)
  8. No all-zero or all-pad blocks (packing bug detection)
  9. EOS density sanity check (documents were packed with separators)

Parallel verification via ProcessPoolExecutor.

Usage:
    python verify.py --shard-dir /tmp/test_shards --tokenizer-dir Tokenizer/output_hybrid
"""

import argparse
import hashlib
import json
import mmap
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

BLOCK_SIZE = 4096
IDX_HEADER_BYTES = 8
BYTES_PER_TOKEN = 4  # uint32


# ═══════════════════════════════════════════════════════════════════════════
# TOKENIZER HASH — must match bin_idx_dataloader.compute_tokenizer_hash
# ═══════════════════════════════════════════════════════════════════════════


def compute_tokenizer_hash(tokenizer_dir: str) -> str:
    """
    SHA-256 hash of tokenizer.json + special_tokens_map.json.
    Identical to bin_idx_dataloader.compute_tokenizer_hash.
    """
    files = ["tokenizer.json", "special_tokens_map.json"]
    h = hashlib.sha256()
    for fname in sorted(files):
        fpath = os.path.join(tokenizer_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "rb") as f:
            h.update(fname.encode())
            h.update(f.read())
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ShardResult:
    """Result of verifying a single shard."""

    shard_dir: str
    ok: bool = True
    num_blocks: int = 0
    num_tokens: int = 0
    eos_count: int = 0
    max_token_id: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# NATURAL SORT
# ═══════════════════════════════════════════════════════════════════════════


def _natural_key(path: str):
    """Sort key that handles shard_2 before shard_10."""
    return [
        int(c) if c.isdigit() else c.lower()
        for c in re.split(r"(\d+)", os.path.basename(path))
    ]


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE SHARD VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════


def verify_shard(
    shard_dir: str,
    vocab_size: int,
    expected_eos_id: int,
    expected_tok_hash: Optional[str] = None,
) -> ShardResult:
    """
    Verify a single shard directory. Uses mmap for zero-syscall block access.

    Returns ShardResult with errors/warnings populated.
    """
    result = ShardResult(shard_dir=shard_dir)
    shard_name = os.path.basename(shard_dir)

    bin_path = os.path.join(shard_dir, "tokens.bin")
    idx_path = os.path.join(shard_dir, "tokens.idx")
    meta_path = os.path.join(shard_dir, "metadata.json")

    # ── Check files exist ──────────────────────────────────────────────
    for path, name in [
        (bin_path, "tokens.bin"),
        (idx_path, "tokens.idx"),
        (meta_path, "metadata.json"),
    ]:
        if not os.path.exists(path):
            result.errors.append(f"{shard_name}: missing {name}")
            result.ok = False
            return result

    # ── Check metadata ─────────────────────────────────────────────────
    with open(meta_path) as f:
        meta = json.load(f)

    required_fields = [
        "tokenizer_hash",
        "eos_token_id",
        "pad_token_id",
        "band",
        "domain",
        "num_blocks",
        "block_size",
    ]
    for fld in required_fields:
        if fld not in meta:
            result.errors.append(f"{shard_name}: metadata missing field '{fld}'")

    if meta.get("block_size") != BLOCK_SIZE:
        result.errors.append(
            f"{shard_name}: block_size={meta.get('block_size')}, expected {BLOCK_SIZE}"
        )

    if meta.get("eos_token_id") != expected_eos_id:
        result.errors.append(
            f"{shard_name}: eos_token_id={meta.get('eos_token_id')}, expected {expected_eos_id}"
        )

    # Tokenizer hash cross-check
    if expected_tok_hash and meta.get("tokenizer_hash"):
        if meta["tokenizer_hash"] != expected_tok_hash:
            result.errors.append(
                f"{shard_name}: tokenizer_hash mismatch — "
                f"shard={meta['tokenizer_hash'][:16]}... expected={expected_tok_hash[:16]}..."
            )

    # Band vs parent directory name cross-check
    meta_band = meta.get("band", "")
    parent_name = os.path.basename(os.path.dirname(shard_dir))  # e.g. "band_B2"
    if meta_band and parent_name.startswith("band_"):
        expected_band = parent_name.replace("band_", "")
        if meta_band != expected_band:
            result.errors.append(
                f"{shard_name}: metadata band='{meta_band}' but parent directory is '{parent_name}' "
                f"(expected band='{expected_band}')"
            )

    pad_id = meta.get("pad_token_id", 0)

    # ── Check .idx offsets ─────────────────────────────────────────────
    with open(idx_path, "rb") as f:
        f.read(IDX_HEADER_BYTES)
        offsets = np.frombuffer(f.read(), dtype=np.uint64)

    if len(offsets) < 2:
        result.errors.append(
            f"{shard_name}: .idx has {len(offsets)} offsets (need at least 2)"
        )
        result.ok = len(result.errors) == 0
        return result

    num_regions = len(offsets) - 1

    if num_regions != meta.get("num_blocks", -1):
        result.errors.append(
            f"{shard_name}: .idx has {num_regions} regions but metadata says {meta.get('num_blocks')} blocks"
        )

    # Use actual .idx region count as source of truth for stats
    result.num_blocks = num_regions
    result.num_tokens = num_regions * BLOCK_SIZE

    if num_regions == 0:
        result.errors.append(f"{shard_name}: empty shard (0 blocks)")
        result.ok = len(result.errors) == 0
        return result

    # Monotonic check — offsets must be strictly increasing
    diffs = np.diff(offsets)
    if not np.all(diffs > 0):
        bad_indices = np.where(diffs <= 0)[0]
        result.errors.append(
            f"{shard_name}: .idx offsets not monotonically increasing at indices: "
            f"{bad_indices[:5].tolist()}{'...' if len(bad_indices) > 5 else ''}"
        )
        result.ok = False
        return result  # can't trust block boundaries

    # Check all blocks have expected byte size
    expected_bytes_per_block = BLOCK_SIZE * BYTES_PER_TOKEN
    block_sizes = diffs.astype(np.int64)
    bad_size_mask = block_sizes != expected_bytes_per_block
    if np.any(bad_size_mask):
        bad_indices = np.where(bad_size_mask)[0]
        for idx in bad_indices[:5]:
            result.errors.append(
                f"{shard_name}: block {idx} has {block_sizes[idx]} bytes, expected {expected_bytes_per_block}"
            )
        if len(bad_indices) > 5:
            result.errors.append(
                f"  ... and {len(bad_indices) - 5} more bad-size blocks"
            )

    # ── Check token data via mmap ──────────────────────────────────────
    bin_size = os.path.getsize(bin_path)
    expected_bin_size = int(offsets[-1])

    if bin_size != expected_bin_size:
        result.errors.append(
            f"{shard_name}: tokens.bin is {bin_size} bytes but .idx expects {expected_bin_size}"
        )

    if bin_size == 0:
        result.errors.append(f"{shard_name}: tokens.bin is empty")
        result.ok = len(result.errors) == 0
        return result

    eos_count = 0
    max_token_id = 0

    with open(bin_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            # Read ALL tokens at once as a single numpy array
            # .copy() so numpy owns the buffer — otherwise mm.close() raises
            # BufferError because reshape/views keep the mmap referenced
            all_tokens = np.frombuffer(mm, dtype=np.uint32).copy()
            expected_total_tokens = num_regions * BLOCK_SIZE

            if len(all_tokens) != expected_total_tokens:
                result.errors.append(
                    f"{shard_name}: expected {expected_total_tokens} tokens, "
                    f"got {len(all_tokens)}"
                )
                # Still check what we have
                if len(all_tokens) == 0:
                    result.ok = len(result.errors) == 0
                    return result

            # Global token range check
            max_token_id = int(all_tokens.max())
            result.max_token_id = max_token_id

            if max_token_id >= vocab_size:
                result.errors.append(
                    f"{shard_name}: max token ID {max_token_id} >= vocab_size {vocab_size}"
                )

            # Global EOS count
            eos_count = int(np.sum(all_tokens == expected_eos_id))
            result.eos_count = eos_count

            # Per-block checks: all-zero and all-pad detection
            # Reshape into blocks for efficient checking
            num_complete = min(num_regions, len(all_tokens) // BLOCK_SIZE)
            if num_complete > 0:
                blocks_2d = all_tokens[: num_complete * BLOCK_SIZE].reshape(
                    num_complete, BLOCK_SIZE
                )

                # All-zero blocks
                block_sums = blocks_2d.sum(axis=1)
                zero_blocks = int(np.sum(block_sums == 0))
                if zero_blocks > 0:
                    result.errors.append(
                        f"{shard_name}: {zero_blocks} all-zero block(s) detected (packing bug)"
                    )

                # All-pad blocks
                if pad_id > 0:  # skip if pad_id == 0, since that's the same as all-zero
                    pad_counts = np.sum(blocks_2d == pad_id, axis=1)
                    all_pad_blocks = int(np.sum(pad_counts == BLOCK_SIZE))
                    if all_pad_blocks > 0:
                        result.errors.append(
                            f"{shard_name}: {all_pad_blocks} all-pad block(s) detected (packing bug)"
                        )

        finally:
            mm.close()

    # EOS density sanity check
    if eos_count == 0:
        result.errors.append(
            f"{shard_name}: no EOS tokens found — documents may not be properly separated"
        )
    elif num_regions > 0:
        eos_per_block = eos_count / num_regions
        if eos_per_block < 0.1:
            # Warn but don't error — some long-form sources legitimately have sparse EOS
            result.warnings.append(
                f"{shard_name}: very low EOS density ({eos_per_block:.3f} per block) — "
                f"expected at least ~1 document boundary per block"
            )

    result.ok = len(result.errors) == 0
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════


def _resolve_eos_id(tok, tokenizer_dir: str) -> int:
    """
    Resolve EOS token ID from special_tokens_map.json, with fallback to known names.
    Mirrors process.py load_tokenizer() logic exactly.
    """
    stm_path = os.path.join(tokenizer_dir, "special_tokens_map.json")
    eos_id = None

    if os.path.exists(stm_path):
        with open(stm_path) as f:
            stm = json.load(f)
        eos_content = stm.get("eos_token", {})
        if isinstance(eos_content, dict):
            eos_content = eos_content.get("content", "")
        if eos_content:
            eos_id = tok.token_to_id(eos_content)

    if eos_id is None:
        eos_id = tok.token_to_id("<|end_of_text|>")

    return eos_id


def main():
    parser = argparse.ArgumentParser(description="Verify bin_idx shards")
    parser.add_argument(
        "--shard-dir", required=True, help="Output directory from process.py"
    )
    parser.add_argument("--tokenizer-dir", required=True, help="Tokenizer directory")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Parallel verification workers",
    )
    args = parser.parse_args()

    from tokenizers import Tokenizer

    t_start = time.time()

    # Load tokenizer for vocab size and EOS ID
    tok = Tokenizer.from_file(os.path.join(args.tokenizer_dir, "tokenizer.json"))
    vocab_size = tok.get_vocab_size()

    eos_id = _resolve_eos_id(tok, args.tokenizer_dir)
    if eos_id is None:
        print(
            "ERROR: Could not determine EOS token ID from special_tokens_map.json or known names"
        )
        sys.exit(1)

    # Compute tokenizer hash (same algorithm as bin_idx_dataloader)
    tok_hash = compute_tokenizer_hash(args.tokenizer_dir)

    print(f"Verifying shards in: {args.shard_dir}")
    print(f"Tokenizer vocab size: {vocab_size:,}")
    print(f"EOS token ID: {eos_id}")
    print(f"Tokenizer hash: {tok_hash[:16]}...")
    print(f"Workers: {args.workers}")
    print()

    # Find all shard directories
    shard_dirs = []
    for root, dirs, files in os.walk(args.shard_dir):
        if "tokens.bin" in files:
            shard_dirs.append(root)

    if not shard_dirs:
        print("ERROR: No shard directories found (looking for tokens.bin)")
        sys.exit(1)

    shard_dirs.sort(key=_natural_key)
    print(f"Found {len(shard_dirs)} shards to verify\n")

    # Verify shards (parallel or serial)
    total_ok = 0
    total_fail = 0
    all_errors: List[str] = []
    all_warnings: List[str] = []
    total_blocks = 0
    total_tokens = 0
    total_eos = 0
    global_max_token_id = 0

    def _accumulate(result: ShardResult, progress_idx: int):
        nonlocal total_ok, total_fail, total_blocks, total_tokens, total_eos, global_max_token_id
        total_blocks += result.num_blocks
        total_tokens += result.num_tokens
        total_eos += result.eos_count
        global_max_token_id = max(global_max_token_id, result.max_token_id)

        if result.ok:
            total_ok += 1
        else:
            total_fail += 1
            all_errors.extend(result.errors)
            for err in result.errors:
                print(f"  FAIL: {err}")

        all_warnings.extend(result.warnings)

        if progress_idx % max(1, len(shard_dirs) // 10) == 0 or progress_idx == len(
            shard_dirs
        ):
            print(
                f"  [{progress_idx}/{len(shard_dirs)}] "
                f"ok={total_ok} fail={total_fail} blocks={total_blocks:,}",
                flush=True,
            )

    FUTURE_TIMEOUT = 300  # seconds per shard — fail loud if stuck

    if args.workers <= 1 or len(shard_dirs) == 1:
        # Serial mode — simpler for debugging
        for i, sd in enumerate(shard_dirs, 1):
            result = verify_shard(sd, vocab_size, eos_id, tok_hash)
            _accumulate(result, i)
    else:
        # Parallel mode
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(verify_shard, sd, vocab_size, eos_id, tok_hash): sd
                for sd in shard_dirs
            }
            completed = 0
            for future in as_completed(futures):
                completed += 1
                sd = futures[future]
                try:
                    result = future.result(timeout=FUTURE_TIMEOUT)
                except TimeoutError:
                    result = ShardResult(shard_dir=sd, ok=False)
                    result.errors.append(
                        f"{os.path.basename(sd)}: verification timed out after {FUTURE_TIMEOUT}s"
                    )
                except Exception as exc:
                    result = ShardResult(shard_dir=sd, ok=False)
                    result.errors.append(
                        f"{os.path.basename(sd)}: worker exception — {type(exc).__name__}: {exc}"
                    )
                _accumulate(result, completed)

    elapsed = time.time() - t_start

    # Print summary
    print()
    print("=" * 60)
    print(f"VERIFICATION {'PASSED' if total_fail == 0 else 'FAILED'}")
    print("=" * 60)
    print(f"  Shards OK:       {total_ok}")
    print(f"  Shards FAILED:   {total_fail}")
    print(f"  Total blocks:    {total_blocks:,}")
    print(f"  Total tokens:    {total_tokens:,}")
    print(f"  Max token ID:    {global_max_token_id:,} (vocab: {vocab_size:,})")
    print(f"  Total EOS:       {total_eos:,}")
    if total_blocks > 0:
        print(f"  EOS/block:       {total_eos / total_blocks:.2f}")

    if total_tokens > 0:
        size_gb = total_blocks * BLOCK_SIZE * BYTES_PER_TOKEN / (1024**3)
        print(f"  Total size:      {size_gb:.2f} GB (tokens.bin)")

    print(f"  Elapsed:         {elapsed:.1f}s")

    # Warnings
    if all_warnings:
        print(f"\n  {len(all_warnings)} warning(s):")
        for w in all_warnings[:10]:
            print(f"    WARN: {w}")
        if len(all_warnings) > 10:
            print(f"    ... and {len(all_warnings) - 10} more")

    # Errors (print ALL, not truncated — truncation was a P1 bug)
    if total_fail > 0:
        print(f"\n  {len(all_errors)} error(s) found:")
        for err in all_errors:
            print(f"    - {err}")
    else:
        print("\n  All shards are valid and compatible with bin_idx_dataloader.")

    # ── Manifest cross-check ──────────────────────────────────────────
    manifest_path = os.path.join(args.shard_dir, "manifest.json")
    manifest_ok = True
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"\n  Manifest: {manifest_path}")
        print(f"    Pipeline version: {manifest.get('pipeline_version')}")
        print(f"    Tokenizer hash:   {manifest.get('tokenizer_hash', '')[:16]}...")
        print(f"    Total docs:       {manifest.get('total_docs', 0):,}")

        # Cross-check manifest tokenizer hash
        if manifest.get("tokenizer_hash") != tok_hash:
            msg = "manifest tokenizer_hash doesn't match loaded tokenizer!"
            all_warnings.append(msg)
            print(f"    WARNING: {msg}")

        # Token-count cross-check: manifest total vs observed total
        manifest_total_tokens = manifest.get("total_tokens", 0)
        if manifest_total_tokens > 0 and manifest_total_tokens != total_tokens:
            msg = (
                f"manifest total_tokens={manifest_total_tokens:,} vs "
                f"observed total_tokens={total_tokens:,} (delta={abs(manifest_total_tokens - total_tokens):,})"
            )
            all_errors.append(msg)
            manifest_ok = False
            total_fail += 1
            print(f"    ERROR: {msg}")
        elif manifest_total_tokens > 0:
            print(f"    Token count cross-check: OK ({total_tokens:,})")

        # Per-band doc count summary
        if "bands" in manifest:
            print("    Bands:")
            for band, info in sorted(manifest["bands"].items()):
                print(
                    f"      {band}: {info.get('docs', '?'):,} docs, sources={info.get('sources', [])}"
                )
    else:
        all_warnings.append("No manifest.json found — cannot cross-check token counts")
        print(f"\n  WARNING: No manifest.json in {args.shard_dir}")

    # ── Cross-shard duplicate block detection ─────────────────────────
    # Hash first + last block of each shard for fast O(N) duplicate detection.
    # Two shards sharing the same first+last block fingerprint is a strong
    # signal of accidental duplication (e.g. process.py resume bug).
    print("\n  Checking for cross-shard duplicates...", flush=True)
    shard_fingerprints: Dict[str, List[str]] = {}  # fingerprint → [shard_dirs]

    for sd in shard_dirs:
        bin_path = os.path.join(sd, "tokens.bin")
        try:
            bin_size = os.path.getsize(bin_path)
            block_bytes = BLOCK_SIZE * BYTES_PER_TOKEN
            if bin_size < block_bytes:
                continue
            with open(bin_path, "rb") as f:
                first_block = f.read(block_bytes)
                # Seek to last block
                last_offset = max(0, bin_size - block_bytes)
                f.seek(last_offset)
                last_block = f.read(block_bytes)
            fp = hashlib.md5(first_block + last_block).hexdigest()
            shard_fingerprints.setdefault(fp, []).append(os.path.basename(sd))
        except (OSError, IOError):
            continue

    dup_groups = {
        fp: shards for fp, shards in shard_fingerprints.items() if len(shards) > 1
    }
    if dup_groups:
        for fp, shards in dup_groups.items():
            msg = f"Duplicate shard fingerprint: {', '.join(shards)} (md5={fp[:12]}...)"
            all_warnings.append(msg)
            print(f"    WARN: {msg}")
        print(
            f"  Found {len(dup_groups)} duplicate group(s) across {sum(len(s) for s in dup_groups.values())} shards"
        )
    else:
        print("  No cross-shard duplicates detected.")

    # ── Write verify_report.json ──────────────────────────────────────
    report = {
        "status": "PASSED" if total_fail == 0 else "FAILED",
        "shards_ok": total_ok,
        "shards_failed": total_fail,
        "total_blocks": total_blocks,
        "total_tokens": total_tokens,
        "max_token_id": global_max_token_id,
        "vocab_size": vocab_size,
        "eos_id": eos_id,
        "total_eos": total_eos,
        "eos_per_block": round(total_eos / total_blocks, 3) if total_blocks > 0 else 0,
        "size_gb": round(total_blocks * BLOCK_SIZE * BYTES_PER_TOKEN / (1024**3), 3),
        "elapsed_s": round(elapsed, 1),
        "errors": all_errors,  # full list, never truncated
        "warnings": all_warnings,  # full list, never truncated
        "duplicate_groups": {fp: shards for fp, shards in dup_groups.items()},
        "shard_count": len(shard_dirs),
        "tokenizer_hash": tok_hash,
    }
    report_path = os.path.join(args.shard_dir, "verify_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report written to: {report_path}")

    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
