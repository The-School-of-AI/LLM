#!/usr/bin/env python3
"""
Mock Source Parquet Generator — 2-Level Architecture
======================================================

Builds a 2-level mock dataset (T3 + T1) under --output-dir/<profile>/ using
real rows sampled from local T3 and T1 source parquets. Use this to test
tokenize_curriculum.py end-to-end without needing S3 access.

Architecture:
  T3 (coreset index)  →  T1 (raw text)      (T2 is entirely bypassed)

Row content is always taken from real parquets — no synthetic data.
The chunk_id = id invariant is maintained: for each chunk_id in the mock T3,
the matching T1 file contains a row whose id equals that chunk_id.
One extra row per T1 file keeps its original id (decoy) to verify filtering.

Profiles
--------
  minimal  : 1  T3 file ×  1 row  →  1 T1 file ×  (1 row + 1 decoy)
             Verifies zero-crash end-to-end. Produces 0 token blocks with
             --drop-remainder (1 row × ~500 tokens < 4096 block size).

  small    : 10 T3 files × 25 rows → 5 T1 files × (50 rows + 1 decoy)
             ~30 output blocks at block_size=4096.

  parallel : 20 T3 files × 25 rows → 5 T1 files × (100 rows + 1 decoy)
             ~61 output blocks. Run tokenizer with --file-parallelism 3.

Within each T3 batch file, rows are distributed across all T1 files
(round-robin by row index). This means each batch file produces multiple
t1_file_path groups, exercising the groupby logic in process_coreset_file().

Usage
-----
  python scripts/create_mock_sources.py \\
      --profile       minimal \\
      --t3-source-dir dataset/source/t3_coresets \\
      --t1-source-dir dataset/source/t1_rawdata/normalized_data \\
      --output-dir    dataset/final

Output
------
  dataset/final/<profile>/
    t3/
      selected_indices_<profile>_batch000000.parquet   ← real T3 rows + t1_file_path
      ...
    t1/
      source=C4/
        part-00000-8299c866-...-c000.zstd.parquet   ← real T1 rows, id replaced
        ...  (1–5 files depending on profile)

Tokenizer smoke test (example for 'small' profile)
---------------------------------------------------
  python tokenize_curriculum.py \\
      --coreset-uri   dataset/final/small/t3/selected_indices_small_batch000000.parquet \\
      --dst-uri       dataset/final/small/tok_out \\
      --tokenizer-path ./tsai_131k_tokenizer \\
      --t1-base-uri   dataset/final/small/t1 \\
      --block-size    4096 \\
      --shard-size-mb 512 \\
      --num-proc      2 \\
      --drop-remainder \\
      --stage         1 \\
      --tokenizer-version v1 \\
      --tmp-dir       /tmp/tok_tmp
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List

import pandas as pd

# ── Hardcoded column names (fixed by schema) ──────────────────────────────────
T3_CHUNK_ID_COL     = "chunk_id"
T3_T1_FILE_PATH_COL = "t1_file_path"
T1_ID_COL           = "id"
T1_SOURCE_SUBDIR    = "source=C4"

# ── Profile definitions ───────────────────────────────────────────────────────
PROFILES: Dict[str, Dict] = {
    # minimal: 1 T3 row, 1 T1 file.  The T1 matched row is chosen from real T1
    # rows whose text is >= 18,000 chars, guaranteeing >= 4,096 TSAI tokens
    # (measured ratio: ~4.29 chars/token at p10 → 18,000 / 4.29 ≈ 4,196 tokens).
    "minimal":  {"num_batches": 1,  "rows_per_batch": 1,  "num_t1_files": 1, "min_text_chars": 18000},
    "small":    {"num_batches": 10, "rows_per_batch": 25, "num_t1_files": 5},
    # parallel: 100 rows/batch → ~16 blocks/batch; with --shard-size-mb 0.1 (6 blocks/shard)
    # → shard_001 + shard_002 + shard_003 visible per batch.
    "parallel": {"num_batches": 20, "rows_per_batch": 100, "num_t1_files": 5},
}


# ── File discovery ────────────────────────────────────────────────────────────

def list_t3_source_files(t3_source_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(t3_source_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {t3_source_dir}")
    return files


def list_t1_source_files(t1_source_dir: str) -> List[str]:
    pattern = os.path.join(t1_source_dir, T1_SOURCE_SUBDIR, "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No .parquet files found under {t1_source_dir}/{T1_SOURCE_SUBDIR}/"
        )
    return files


# ── Step 1: Sample T3 rows ────────────────────────────────────────────────────

def sample_t3_batches(
    t3_source_files: List[str],
    num_batches: int,
    rows_per_batch: int,
    min_token_count: int = 0,
) -> List[pd.DataFrame]:
    """
    Return num_batches DataFrames, each with rows_per_batch rows sampled from
    real T3 source parquets (cycling through source files when num_batches
    exceeds the number of available files). Random state seeded per batch
    for reproducibility.

    If min_token_count > 0, only rows where token_count >= min_token_count are
    eligible — used by the 'minimal' profile to guarantee each sampled document
    is long enough to fill at least one 4096-token block on its own.
    """
    batches: List[pd.DataFrame] = []
    for i in range(num_batches):
        src_file = t3_source_files[i % len(t3_source_files)]
        df = pd.read_parquet(src_file)
        if min_token_count > 0 and "token_count" in df.columns:
            df = df[df["token_count"] >= min_token_count].reset_index(drop=True)
            if df.empty:
                raise ValueError(
                    f"No rows with token_count >= {min_token_count} in {src_file}"
                )
        n = min(rows_per_batch, len(df))
        if n < rows_per_batch:
            print(f"  [WARN] batch {i}: only {n} rows available (wanted {rows_per_batch})")
        sampled = df.sample(n=n, random_state=i).reset_index(drop=True)
        batches.append(sampled)
    return batches


# ── Step 2: Assign t1_file_path ───────────────────────────────────────────────

def assign_t1_paths(
    batches: List[pd.DataFrame],
    t1_basenames: List[str],
    num_t1_files: int,
) -> List[pd.DataFrame]:
    """
    Add t1_file_path column to each batch DataFrame.

    Row j (0-indexed within each batch) maps to:
        t1_basenames[j % num_t1_files]

    Distributing rows across all T1 files within every batch exercises the
    groupby(t1_file_path) logic in process_coreset_file().
    """
    result = []
    for df in batches:
        df = df.copy()
        df[T3_T1_FILE_PATH_COL] = [
            f"{T1_SOURCE_SUBDIR}/{t1_basenames[j % num_t1_files]}"
            for j in range(len(df))
        ]
        result.append(df)
    return result


# ── Step 3: Write mock T3 ─────────────────────────────────────────────────────

def write_mock_t3(
    batches: List[pd.DataFrame],
    profile: str,
    output_dir: str,
) -> List[str]:
    """Write mock T3 batch files; return list of written paths."""
    out_dir = os.path.join(output_dir, profile, "t3")
    os.makedirs(out_dir, exist_ok=True)

    written = []
    for i, df in enumerate(batches):
        fname = f"selected_indices_{profile}_batch{i:06d}.parquet"
        out_path = os.path.join(out_dir, fname)
        df.to_parquet(out_path, index=False)
        written.append(out_path)
        print(
            f"  T3 batch {i:02d}: {fname}  "
            f"({len(df)} rows, {df[T3_T1_FILE_PATH_COL].nunique()} unique t1_file_path)"
        )
    return written


# ── Step 4: Write mock T1 ─────────────────────────────────────────────────────

def build_t1_chunk_map(
    batches: List[pd.DataFrame],
    t1_basenames: List[str],
    num_t1_files: int,
) -> Dict[str, List[str]]:
    """
    For each T1 filename, collect all chunk_ids that must appear in its id
    column (preserving insertion order; deduplicating to guard against
    repeated samples from the same source file).

    Returns: {t1_basename: [chunk_id, ...]}
    """
    t1_chunk_map: Dict[str, List[str]] = {b: [] for b in t1_basenames[:num_t1_files]}
    seen: Dict[str, set] = {b: set() for b in t1_basenames[:num_t1_files]}

    for df in batches:
        for _, row in df.iterrows():
            t1_basename = os.path.basename(str(row[T3_T1_FILE_PATH_COL]))
            chunk_id = str(row[T3_CHUNK_ID_COL])
            if t1_basename in t1_chunk_map and chunk_id not in seen[t1_basename]:
                t1_chunk_map[t1_basename].append(chunk_id)
                seen[t1_basename].add(chunk_id)

    return t1_chunk_map


def write_mock_t1(
    t1_source_files: List[str],
    t1_chunk_map: Dict[str, List[str]],
    profile: str,
    output_dir: str,
    min_text_chars: int = 0,
) -> None:
    """
    Write mock T1 parquets to output_dir/<profile>/t1/source=C4/.

    For each T1 file:
      - Read the real T1 source parquet.
      - If min_text_chars > 0, filter rows to those where len(text) >= min_text_chars
        and sort by text length descending so the N matched rows are the longest
        available documents.  This guarantees the tokenizer sees >= 4096 tokens
        per matched row (used by the 'minimal' profile).
      - Slice to exactly N + 1 rows (N chunk_ids + 1 decoy).
      - Replace rows 0..N-1 id values with the N chunk_ids from T3
        (all other columns — text, hash, domain, language, metadata, etc.
        are preserved from the real T1 rows unchanged).
      - Row N keeps its original id (not in any T3 chunk_id) — the decoy.
    """
    out_dir = os.path.join(output_dir, profile, "t1", T1_SOURCE_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    src_by_basename = {os.path.basename(f): f for f in t1_source_files}

    for t1_basename, chunk_ids in t1_chunk_map.items():
        src_path = src_by_basename.get(t1_basename)
        if src_path is None:
            print(f"  [WARN] No source file for {t1_basename} — skipping")
            continue

        n = len(chunk_ids)
        print(f"  T1 {t1_basename}: {n} chunk_id(s) to embed + 1 decoy")

        df = pd.read_parquet(src_path)

        if min_text_chars > 0 and "text" in df.columns:
            df = df[df["text"].str.len() >= min_text_chars].copy()
            if len(df) < n + 1:
                raise ValueError(
                    f"{src_path}: only {len(df)} rows have text >= {min_text_chars} chars "
                    f"but {n} chunk_ids + 1 decoy are needed."
                )
            # Sort longest-first so matched rows have the most tokens
            df = df.assign(_tlen=df["text"].str.len()).sort_values(
                "_tlen", ascending=False
            ).drop(columns="_tlen").reset_index(drop=True)
            print(f"    min_text_chars={min_text_chars}: {len(df)} qualifying rows; "
                  f"matched row text length = {df['text'].str.len().iloc[0]:,} chars")

        if n + 1 > len(df):
            raise ValueError(
                f"{src_path} has {len(df)} rows but needs {n} chunk_ids + 1 decoy. "
                "Not enough real T1 rows."
            )

        # Slice to exactly N + 1 rows: rows 0..N-1 get chunk_ids, row N is decoy
        df_slice = df.iloc[: n + 1].copy()
        id_col_idx = df_slice.columns.get_loc(T1_ID_COL)
        for j, chunk_id in enumerate(chunk_ids):
            df_slice.iloc[j, id_col_idx] = chunk_id
        # Row N is untouched — its original id is the decoy

        out_path = os.path.join(out_dir, t1_basename)
        df_slice.to_parquet(out_path, index=False)
        print(f"    Written: {out_path}  ({n + 1} rows: {n} matched + 1 decoy at row {n})")


# ── Summary / next-step hints ─────────────────────────────────────────────────

def print_next_steps(profile: str, output_dir: str, batches: List[pd.DataFrame]) -> None:
    profile_dir = os.path.join(output_dir, profile)
    total_rows  = sum(len(b) for b in batches)
    est_tokens  = total_rows * 500
    est_blocks  = est_tokens // 4096

    print(f"\n{'-' * 62}")
    print(f"Done. Output: {profile_dir}/")
    print(f"  T3 batches    : {len(batches)}")
    print(f"  Total T3 rows : {total_rows}")
    print(f"  Est. tokens   : ~{est_tokens:,}  (~{est_blocks} blocks at 4096 tokens/block)")
    if profile == "minimal":
        print(f"  Note: minimal profile samples long docs (token_count >= 4096); expect 1+ complete blocks")

    first_batch = f"selected_indices_{profile}_batch000000.parquet"
    parallelism_line = "      --file-parallelism 3 \\\n" if profile == "parallel" else ""

    print(f"""
Next - smoke test ({profile}):
  python tokenize_curriculum.py \\
      --coreset-uri   {profile_dir}/t3/{first_batch} \\
      --dst-uri       {profile_dir}/tok_out \\
      --tokenizer-path ./tsai_131k_tokenizer \\
      --t1-base-uri   {profile_dir}/t1 \\
      --block-size    4096 \\
      --shard-size-mb 512 \\
      --num-proc      2 \\
{parallelism_line}      --drop-remainder \\
      --stage         1 \\
      --tokenizer-version v1 \\
      --tmp-dir       /tmp/tok_tmp""")

    if profile in ("small", "parallel"):
        print(f"""
Directory-level run (all {len(batches)} T3 batch files, single tokenizer call):
  python tokenize_curriculum.py \\
      --coreset-uri   {profile_dir}/t3 \\
      --dst-uri       {profile_dir}/tok_out \\
      --tokenizer-path ./tsai_131k_tokenizer \\
      --t1-base-uri   {profile_dir}/t1 \\
      --block-size    4096 \\
      --shard-size-mb 512 \\
      --num-proc      2 \\
{parallelism_line}      --drop-remainder \\
      --stage         1 \\
      --tokenizer-version v1 \\
      --tmp-dir       /tmp/tok_tmp""")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Create mock 2-level dataset (T3 + T1) for local testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--profile", required=True, choices=list(PROFILES),
        help="Test profile: minimal | small | parallel",
    )
    p.add_argument(
        "--t3-source-dir", required=True,
        help="Directory containing real T3 batch .parquet files",
    )
    p.add_argument(
        "--t1-source-dir", required=True,
        help="Directory containing real T1 parquets (source=C4/ sub-dir expected inside)",
    )
    p.add_argument(
        "--output-dir", default="dataset/final",
        help="Output root. <profile>/ sub-dir is created inside. (default: dataset/final)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg              = PROFILES[args.profile]
    num_batches      = cfg["num_batches"]
    rows_per_batch   = cfg["rows_per_batch"]
    num_t1_files     = cfg["num_t1_files"]
    min_text_chars   = cfg.get("min_text_chars", 0)

    print(f"Profile : {args.profile}")
    print(f"  T3    : {num_batches} batch file(s) x {rows_per_batch} rows each")
    print(f"  T1    : {num_t1_files} file(s)"
          + (f", matched rows filtered to text >= {min_text_chars:,} chars" if min_text_chars else ""))

    # ── Discover source files ─────────────────────────────────────────────────
    print(f"\nDiscovering source files...")
    t3_source_files = list_t3_source_files(args.t3_source_dir)
    t1_source_files = list_t1_source_files(args.t1_source_dir)
    print(f"  T3 source: {len(t3_source_files)} file(s) in {args.t3_source_dir}")
    print(f"  T1 source: {len(t1_source_files)} file(s) in "
          f"{args.t1_source_dir}/{T1_SOURCE_SUBDIR}/")

    if len(t1_source_files) < num_t1_files:
        raise ValueError(
            f"Profile '{args.profile}' needs {num_t1_files} T1 source files but only "
            f"{len(t1_source_files)} found under "
            f"{args.t1_source_dir}/{T1_SOURCE_SUBDIR}/"
        )

    t1_basenames = [os.path.basename(f) for f in t1_source_files]

    # ── Step 1: Sample T3 rows ────────────────────────────────────────────────
    print(f"\nStep 1: Sampling T3 rows...")
    batches = sample_t3_batches(t3_source_files, num_batches, rows_per_batch)

    # ── Step 2: Assign t1_file_path ───────────────────────────────────────────
    print(f"\nStep 2: Assigning t1_file_path values (round-robin across {num_t1_files} T1 files)...")
    batches = assign_t1_paths(batches, t1_basenames, num_t1_files)

    # ── Step 3: Write mock T3 ─────────────────────────────────────────────────
    print(f"\nStep 3: Writing mock T3 parquets...")
    write_mock_t3(batches, args.profile, args.output_dir)

    # ── Step 4: Write mock T1 ─────────────────────────────────────────────────
    print(f"\nStep 4: Writing mock T1 parquets (id column replaced with chunk_ids)...")
    t1_chunk_map = build_t1_chunk_map(batches, t1_basenames, num_t1_files)
    write_mock_t1(t1_source_files, t1_chunk_map, args.profile, args.output_dir, min_text_chars)

    # ── Next-step hints ───────────────────────────────────────────────────────
    print_next_steps(args.profile, args.output_dir, batches)


if __name__ == "__main__":
    main()
