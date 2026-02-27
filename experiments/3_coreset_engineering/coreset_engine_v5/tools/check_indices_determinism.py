#!/usr/bin/env python3
"""Check determinism of coreset selected-indices outputs across multiple reruns.

This tool compares coreset selected-indices outputs across multiple output
folders produced by different runs (e.g., with and without band inference).

The user supplies one or more output directories. For each directory, the tool
tries to locate the coreset root:
  - <output_dir>/coresets
  - <output_dir>/output/coresets
  - <output_dir> (if it looks like a coreset root already)

It then finds index files under each stage directory.

Comparison is done against the first directory as the baseline.

Default comparison (recommended for sharded runs):
    - For each stage, group selections by `band`
    - For each (stage, band), compare the *set* of `chunk_id` across runs
    - Ignores row order and shard file boundaries

Strict comparison (legacy):
    - Same relative file set (no missing/extra files)
    - Same columns (schema) and row counts
    - Same per-column value hashes (streamed, avoids loading entire parquet)

Exit codes:
  0  all compared files match
  2  mismatches found
  1  tool error (e.g., missing paths)

Examples:
  python tools/check_indices_determinism.py \
    --output-dirs output_wo_bandinference output_w_bandinferauto \
    --stages 1B 3B

  # Include a small row-level diff when mismatches occur
  python tools/check_indices_determinism.py \
    --output-dirs output_a output_b \
    --show-first-diff

Notes:
- The default mode checks `chunk_id` presence only; it intentionally ignores
    other columns like token counts, byte lengths, URLs, etc.
- For strict, order-sensitive file-level checks, use `--mode strict`.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

_MISSING = "__MISSING__"


@dataclass(frozen=True)
class FileStats:
    relpath: str
    ext: str
    rows: int
    columns: Tuple[str, ...]
    file_digest: str
    column_digests: Dict[str, str]


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _sha256() -> "hashlib._Hash":
    return hashlib.sha256()


def _digest_hex(h: "hashlib._Hash") -> str:
    return h.hexdigest()


def _looks_like_stage_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    name = p.name
    return name in {"1B", "3B", "8B", "70B", "SFT", "ALIGNMENT"} or name.endswith("B")


def _resolve_coreset_root(output_dir: Path) -> Path:
    candidates = [
        output_dir / "coresets",
        output_dir / "output" / "coresets",
        output_dir,
    ]
    for c in candidates:
        if not c.exists() or not c.is_dir():
            continue
        # Heuristic: it should contain at least one stage directory.
        stage_dirs = [x for x in c.iterdir() if _looks_like_stage_dir(x)]
        if stage_dirs:
            return c
    raise FileNotFoundError(
        f"Could not locate coreset root under: {output_dir} (tried coresets/, output/coresets/, and the dir itself)"
    )


def _iter_stage_dirs(coreset_root: Path, stages: Optional[Sequence[str]]) -> List[Path]:
    if stages:
        return [coreset_root / s for s in stages]

    out: List[Path] = []
    for p in sorted(coreset_root.iterdir(), key=lambda x: x.name):
        if _looks_like_stage_dir(p):
            out.append(p)
    return out


def _collect_index_files(
    coreset_root: Path, stages: Optional[Sequence[str]]
) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for stage_dir in _iter_stage_dirs(coreset_root, stages):
        if not stage_dir.exists() or not stage_dir.is_dir():
            continue

        patterns = [
            "selected_indices_part_*.parquet",
            "selected_indices_part_*.jsonl",
            "selected_indices.parquet",
            "selected_indices.jsonl",
            "selected_indices.csv",
        ]
        for pat in patterns:
            for p in sorted(stage_dir.glob(pat)):
                rel = p.relative_to(coreset_root).as_posix()
                files[rel] = p
    return files


def _pd() -> Any:
    import pandas as pd  # type: ignore

    return pd


def _hash_series_values(series: Any) -> bytes:
    """Return stable bytes representing the values of a pandas Series.

    Uses pandas' hash_pandas_object, which hashes values (not their repr).
    """

    pd = _pd()
    hashed = pd.util.hash_pandas_object(series, index=False)
    # hashed is a Series/Index of uint64; tobytes() is stable.
    return hashed.to_numpy(dtype="uint64", copy=False).tobytes()


def _stats_for_parquet(path: Path, *, batch_rows: int = 65_536) -> FileStats:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pyarrow is required to compare parquet indices") from e

    pf = pq.ParquetFile(str(path))
    schema = pf.schema_arrow
    columns = tuple(schema.names)

    file_hash = _sha256()
    col_hashes: Dict[str, "hashlib._Hash"] = {c: _sha256() for c in columns}

    rows = 0
    for batch in pf.iter_batches(batch_size=batch_rows):
        # Convert to pandas per batch to get stable value hashes.
        df = batch.to_pandas()
        rows += len(df)

        # Include column order in file digest.
        file_hash.update("|".join(columns).encode("utf-8"))
        file_hash.update(b"\n")

        for c in columns:
            # Column-level digest
            b = _hash_series_values(df[c])
            col_hashes[c].update(b)
            # File-level digest
            file_hash.update(c.encode("utf-8"))
            file_hash.update(b)

    return FileStats(
        relpath="",
        ext=".parquet",
        rows=rows,
        columns=columns,
        file_digest=_digest_hex(file_hash),
        column_digests={c: _digest_hex(h) for c, h in col_hashes.items()},
    )


def _iter_jsonl_rows(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _json_canon(v: Any) -> str:
    # Canonical encoding for hashing comparisons.
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stats_for_jsonl(path: Path) -> FileStats:
    # Two-pass: first discover columns and row count.
    columns_set: set[str] = set()
    rows = 0
    for row in _iter_jsonl_rows(path):
        rows += 1
        if isinstance(row, dict):
            columns_set.update(str(k) for k in row.keys())

    columns = tuple(sorted(columns_set))

    file_hash = _sha256()
    col_hashes: Dict[str, "hashlib._Hash"] = {c: _sha256() for c in columns}

    # Second pass: per-column value hashing with missing sentinel.
    for row in _iter_jsonl_rows(path):
        if not isinstance(row, dict):
            # Not expected, but keep deterministic.
            row = {"__row__": row}

        file_hash.update("|".join(columns).encode("utf-8"))
        file_hash.update(b"\n")

        for c in columns:
            v = row.get(c, _MISSING)
            s = _json_canon(v)
            col_hashes[c].update(s.encode("utf-8"))
            file_hash.update(c.encode("utf-8"))
            file_hash.update(s.encode("utf-8"))

    return FileStats(
        relpath="",
        ext=".jsonl",
        rows=rows,
        columns=columns,
        file_digest=_digest_hex(file_hash),
        column_digests={c: _digest_hex(h) for c, h in col_hashes.items()},
    )


def _stats_for_csv(path: Path, *, chunksize: int = 50_000) -> FileStats:
    pd = _pd()

    file_hash = _sha256()
    col_hashes: Dict[str, "hashlib._Hash"] = {}
    columns: Optional[Tuple[str, ...]] = None
    rows = 0

    for chunk in pd.read_csv(path, chunksize=chunksize):
        if columns is None:
            columns = tuple(str(c) for c in chunk.columns.tolist())
            col_hashes = {c: _sha256() for c in columns}

        rows += len(chunk)
        file_hash.update("|".join(columns).encode("utf-8"))
        file_hash.update(b"\n")

        for c in columns:
            b = _hash_series_values(chunk[c])
            col_hashes[c].update(b)
            file_hash.update(c.encode("utf-8"))
            file_hash.update(b)

    if columns is None:
        columns = tuple()

    return FileStats(
        relpath="",
        ext=".csv",
        rows=rows,
        columns=columns,
        file_digest=_digest_hex(file_hash),
        column_digests={c: _digest_hex(h) for c, h in col_hashes.items()},
    )


def _compute_stats(relpath: str, path: Path) -> FileStats:
    ext = path.suffix.lower()
    if ext == ".parquet":
        s = _stats_for_parquet(path)
    elif ext == ".jsonl":
        s = _stats_for_jsonl(path)
    elif ext == ".csv":
        s = _stats_for_csv(path)
    else:
        raise ValueError(f"Unsupported indices file extension: {path}")

    return FileStats(
        relpath=relpath,
        ext=ext,
        rows=s.rows,
        columns=s.columns,
        file_digest=s.file_digest,
        column_digests=s.column_digests,
    )


def _try_show_first_diff(
    baseline_path: Path, other_path: Path, *, max_rows: int = 200_000
) -> Optional[str]:
    """Attempt to find a concrete first differing cell for small-ish files."""

    ext = baseline_path.suffix.lower()
    if ext != other_path.suffix.lower():
        return f"Different extensions: {baseline_path.suffix} vs {other_path.suffix}"

    pd = _pd()

    def _load_df(p: Path) -> Any:
        if ext == ".parquet":
            return pd.read_parquet(p)
        if ext == ".jsonl":
            rows = list(_iter_jsonl_rows(p))
            return pd.DataFrame(rows)
        if ext == ".csv":
            return pd.read_csv(p)
        raise ValueError(ext)

    # Guard memory.
    try:
        df1 = _load_df(baseline_path)
        df2 = _load_df(other_path)
    except Exception as e:
        return f"Could not load for row-level diff: {e}"

    if len(df1) > max_rows or len(df2) > max_rows:
        return f"Row-level diff skipped (rows too large: {len(df1)} vs {len(df2)}; limit={max_rows})"

    cols = sorted(set(df1.columns.tolist()) | set(df2.columns.tolist()))
    df1 = df1.reindex(columns=cols)
    df2 = df2.reindex(columns=cols)

    if len(df1) != len(df2):
        return f"Row count differs: {len(df1)} vs {len(df2)}"

    # Compare row-by-row.
    for c in cols:
        s1 = df1[c]
        s2 = df2[c]
        # Normalize NaN/None.
        a = s1.astype(object).where(~pd.isna(s1), None)
        b = s2.astype(object).where(~pd.isna(s2), None)
        neq = a.ne(b)
        if bool(neq.any()):
            i = int(neq.idxmax())
            return f"First mismatch at row={i}, column='{c}': baseline={a.iloc[i]!r}, other={b.iloc[i]!r}"

    return "No cell-level mismatch found (unexpected)"


def _sample_set_items(values: "set[str]", *, k: int = 10) -> List[str]:
    # Deterministic-ish and efficient for large sets.
    return heapq.nsmallest(k, values)


def _iter_stage_index_paths(stage_dir: Path) -> List[Path]:
    """Return the list of index files to use for a stage.

    If part files exist, use ONLY those (avoid double-counting if a merged file
    also exists). Otherwise fall back to merged selected_indices.*.
    """

    part_paths: List[Path] = []
    for ext in ("parquet", "jsonl"):
        part_paths.extend(sorted(stage_dir.glob(f"selected_indices_part_*.{ext}")))
    if part_paths:
        return part_paths

    merged: List[Path] = []
    for name in (
        "selected_indices.parquet",
        "selected_indices.jsonl",
        "selected_indices.csv",
    ):
        p = stage_dir / name
        if p.exists() and p.is_file():
            merged.append(p)
    return merged


def _iter_chunk_band_pairs_from_parquet(
    path: Path, *, batch_rows: int = 65_536
) -> Iterator[Tuple[str, str]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pyarrow is required to compare parquet indices") from e

    pf = pq.ParquetFile(str(path))
    names = set(pf.schema.names)
    has_band = "band" in names
    cols = ["chunk_id"] + (["band"] if has_band else [])

    for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
        chunk_arr = batch.column(0)
        band_arr = batch.column(1) if has_band else None
        for i in range(batch.num_rows):
            chunk_id = chunk_arr[i].as_py()
            if chunk_id is None:
                continue
            band = band_arr[i].as_py() if band_arr is not None else None
            yield (str(chunk_id), str(band) if band is not None else "__NO_BAND__")


def _iter_chunk_band_pairs_from_jsonl(path: Path) -> Iterator[Tuple[str, str]]:
    for row in _iter_jsonl_rows(path):
        if not isinstance(row, dict):
            continue
        chunk_id = row.get("chunk_id")
        if chunk_id is None:
            continue
        band = row.get("band")
        yield (str(chunk_id), str(band) if band is not None else "__NO_BAND__")


def _iter_chunk_band_pairs_from_csv(
    path: Path, *, chunksize: int = 100_000
) -> Iterator[Tuple[str, str]]:
    pd = _pd()
    # Prefer reading only needed columns.
    usecols = ["chunk_id", "band"]
    try:
        reader = pd.read_csv(path, chunksize=chunksize, usecols=usecols)
        has_band = True
    except Exception:
        reader = pd.read_csv(path, chunksize=chunksize, usecols=["chunk_id"])
        has_band = False

    for chunk in reader:
        if "chunk_id" not in chunk.columns:
            continue
        if has_band and "band" in chunk.columns:
            for chunk_id, band in zip(
                chunk["chunk_id"].astype(object), chunk["band"].astype(object)
            ):
                if chunk_id is None or (
                    isinstance(chunk_id, float) and pd.isna(chunk_id)
                ):
                    continue
                if band is None or (isinstance(band, float) and pd.isna(band)):
                    band = "__NO_BAND__"
                yield (str(chunk_id), str(band))
        else:
            for chunk_id in chunk["chunk_id"].astype(object):
                if chunk_id is None or (
                    isinstance(chunk_id, float) and pd.isna(chunk_id)
                ):
                    continue
                yield (str(chunk_id), "__NO_BAND__")


def _iter_chunk_band_pairs(path: Path) -> Iterator[Tuple[str, str]]:
    ext = path.suffix.lower()
    if ext == ".parquet":
        yield from _iter_chunk_band_pairs_from_parquet(path)
        return
    if ext == ".jsonl":
        yield from _iter_chunk_band_pairs_from_jsonl(path)
        return
    if ext == ".csv":
        yield from _iter_chunk_band_pairs_from_csv(path)
        return
    raise ValueError(f"Unsupported indices file extension: {path}")


def _collect_chunk_ids_by_stage_band(
    coreset_root: Path, stages: Optional[Sequence[str]]
) -> Dict[Tuple[str, str], set[str]]:
    out: Dict[Tuple[str, str], set[str]] = {}
    for stage_dir in _iter_stage_dirs(coreset_root, stages):
        if not stage_dir.exists() or not stage_dir.is_dir():
            continue
        stage = stage_dir.name
        paths = _iter_stage_index_paths(stage_dir)
        for p in paths:
            for chunk_id, band in _iter_chunk_band_pairs(p):
                out.setdefault((stage, band), set()).add(chunk_id)
    return out


def _run_strict_file_comparison(
    out_dirs: List[Path],
    resolved_roots: Dict[Path, Path],
    stages: Optional[Sequence[str]],
    *,
    show_first_diff: bool,
) -> int:
    baseline_dir = out_dirs[0]
    baseline_root = resolved_roots[baseline_dir]

    print("[INFO] Baseline:")
    print(f"  dir  : {baseline_dir}")
    print(f"  root : {baseline_root}")

    file_maps: Dict[Path, Dict[str, Path]] = {}
    for original in out_dirs:
        root = resolved_roots[original]
        m = _collect_index_files(root, stages)
        file_maps[original] = m
        print(f"[INFO] Found {len(m)} indices files under {original}")

    baseline_files = file_maps[baseline_dir]
    baseline_set = set(baseline_files.keys())

    mismatches = 0

    for d in out_dirs[1:]:
        other_set = set(file_maps[d].keys())
        missing = sorted(baseline_set - other_set)
        extra = sorted(other_set - baseline_set)
        if missing:
            mismatches += 1
            print(f"[FAIL] {d}: missing {len(missing)} files (first 10):")
            for p in missing[:10]:
                print(f"  - {p}")
        if extra:
            mismatches += 1
            print(f"[FAIL] {d}: extra {len(extra)} files (first 10):")
            for p in extra[:10]:
                print(f"  + {p}")

    common_relpaths = sorted(
        set.intersection(*[set(file_maps[d].keys()) for d in out_dirs])
        if out_dirs
        else set()
    )

    print(
        f"[INFO] Comparing {len(common_relpaths)} common files across {len(out_dirs)} runs..."
    )

    baseline_stats: Dict[str, FileStats] = {}
    for rel in common_relpaths:
        try:
            baseline_stats[rel] = _compute_stats(rel, baseline_files[rel])
        except Exception as e:
            _eprint(f"ERROR computing baseline stats for {rel}: {e}")
            return 1

    for d in out_dirs[1:]:
        other_files = file_maps[d]
        print(f"\n[INFO] Comparing to: {d}")
        for rel in common_relpaths:
            b_path = baseline_files[rel]
            o_path = other_files[rel]
            b = baseline_stats[rel]
            try:
                o = _compute_stats(rel, o_path)
            except Exception as e:
                mismatches += 1
                print(f"[FAIL] {rel}: could not compute stats for {d}: {e}")
                continue

            if b.ext != o.ext:
                mismatches += 1
                print(f"[FAIL] {rel}: extension differs baseline={b.ext} other={o.ext}")
                continue

            if b.rows != o.rows:
                mismatches += 1
                print(
                    f"[FAIL] {rel}: row count differs baseline={b.rows} other={o.rows}"
                )
                if show_first_diff:
                    print(f"       diff: {_try_show_first_diff(b_path, o_path)}")
                continue

            if b.columns != o.columns:
                mismatches += 1
                b_only = sorted(set(b.columns) - set(o.columns))
                o_only = sorted(set(o.columns) - set(b.columns))
                print(f"[FAIL] {rel}: columns differ")
                if b_only:
                    print(f"       baseline-only (first 20): {b_only[:20]}")
                if o_only:
                    print(f"       other-only (first 20): {o_only[:20]}")
                if show_first_diff:
                    print(f"       diff: {_try_show_first_diff(b_path, o_path)}")
                continue

            if b.file_digest != o.file_digest:
                mismatches += 1
                diff_cols = [
                    c
                    for c in b.columns
                    if b.column_digests.get(c) != o.column_digests.get(c)
                ]
                print(
                    f"[FAIL] {rel}: value digests differ (columns changed: {len(diff_cols)})"
                )
                if diff_cols:
                    print(f"       columns (first 20): {diff_cols[:20]}")
                if show_first_diff:
                    print(f"       diff: {_try_show_first_diff(b_path, o_path)}")
                continue

        if mismatches == 0:
            print(f"[OK] {d}: all common files match baseline")

    if mismatches == 0:
        print("\n[OK] Determinism check passed: all compared indices match.")
        return 0

    print(f"\n[FAIL] Determinism check failed: {mismatches} mismatches found.")
    return 2


def _run_chunk_id_by_stage_band_comparison(
    out_dirs: List[Path],
    resolved_roots: Dict[Path, Path],
    stages: Optional[Sequence[str]],
) -> int:
    baseline_dir = out_dirs[0]
    baseline_root = resolved_roots[baseline_dir]

    print("[INFO] Baseline:")
    print(f"  dir  : {baseline_dir}")
    print(f"  root : {baseline_root}")
    print("[INFO] Mode: chunk_id-by-stage-band (order-agnostic)")

    try:
        baseline_map = _collect_chunk_ids_by_stage_band(baseline_root, stages)
    except Exception as e:
        _eprint(f"ERROR: Failed to collect baseline chunk_ids: {e}")
        return 1

    mismatches = 0

    for d in out_dirs[1:]:
        other_root = resolved_roots[d]
        print(f"\n[INFO] Comparing to: {d}")
        try:
            other_map = _collect_chunk_ids_by_stage_band(other_root, stages)
        except Exception as e:
            mismatches += 1
            print(f"[FAIL] {d}: could not collect chunk_ids: {e}")
            continue

        base_keys = set(baseline_map.keys())
        other_keys = set(other_map.keys())
        missing_keys = base_keys - other_keys
        extra_keys = other_keys - base_keys
        if missing_keys:
            mismatches += 1
            print(
                f"[FAIL] Missing (stage, band) groups: {len(missing_keys)} (first 10):"
            )
            for k in sorted(list(missing_keys))[:10]:
                print(f"  - {k[0]}/{k[1]}")
        if extra_keys:
            mismatches += 1
            print(f"[FAIL] Extra (stage, band) groups: {len(extra_keys)} (first 10):")
            for k in sorted(list(extra_keys))[:10]:
                print(f"  + {k[0]}/{k[1]}")

        common_keys = base_keys & other_keys
        for stage, band in sorted(list(common_keys)):
            bset = baseline_map[(stage, band)]
            oset = other_map[(stage, band)]
            if bset == oset:
                continue

            missing = bset - oset
            extra = oset - bset
            mismatches += 1
            print(
                f"[FAIL] {stage}/{band}: chunk_id set differs (baseline={len(bset)} other={len(oset)} missing={len(missing)} extra={len(extra)})"
            )
            if missing:
                print(
                    f"       missing chunk_id (first 10): {_sample_set_items(missing)}"
                )
            if extra:
                print(f"       extra   chunk_id (first 10): {_sample_set_items(extra)}")

    if mismatches == 0:
        print(
            "\n[OK] chunk_id-by-stage-band check passed: all compared stages/bands match."
        )
        return 0

    print(
        f"\n[FAIL] chunk_id-by-stage-band check failed: {mismatches} mismatches found."
    )
    return 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Determinism check for coreset selected-indices outputs (default: chunk_id presence by stage+band)."
    )
    parser.add_argument(
        "--output-dirs",
        nargs="+",
        required=True,
        help="List of output directories from multiple reruns (at least 2).",
    )
    parser.add_argument(
        "--stages",
        nargs="*",
        default=None,
        help="Optional subset of stages to compare (e.g., 1B 3B). Default: all stages found.",
    )
    parser.add_argument(
        "--mode",
        choices=["chunk_id_by_stage_band", "strict"],
        default="chunk_id_by_stage_band",
        help=(
            "Comparison mode. 'chunk_id_by_stage_band' compares chunk_id sets grouped by (stage, band) "
            "and ignores row order/sharding. 'strict' does file-by-file, order-sensitive comparisons."
        ),
    )
    parser.add_argument(
        "--show-first-diff",
        action="store_true",
        help="(strict mode only) If mismatches are found, try to print a first differing cell for small files.",
    )
    args = parser.parse_args(argv)

    out_dirs = [Path(p) for p in args.output_dirs]
    if len(out_dirs) < 2:
        _eprint("ERROR: Provide at least two --output-dirs.")
        return 1

    try:
        resolved_roots = {d: _resolve_coreset_root(d) for d in out_dirs}
    except Exception as e:
        _eprint(f"ERROR: {e}")
        return 1

    if args.mode == "strict":
        return _run_strict_file_comparison(
            out_dirs,
            resolved_roots,
            args.stages,
            show_first_diff=args.show_first_diff,
        )

    # Default: chunk_id presence by (stage, band)
    if args.show_first_diff:
        _eprint("[WARN] --show-first-diff is ignored unless --mode strict")

    return _run_chunk_id_by_stage_band_comparison(out_dirs, resolved_roots, args.stages)


if __name__ == "__main__":
    raise SystemExit(main())
