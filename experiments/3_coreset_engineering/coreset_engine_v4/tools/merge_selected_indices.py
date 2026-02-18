#!/usr/bin/env python3
"""Merge per-batch/per-shard Parquet part files into a single selected_indices.parquet.

Streaming / clustered runs emit files like:
  output/coresets/<stage>/selected_indices_part_shard###_batch######.parquet

This utility merges those part files into:
  output/coresets/<stage>/selected_indices.parquet

Designed to be low-memory: it streams each part file into a ParquetWriter.

Usage examples:
  python tools/merge_selected_indices.py --coreset-root output/coresets --stage 1B
  python tools/merge_selected_indices.py --coreset-root output/coresets --stages 1B 3B 8B 70B

Notes:
- This does not attempt to merge manifests; shard manifests may exist separately.
- Deduplication is optional and can be expensive at very large scale.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

DEFAULT_COLUMNS: List[str] = [
    "chunk_id",
    "dataset_id",
    "token_count",
    "band",
    "domain",
    "language",
]


@dataclass(frozen=True)
class MergeResult:
    stage: str
    part_files: int
    rows_written: int
    output_path: Path


def _list_part_files(stage_dir: Path, pattern: str) -> List[Path]:
    return sorted(stage_dir.glob(pattern))


def merge_stage_parts(
    stage_dir: Path,
    *,
    output_name: str = "selected_indices.parquet",
    pattern: str = "selected_indices_part_*.parquet",
    columns: Optional[Sequence[str]] = None,
    overwrite: bool = False,
) -> MergeResult:
    """Merge part parquet files under a stage directory into one parquet file."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pyarrow is required to merge parquet part files") from e

    stage_dir = Path(stage_dir)
    if not stage_dir.exists():
        raise FileNotFoundError(f"Stage dir not found: {stage_dir}")

    part_files = _list_part_files(stage_dir, pattern)
    if not part_files:
        raise ValueError(f"No part files matching '{pattern}' under {stage_dir}")

    output_path = stage_dir / output_name
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing {output_path} (use --overwrite)"
        )

    use_columns = list(columns) if columns is not None else None

    first = pq.read_table(part_files[0], columns=use_columns)
    schema = first.schema

    rows_written = 0
    writer = pq.ParquetWriter(str(output_path), schema=schema, compression="snappy")
    try:
        writer.write_table(first)
        rows_written += first.num_rows

        for part in part_files[1:]:
            table = pq.read_table(part, columns=use_columns)
            if table.schema != schema:
                # Best-effort cast to the first file's schema.
                table = table.cast(schema, safe=False)
            writer.write_table(table)
            rows_written += table.num_rows
    finally:
        writer.close()

    return MergeResult(
        stage=stage_dir.name,
        part_files=len(part_files),
        rows_written=rows_written,
        output_path=output_path,
    )


def merge_coreset_root(
    coreset_root: Path,
    stages: Iterable[str],
    *,
    output_name: str,
    pattern: str,
    columns: Optional[Sequence[str]],
    overwrite: bool,
) -> List[MergeResult]:
    results: List[MergeResult] = []
    for stage in stages:
        stage_dir = Path(coreset_root) / stage
        results.append(
            merge_stage_parts(
                stage_dir,
                output_name=output_name,
                pattern=pattern,
                columns=columns,
                overwrite=overwrite,
            )
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge selected_indices Parquet part files per stage"
    )
    parser.add_argument(
        "--coreset-root",
        type=str,
        default="output/coresets",
        help="Root directory containing stage folders (default: output/coresets)",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage", type=str, help="Single stage to merge (e.g., 1B)")
    group.add_argument(
        "--stages", type=str, nargs="+", help="Stages to merge (e.g., 1B 3B 8B 70B)"
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="selected_indices.parquet",
        help="Output parquet filename inside each stage directory",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="selected_indices_part_*.parquet",
        help="Glob pattern for part files",
    )
    parser.add_argument(
        "--columns",
        type=str,
        nargs="+",
        default=None,
        help=f"Optional column subset (default: all columns). Typical: {' '.join(DEFAULT_COLUMNS)}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite selected_indices.parquet if it already exists",
    )

    args = parser.parse_args()
    coreset_root = Path(args.coreset_root)

    stages = [args.stage] if args.stage else list(args.stages)

    results = merge_coreset_root(
        coreset_root,
        stages,
        output_name=args.output_name,
        pattern=args.pattern,
        columns=args.columns,
        overwrite=args.overwrite,
    )

    for r in results:
        print(
            f"{r.stage}: merged {r.part_files} parts -> {r.rows_written} rows at {r.output_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
