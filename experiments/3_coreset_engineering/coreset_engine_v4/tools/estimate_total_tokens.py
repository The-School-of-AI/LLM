#!/usr/bin/env python3
"""Estimate total tokens in an input dataset for streaming runs.

This is intended to provide a practical value for:
  coreset_builder.py --total-input-tokens-estimate <N>

Supported inputs:
- JSONL (file or directory): sums row["token_count"] (fallback: len(token_ids) if present)
- Parquet (file or directory): sums token_count column via row groups (low memory)

Examples:
  python tools/estimate_total_tokens.py --input-path data/datasets/large_sample_chunks.jsonl --input-format jsonl
  python tools/estimate_total_tokens.py --input-path data/datasets --input-format jsonl
  python tools/estimate_total_tokens.py --input-path data/datasets/sample.parquet --input-format parquet

Exit code:
- 0 on success
- 2 if no files found
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple


def _iter_files(root: Path, suffix: str) -> Iterator[Path]:
    root = Path(root)
    if root.is_file():
        if root.suffix.lower() == f".{suffix}":
            yield root
        return
    if root.is_dir():
        yield from sorted(root.rglob(f"*.{suffix}"))


def _estimate_jsonl(paths: Iterable[Path]) -> Tuple[int, int, int]:
    total_tokens = 0
    total_rows = 0
    bad_rows = 0

    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    bad_rows += 1
                    continue

                token_count = row.get("token_count")
                if token_count is None:
                    token_ids = row.get("token_ids")
                    if isinstance(token_ids, list):
                        token_count = len(token_ids)
                try:
                    token_count_int = int(token_count or 0)
                except Exception:
                    bad_rows += 1
                    continue

                total_rows += 1
                total_tokens += max(0, token_count_int)

    return total_tokens, total_rows, bad_rows


def _estimate_parquet(paths: Iterable[Path]) -> Tuple[int, int, int]:
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pyarrow is required for parquet estimation (pip install pyarrow)") from e

    total_tokens = 0
    total_rows = 0
    bad_files = 0

    for p in paths:
        try:
            pf = pq.ParquetFile(str(p))
        except Exception:
            bad_files += 1
            continue

        md = pf.metadata
        if md is not None:
            total_rows += int(md.num_rows or 0)

        for rg in range(pf.num_row_groups):
            try:
                table = pf.read_row_group(rg, columns=["token_count"])
                col = table.column(0)
                # Sum can return null if empty.
                s = pc.sum(col).as_py()
                total_tokens += int(s or 0)
            except Exception:
                # Fallback: try token_ids length if present.
                try:
                    table = pf.read_row_group(rg, columns=["token_ids"])
                    arr = table.column(0)
                    # token_ids is list<...>; sum(list_value_length)
                    lengths = pc.list_value_length(arr)
                    s = pc.sum(lengths).as_py()
                    total_tokens += int(s or 0)
                except Exception:
                    bad_files += 1
                    break

    return total_tokens, total_rows, bad_files


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate total tokens for streaming input datasets")
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Input dataset path (file or directory)",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        required=True,
        choices=["jsonl", "parquet"],
        help="Input dataset format",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the total token estimate",
    )

    args = parser.parse_args(argv)
    root = Path(args.input_path)

    if args.input_format == "jsonl":
        files = list(_iter_files(root, "jsonl"))
        if not files:
            raise SystemExit(2)
        total_tokens, total_rows, bad = _estimate_jsonl(files)
        if args.quiet:
            print(int(total_tokens))
        else:
            print(f"files={len(files)} rows={total_rows:,} total_tokens={total_tokens:,} bad_rows={bad:,}")
        return 0

    if args.input_format == "parquet":
        files = list(_iter_files(root, "parquet"))
        if not files:
            raise SystemExit(2)
        total_tokens, total_rows, bad_files = _estimate_parquet(files)
        if args.quiet:
            print(int(total_tokens))
        else:
            print(f"files={len(files)} rows={total_rows:,} total_tokens={total_tokens:,} bad_files={bad_files:,}")
        return 0

    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
