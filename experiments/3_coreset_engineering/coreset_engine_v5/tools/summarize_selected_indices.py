#!/usr/bin/env python3
"""Summarize selected index outputs (JSONL or Parquet).

Computes:
- Duplicate chunk identifiers (by chunk_id / uid / guid / id)
- Percentage breakdown by band / language / domain (by row count)

Designed for large streaming outputs:
- Duplicate detection supports an on-disk SQLite mode to avoid high RAM usage.

Examples:
  # JSONL output (e.g., post-export)
  python tools/summarize_selected_indices.py \
    --input-path output/coresets/1B/selected_indices.jsonl \
    --input-format jsonl

  # Parquet output (typical merged streaming output)
  python tools/summarize_selected_indices.py \
    --input-path output/coresets/1B/selected_indices.parquet \
    --input-format parquet

  # Large files: use sqlite duplicate mode (default)
  python tools/summarize_selected_indices.py \
    --input-path output/coresets/1B/selected_indices.parquet \
    --input-format parquet \
    --duplicate-mode sqlite

Exit code:
- 0 on success
- 2 if no files found / invalid args
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Counter, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Summary:
    total_rows: int
    bad_rows: int
    missing_id_rows: int
    duplicates_total_extra_rows: int
    duplicated_ids_count: int
    top_duplicates: List[Tuple[str, int]]
    counts: Dict[str, Counter[str]]


def _iter_files(root: Path, suffix: str) -> Iterator[Path]:
    root = Path(root)
    if root.is_file():
        if root.suffix.lower() == f".{suffix}":
            yield root
        return
    if root.is_dir():
        yield from sorted(root.rglob(f"*.{suffix}"))


def _open_text_maybe_gzip(path: Path):
    if path.suffix.lower().endswith(".gz") or str(path).lower().endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _first_non_empty_str(row: dict, keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            v = value.strip()
            if v:
                return v
            continue
        try:
            v = str(value).strip()
        except Exception:
            continue
        if v:
            return v
    return None


def _pct(n: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return (n / denom) * 100.0


class _DuplicateCounter:
    def add(self, chunk_id: Optional[str]) -> None:  # pragma: no cover
        raise NotImplementedError

    def finalize(
        self, top_n: int
    ) -> Tuple[int, int, List[Tuple[str, int]]]:  # pragma: no cover
        """Returns (duplicates_total_extra_rows, duplicated_ids_count, top_duplicates)."""
        raise NotImplementedError

    def close(self) -> None:
        return


class _MemoryDuplicateCounter(_DuplicateCounter):
    def __init__(self) -> None:
        self._counts: Counter[str] = collections.Counter()

    def add(self, chunk_id: Optional[str]) -> None:
        if chunk_id is None:
            return
        self._counts[chunk_id] += 1

    def finalize(self, top_n: int) -> Tuple[int, int, List[Tuple[str, int]]]:
        dups = [(k, v) for k, v in self._counts.items() if v > 1]
        dups.sort(key=lambda x: (-x[1], x[0]))
        duplicates_total_extra_rows = sum(v - 1 for _, v in dups)
        return duplicates_total_extra_rows, len(dups), dups[:top_n]


class _SqliteDuplicateCounter(_DuplicateCounter):
    def __init__(
        self,
        sqlite_path: Optional[Path],
        table_name: str = "counts",
    ) -> None:
        self._tmp_path: Optional[Path] = None
        if sqlite_path is None:
            fd, name = tempfile.mkstemp(prefix="chunk_id_counts_", suffix=".sqlite")
            os.close(fd)
            sqlite_path = Path(name)
            self._tmp_path = sqlite_path

        self._path = Path(sqlite_path)
        self._table = table_name
        self._con = sqlite3.connect(str(self._path))
        self._con.execute("PRAGMA journal_mode=WAL;")
        self._con.execute("PRAGMA synchronous=NORMAL;")
        self._con.execute(
            f"CREATE TABLE IF NOT EXISTS {self._table} (chunk_id TEXT PRIMARY KEY, c INTEGER NOT NULL)"
        )
        self._con.commit()

        self._pending: List[Tuple[str, int]] = []
        self._pending_limit = 10_000

    def add(self, chunk_id: Optional[str]) -> None:
        if chunk_id is None:
            return
        self._pending.append((chunk_id, 1))
        if len(self._pending) >= self._pending_limit:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        # Upsert counts.
        self._con.executemany(
            f"INSERT INTO {self._table} (chunk_id, c) VALUES (?, ?) "
            f"ON CONFLICT(chunk_id) DO UPDATE SET c = c + excluded.c",
            self._pending,
        )
        self._con.commit()
        self._pending.clear()

    def finalize(self, top_n: int) -> Tuple[int, int, List[Tuple[str, int]]]:
        self._flush()
        duplicated_ids_count = int(
            self._con.execute(
                f"SELECT COUNT(*) FROM {self._table} WHERE c > 1"
            ).fetchone()[0]
        )
        duplicates_total_extra_rows = int(
            self._con.execute(
                f"SELECT COALESCE(SUM(c - 1), 0) FROM {self._table} WHERE c > 1"
            ).fetchone()[0]
        )
        rows = self._con.execute(
            f"SELECT chunk_id, c FROM {self._table} WHERE c > 1 ORDER BY c DESC, chunk_id ASC LIMIT ?",
            (int(top_n),),
        ).fetchall()
        top_duplicates = [(str(k), int(v)) for (k, v) in rows]
        return duplicates_total_extra_rows, duplicated_ids_count, top_duplicates

    def close(self) -> None:
        try:
            self._flush()
        except Exception:
            pass
        try:
            self._con.close()
        except Exception:
            pass
        if self._tmp_path is not None:
            try:
                self._tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def _make_dup_counter(mode: str, sqlite_path: Optional[Path]) -> _DuplicateCounter:
    mode = mode.lower().strip()
    if mode == "none":
        return _DuplicateCounter()  # type: ignore[return-value]
    if mode == "memory":
        return _MemoryDuplicateCounter()
    if mode == "sqlite":
        return _SqliteDuplicateCounter(sqlite_path=sqlite_path)
    raise ValueError(f"Unknown duplicate mode: {mode}")


def _iter_rows_jsonl(files: Iterable[Path]) -> Iterator[dict]:
    for p in files:
        with _open_text_maybe_gzip(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield {"__raw__": line, "__path__": str(p)}


def _summarize_jsonl(
    files: List[Path],
    id_fields: Sequence[str],
    group_fields: Sequence[str],
    duplicate_mode: str,
    sqlite_path: Optional[Path],
    top_duplicates: int,
) -> Summary:
    total_rows = 0
    bad_rows = 0
    missing_id_rows = 0

    counts: Dict[str, Counter[str]] = {f: collections.Counter() for f in group_fields}

    dup_counter: Optional[_DuplicateCounter] = None
    if duplicate_mode != "none":
        dup_counter = _make_dup_counter(duplicate_mode, sqlite_path)

    try:
        for p in files:
            with _open_text_maybe_gzip(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total_rows += 1
                    try:
                        row = json.loads(line)
                    except Exception:
                        bad_rows += 1
                        continue

                    chunk_id = _first_non_empty_str(row, id_fields)
                    if chunk_id is None:
                        missing_id_rows += 1

                    if dup_counter is not None:
                        dup_counter.add(chunk_id)

                    for gf in group_fields:
                        v = row.get(gf)
                        if v is None:
                            counts[gf]["<missing>"] += 1
                        else:
                            counts[gf][str(v)] += 1

        duplicates_total_extra_rows = 0
        duplicated_ids_count = 0
        top_dups: List[Tuple[str, int]] = []
        if dup_counter is not None:
            duplicates_total_extra_rows, duplicated_ids_count, top_dups = (
                dup_counter.finalize(top_duplicates)
            )

        return Summary(
            total_rows=total_rows,
            bad_rows=bad_rows,
            missing_id_rows=missing_id_rows,
            duplicates_total_extra_rows=duplicates_total_extra_rows,
            duplicated_ids_count=duplicated_ids_count,
            top_duplicates=top_dups,
            counts=counts,
        )
    finally:
        if dup_counter is not None:
            dup_counter.close()


def _summarize_parquet(
    files: List[Path],
    id_fields: Sequence[str],
    group_fields: Sequence[str],
    duplicate_mode: str,
    sqlite_path: Optional[Path],
    top_duplicates: int,
) -> Summary:
    try:
        import pyarrow.parquet as pq
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required for parquet summarization (pip install pyarrow)"
        ) from e

    total_rows = 0
    bad_rows = 0
    missing_id_rows = 0

    counts: Dict[str, Counter[str]] = {f: collections.Counter() for f in group_fields}

    dup_counter: Optional[_DuplicateCounter] = None
    if duplicate_mode != "none":
        dup_counter = _make_dup_counter(duplicate_mode, sqlite_path)

    # Read in row groups to keep memory bounded.
    try:
        for p in files:
            try:
                pf = pq.ParquetFile(str(p))
            except Exception:
                bad_rows += 1
                continue

            for rg in range(pf.num_row_groups):
                cols = list(dict.fromkeys([*id_fields, *group_fields]))
                try:
                    table = pf.read_row_group(rg, columns=cols)
                except Exception:
                    bad_rows += int(
                        pf.metadata.row_group(rg).num_rows if pf.metadata else 0
                    )
                    continue

                num = int(table.num_rows)
                total_rows += num

                # Build id column (first available non-empty).
                # For Parquet, id fields might not exist; handle by checking schema.
                schema_names = set(table.schema.names)
                id_arrays = [table.column(f) for f in id_fields if f in schema_names]
                if id_arrays:
                    # Convert each to python strings (smallish per row group).
                    # If multiple id fields exist, we pick first non-empty per row.
                    id_cols = [arr.to_pylist() for arr in id_arrays]
                    for i in range(num):
                        cid: Optional[str] = None
                        for col in id_cols:
                            v = col[i]
                            if v is None:
                                continue
                            s = str(v).strip()
                            if s:
                                cid = s
                                break
                        if cid is None:
                            missing_id_rows += 1
                        if dup_counter is not None:
                            dup_counter.add(cid)
                else:
                    missing_id_rows += num
                    # No ids -> can't meaningfully check duplicates.

                for gf in group_fields:
                    if gf not in schema_names:
                        counts[gf]["<missing>"] += num
                        continue
                    values = table.column(gf).to_pylist()
                    for v in values:
                        if v is None:
                            counts[gf]["<missing>"] += 1
                        else:
                            counts[gf][str(v)] += 1

        duplicates_total_extra_rows = 0
        duplicated_ids_count = 0
        top_dups: List[Tuple[str, int]] = []
        if dup_counter is not None:
            duplicates_total_extra_rows, duplicated_ids_count, top_dups = (
                dup_counter.finalize(top_duplicates)
            )

        return Summary(
            total_rows=total_rows,
            bad_rows=bad_rows,
            missing_id_rows=missing_id_rows,
            duplicates_total_extra_rows=duplicates_total_extra_rows,
            duplicated_ids_count=duplicated_ids_count,
            top_duplicates=top_dups,
            counts=counts,
        )
    finally:
        if dup_counter is not None:
            dup_counter.close()


def _print_summary(summary: Summary, group_fields: Sequence[str]) -> None:
    print(
        f"total_rows={summary.total_rows:,} bad_rows={summary.bad_rows:,} missing_id_rows={summary.missing_id_rows:,}"
    )
    print(
        "duplicates: "
        f"duplicated_ids={summary.duplicated_ids_count:,} "
        f"extra_rows_due_to_dupes={summary.duplicates_total_extra_rows:,}"
    )

    if summary.top_duplicates:
        print("top_duplicates:")
        for cid, c in summary.top_duplicates:
            print(f"  {cid}: {c}")

    for gf in group_fields:
        c = summary.counts.get(gf, collections.Counter())
        print("")
        print(f"{gf} (% of chunks)")
        for k, v in c.most_common():
            print(f"  {k}: {v:,} ({_pct(v, summary.total_rows):.2f}%)")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize selected_indices outputs (JSONL/Parquet)"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Input file or directory containing selected_indices.{jsonl|parquet}",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        required=True,
        choices=["jsonl", "parquet"],
        help="Input format",
    )
    parser.add_argument(
        "--id-fields",
        type=str,
        default="chunk_id,uid,guid,id",
        help="Comma-separated list of identifier fields to use (first non-empty wins)",
    )
    parser.add_argument(
        "--group-fields",
        type=str,
        default="band,language,domain",
        help="Comma-separated fields to compute % breakdowns for",
    )
    parser.add_argument(
        "--duplicate-mode",
        type=str,
        default="sqlite",
        choices=["sqlite", "memory", "none"],
        help="How to detect duplicate ids: sqlite (disk), memory (RAM), or none",
    )
    parser.add_argument(
        "--sqlite-path",
        type=str,
        default=None,
        help="Optional sqlite db path (only used when --duplicate-mode sqlite). Defaults to a temp file.",
    )
    parser.add_argument(
        "--top-duplicates",
        type=int,
        default=20,
        help="How many duplicate ids to print",
    )

    args = parser.parse_args(argv)

    root = Path(args.input_path)
    id_fields = [s.strip() for s in args.id_fields.split(",") if s.strip()]
    group_fields = [s.strip() for s in args.group_fields.split(",") if s.strip()]
    if not id_fields:
        raise SystemExit(2)
    if not group_fields:
        raise SystemExit(2)

    sqlite_path = Path(args.sqlite_path) if args.sqlite_path else None

    if args.input_format == "jsonl":
        # Allow .jsonl.gz too when a file is passed.
        files: List[Path]
        if root.is_file():
            files = [root]
        else:
            files = list(_iter_files(root, "jsonl"))
            # Also include jsonl.gz
            files += sorted(root.rglob("*.jsonl.gz")) if root.is_dir() else []
        if not files:
            raise SystemExit(2)
        summary = _summarize_jsonl(
            files=files,
            id_fields=id_fields,
            group_fields=group_fields,
            duplicate_mode=args.duplicate_mode,
            sqlite_path=sqlite_path,
            top_duplicates=int(args.top_duplicates),
        )
        _print_summary(summary, group_fields)
        return 0

    if args.input_format == "parquet":
        files = list(_iter_files(root, "parquet"))
        if not files:
            raise SystemExit(2)
        summary = _summarize_parquet(
            files=files,
            id_fields=id_fields,
            group_fields=group_fields,
            duplicate_mode=args.duplicate_mode,
            sqlite_path=sqlite_path,
            top_duplicates=int(args.top_duplicates),
        )
        _print_summary(summary, group_fields)
        return 0

    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
