"""
Unified file reader — yields plain Python dicts from JSONL, JSON, Parquet, or CSV.
Format is auto-detected from file extension unless overridden.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator


def iter_records(path: str | Path, fmt: str | None = None) -> Iterator[dict]:
    """
    Yield one dict per record from *path*.

    Args:
        path: File path (local or any path supported by the caller's filesystem).
        fmt:  Format override ("jsonl", "json", "parquet", "csv").
              When None, format is inferred from file extension.
    """
    path = Path(path)
    if fmt is None:
        fmt = _detect_format(path)
    fmt = fmt.lower()

    if fmt == "jsonl":
        yield from _read_jsonl(path)
    elif fmt == "json":
        yield from _read_json(path)
    elif fmt == "parquet":
        yield from _read_parquet(path)
    elif fmt == "csv":
        yield from _read_csv(path)
    else:
        raise ValueError(f"Unsupported format '{fmt}' for file: {path}")


def count_records(path: str | Path, fmt: str | None = None) -> int:
    """Count total records in a file (reads fully; use sparingly)."""
    return sum(1 for _ in iter_records(path, fmt))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _detect_format(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    mapping = {
        "jsonl": "jsonl",
        "ndjson": "jsonl",
        "json": "json",
        "parquet": "parquet",
        "csv": "csv",
        "tsv": "csv",
    }
    if ext not in mapping:
        raise ValueError(
            f"Cannot detect format from extension '{ext}'. "
            f"Pass fmt= explicitly (jsonl|json|parquet|csv)."
        )
    return mapping[ext]


def _read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON — {exc}") from exc


def _read_json(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        yield data
    else:
        raise ValueError(f"{path}: expected JSON array or object, got {type(data)}")


def _read_parquet(path: Path) -> Iterator[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required for Parquet support: pip install pyarrow")
    table = pq.read_table(path)
    for batch in table.to_batches():
        rows = batch.to_pydict()
        keys = list(rows.keys())
        n = len(rows[keys[0]]) if keys else 0
        for i in range(n):
            yield {k: rows[k][i] for k in keys}


def _read_csv(path: Path) -> Iterator[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)
