from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional


@dataclass(frozen=True)
class ScanLimits:
    max_files: Optional[int]
    max_lines_per_file: Optional[int]


def _iter_jsonl_rows(path: Path, *, max_lines: Optional[int]) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                return
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _band_from_row(row: dict, *, fallback: str) -> str:
    band = row.get("band") or row.get("difficulty_band") or row.get("band_name")
    if band is None:
        return fallback
    b = str(band).strip()
    if not b:
        return fallback
    return b.upper()


def _token_count_from_row(row: dict) -> int:
    tok = row.get("token_count")
    if tok is None:
        tok = row.get("token_count_estimate")
    if tok is None:
        tok = row.get("tokens")
    try:
        return int(tok or 0)
    except Exception:
        return 0


def _domain_from_row(row: dict) -> str:
    dom = row.get("domain")
    if dom is None:
        dom = row.get("domain_id")
    return str(dom or "")


def _language_from_row(row: dict) -> str:
    lang = row.get("language")
    if lang is None:
        lang = row.get("lang")
    return str(lang or "")


def _glob_sorted(base: Path, pattern: str, *, max_files: Optional[int]) -> list[Path]:
    paths = sorted(base.glob(pattern))
    if max_files is not None:
        paths = paths[: max(0, int(max_files))]
    return paths


def scan_outputv2(base: Path, *, bands: Iterable[str], limits: ScanLimits) -> None:
    tokens_by_band = Counter()
    tokens_by_domain = Counter()
    tokens_by_lang = Counter()
    tokens_by_band_domain = Counter()
    rows_by_domain = Counter()

    files_scanned = 0
    rows_scanned = 0

    for band_prefix in bands:
        shard_paths = _glob_sorted(
            base, f"{band_prefix}_shard_*.jsonl", max_files=limits.max_files
        )
        for p in shard_paths:
            files_scanned += 1
            fallback = band_prefix.upper().replace("B", "B")
            for row in _iter_jsonl_rows(p, max_lines=limits.max_lines_per_file):
                rows_scanned += 1
                band = _band_from_row(row, fallback=fallback)
                dom = _domain_from_row(row)
                lang = _language_from_row(row)
                tok = _token_count_from_row(row)

                tokens_by_band[band] += tok
                tokens_by_domain[dom] += tok
                tokens_by_lang[lang] += tok
                tokens_by_band_domain[(band, dom)] += tok
                rows_by_domain[dom] += 1

    print("=== outputv2 scan (jsonl) ===")
    print(f"Base: {base}")
    print(f"Files scanned: {files_scanned:,}")
    print(f"Rows scanned:  {rows_scanned:,}")

    total_tokens = sum(tokens_by_band.values())
    print(f"Total tokens (scanned rows only): {total_tokens:,}")

    print("\nTop bands by tokens:")
    for band, tok in tokens_by_band.most_common(10):
        print(f"  {band}: {tok:,}")

    print("\nTop domains by tokens:")
    for dom, tok in tokens_by_domain.most_common(20):
        dom_disp = dom if dom else "<EMPTY>"
        print(f"  {dom_disp}: {tok:,}")

    print("\nTop languages by tokens:")
    for lang, tok in tokens_by_lang.most_common(15):
        lang_disp = lang if lang else "<EMPTY>"
        print(f"  {lang_disp}: {tok:,}")

    # B0 domain breakdown is usually the first thing that exposes curriculum/domain mismatches.
    b0_domains = [
        (dom, tok) for (band, dom), tok in tokens_by_band_domain.items() if band == "B0"
    ]
    b0_domains.sort(key=lambda x: x[1], reverse=True)
    if b0_domains:
        print("\nB0 domains by tokens:")
        for dom, tok in b0_domains[:20]:
            dom_disp = dom if dom else "<EMPTY>"
            print(f"  {dom_disp}: {tok:,}")

    print("\nDomain keys (sample):")
    for dom, n in rows_by_domain.most_common(30):
        dom_disp = dom if dom else "<EMPTY>"
        print(f"  {dom_disp} (rows={n:,})")

    print("\nNote: if you ran with --max-files/--max-lines, totals are a sample.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Analyze data/outputv2 shards for band/domain/language/token availability"
    )
    ap.add_argument(
        "--path", type=str, default="data/outputv2", help="Path to outputv2 directory"
    )
    ap.add_argument(
        "--bands",
        type=str,
        nargs="+",
        default=["b0", "b1", "b2", "b3", "b4", "b5"],
        help="Shard prefixes to scan (e.g., b0 b1 ...)",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=30,
        help="Max shard files per band to scan (sample). Use 0 for none.",
    )
    ap.add_argument(
        "--max-lines",
        type=int,
        default=5000,
        help="Max JSONL lines per file to scan (sample).",
    )

    args = ap.parse_args()
    base = Path(args.path)
    if not base.exists():
        raise FileNotFoundError(f"Not found: {base}")

    limits = ScanLimits(
        max_files=(None if args.max_files < 0 else int(args.max_files)),
        max_lines_per_file=(None if args.max_lines < 0 else int(args.max_lines)),
    )

    scan_outputv2(base, bands=args.bands, limits=limits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
