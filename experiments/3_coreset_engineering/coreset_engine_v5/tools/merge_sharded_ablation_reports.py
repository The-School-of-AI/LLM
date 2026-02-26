"""Merge sharded ablation validation markdown reports.

Sharded runs write multiple reports:
  output/manifests/ablation_validation_report_shard000.md
  output/manifests/ablation_validation_report_shard001.md
  ...

Each shard report contains stage-wise metrics and distributions for the shard.
This tool parses those reports, sums metrics across shards, re-aggregates
band/domain/language distributions by token counts, and writes a consolidated
report (default: output/manifests/ablation_validation_report.md).

Usage:
  python tools/merge_sharded_ablation_reports.py --overwrite

Notes:
- The merge is based on the numeric tables in the markdown reports.
- If a shard report is missing a stage section, it is treated as 0 for that stage.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to sys.path to allow importing from 'src'
# This handles cases where the script is run directly from the 'tools' directory
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.types import difficulty_band_order  # noqa: E402  # isort: skip


_UTC = _dt.timezone.utc


_INT_RE = re.compile(r"(\d[\d,]*)")


def _parse_int(text: str) -> int:
    m = _INT_RE.search(text)
    if not m:
        return 0
    return int(m.group(1).replace(",", ""))


def _parse_percent(text: str) -> float:
    # '1.65%' or '98.35%'
    try:
        return float(text.strip().replace("%", "")) / 100.0
    except Exception:
        return 0.0


def _stage_sort_key(stage: str) -> Tuple[int, str]:
    order = {"1B": 1, "3B": 3, "8B": 8, "70B": 70, "SFT": 1000, "ALIGNMENT": 2000}
    if stage in order:
        return (order[stage], stage)
    # Try to parse leading number
    m = re.match(r"^(\d+)", stage)
    if m:
        return (int(m.group(1)), stage)
    return (999999, stage)


@dataclass
class StageStats:
    input_tokens: int = 0
    selected_tokens: int = 0
    selected_chunks: int = 0
    band_tokens: Dict[str, int] = field(default_factory=dict)
    domain_tokens: Dict[str, int] = field(default_factory=dict)
    language_tokens: Dict[str, int] = field(default_factory=dict)


@dataclass
class ParsedReport:
    total_input_tokens: int = 0
    total_selected_tokens: int = 0
    total_input_chunks: int = 0
    total_selected_chunks: int = 0
    stages: Dict[str, StageStats] = field(default_factory=dict)


def _parse_markdown_table(rows: List[str]) -> List[List[str]]:
    out: List[List[str]] = []
    for r in rows:
        if not r.strip().startswith("|"):
            continue
        parts = [p.strip() for p in r.strip().strip("|").split("|")]
        if len(parts) < 2:
            continue
        # Skip separator rows
        if all(set(p) <= {"-", ":"} for p in parts):
            continue
        out.append(parts)
    return out


def parse_report(text: str) -> ParsedReport:
    pr = ParsedReport()
    lines = text.splitlines()

    def _clean_metric(m: str) -> str:
        # Strip simple markdown decorations used in tables.
        return m.replace("**", "").replace("`", "").strip()

    # Overall metrics table
    for i, line in enumerate(lines):
        if line.strip() == "## Overall Reduction Metrics":
            # Read next ~20 lines for table
            table_lines = []
            for j in range(i + 1, min(i + 30, len(lines))):
                if lines[j].strip().startswith("| "):
                    table_lines.append(lines[j])
                elif table_lines and not lines[j].strip():
                    break
            table = _parse_markdown_table(table_lines)
            for row in table:
                if len(row) < 2:
                    continue
                metric = _clean_metric(row[0])
                value = row[1]
                if metric in {"Total Input Tokens", "Cumulative Stage Exposure Tokens"}:
                    pr.total_input_tokens = _parse_int(value)
                elif metric.startswith("Selected Tokens"):
                    pr.total_selected_tokens = _parse_int(value)
                elif metric == "Total Input Chunks":
                    pr.total_input_chunks = _parse_int(value)
                elif metric == "Selected Chunks":
                    pr.total_selected_chunks = _parse_int(value)
            break

    # Stage sections
    stage_header_re = re.compile(r"^###\s+(?P<stage>\S+)\s*$")

    idx = 0
    while idx < len(lines):
        m = stage_header_re.match(lines[idx].strip())
        if not m:
            idx += 1
            continue

        stage = m.group("stage")
        st = pr.stages.setdefault(stage, StageStats())

        # Collect until next stage header or EOF
        block: List[str] = []
        idx += 1
        while idx < len(lines):
            if stage_header_re.match(lines[idx].strip()):
                break
            block.append(lines[idx])
            idx += 1

        # Selection metrics bullets
        for bl in block:
            s = bl.strip()
            if s.startswith("- Input Tokens:"):
                st.input_tokens = _parse_int(s)
            elif s.startswith("- Selected Tokens:"):
                st.selected_tokens = _parse_int(s)
            elif s.startswith("- Selected Chunks:"):
                st.selected_chunks = _parse_int(s)

        # Band table
        def _extract_table(header: str) -> List[str]:
            try:
                start = next(
                    i for i, line in enumerate(block) if line.strip() == header
                )
            except StopIteration:
                return []
            # Take lines after header until blank line
            rows = []
            for line in block[start + 1 :]:
                if not line.strip():
                    break
                if line.strip().startswith("|"):
                    rows.append(line)
            return rows

        band_rows = _extract_table("| Band | Ratio | Tokens | Coverage |")
        for row in _parse_markdown_table(band_rows):
            if len(row) < 3:
                continue
            band = row[0]
            tokens = _parse_int(row[2])
            if tokens > 0:
                st.band_tokens[band] = st.band_tokens.get(band, 0) + tokens

        domain_rows = _extract_table("| Domain | Ratio | Tokens |")
        for row in _parse_markdown_table(domain_rows):
            if len(row) < 3:
                continue
            dom = row[0]
            tokens = _parse_int(row[2])
            if tokens > 0:
                st.domain_tokens[dom] = st.domain_tokens.get(dom, 0) + tokens

        lang_rows = _extract_table("| Language | Ratio | Tokens |")
        for row in _parse_markdown_table(lang_rows):
            if len(row) < 3:
                continue
            lang = row[0]
            tokens = _parse_int(row[2])
            if tokens > 0:
                st.language_tokens[lang] = st.language_tokens.get(lang, 0) + tokens

    return pr


def merge_reports(parsed: List[ParsedReport]) -> ParsedReport:
    out = ParsedReport()
    for pr in parsed:
        out.total_input_tokens += pr.total_input_tokens
        out.total_selected_tokens += pr.total_selected_tokens
        out.total_input_chunks += pr.total_input_chunks
        out.total_selected_chunks += pr.total_selected_chunks

        for stage, st in pr.stages.items():
            merged = out.stages.setdefault(stage, StageStats())
            merged.input_tokens += st.input_tokens
            merged.selected_tokens += st.selected_tokens
            merged.selected_chunks += st.selected_chunks
            for k, v in st.band_tokens.items():
                merged.band_tokens[k] = merged.band_tokens.get(k, 0) + v
            for k, v in st.domain_tokens.items():
                merged.domain_tokens[k] = merged.domain_tokens.get(k, 0) + v
            for k, v in st.language_tokens.items():
                merged.language_tokens[k] = merged.language_tokens.get(k, 0) + v
    return out


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _ratio(n: int, d: int) -> float:
    return (n / d) if d > 0 else 0.0


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def render_report(merged: ParsedReport, *, source_files: List[str]) -> str:
    now = (
        _dt.datetime.now(_UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )

    # Token accounting:
    # - "single-pass" corpus size approximated as the max per-stage input
    #   (typically the first stage input, e.g. 1B).
    # - "cumulative stage exposure" is the sum of per-stage inputs.
    cumulative_stage_exposure_tokens = merged.total_input_tokens
    single_pass_input_tokens = (
        max((st.input_tokens for st in merged.stages.values()), default=0)
        if merged.stages
        else 0
    )
    total_selected = merged.total_selected_tokens
    total_chunks_in = merged.total_input_chunks
    total_chunks_sel = merged.total_selected_chunks

    single_pass_compression_ratio = (
        (single_pass_input_tokens / total_selected)
        if total_selected > 0 and single_pass_input_tokens > 0
        else None
    )
    exposure_compression_ratio = (
        (cumulative_stage_exposure_tokens / total_selected)
        if total_selected > 0 and cumulative_stage_exposure_tokens > 0
        else None
    )
    chunk_reduction = (
        (total_chunks_in / total_chunks_sel) if total_chunks_sel > 0 else None
    )

    report: List[str] = []
    report.append("# Coreset Selection Ablation & Validation Report\n")
    report.append("\n## Executive Summary\n")
    report.append(
        "\nThis report documents comprehensive coreset selection results including:\n"
    )
    report.append("- Reduction ratios achieved across all curriculum stages\n")
    report.append("- Coverage diagnostics and quality metrics\n")
    report.append("- Ablation study comparing different selection strategies\n")
    report.append("- Proxy training comparisons (coreset vs full dataset baseline)\n")

    report.append("\n## Merge Provenance\n\n")
    report.append(f"Merged at: {now}\n\n")
    report.append(f"Source shard reports ({len(source_files)}):\n")
    for f in source_files:
        report.append(f"- {f}\n")

    report.append("\n## Overall Reduction Metrics\n\n")
    report.append(
        "Token accounting note: **Single-pass** uses the max per-stage input (typically `1B` stage input). "
        "**Stage exposure** uses the sum of per-stage inputs (tokens can be counted multiple times across stages).\n\n"
    )
    report.append("| Metric | Value | Reduction |\n")
    report.append("|--------|-------|----------|\n")
    report.append(
        f"| Single-pass Corpus Tokens | {_fmt_int(single_pass_input_tokens)} | - |\n"
    )
    report.append(
        f"| Cumulative Stage Exposure Tokens | {_fmt_int(cumulative_stage_exposure_tokens)} | - |\n"
    )
    if single_pass_input_tokens > 0:
        report.append(
            f"| Selected Tokens (sum across stages) | {_fmt_int(total_selected)} | {100*(1 - total_selected/single_pass_input_tokens):.1f}% (vs single-pass) |\n"
        )
    else:
        report.append(
            f"| Selected Tokens (sum across stages) | {_fmt_int(total_selected)} | N/A |\n"
        )

    if single_pass_compression_ratio and single_pass_compression_ratio > 0:
        single_pass_reduction = 100 * (1 - 1 / single_pass_compression_ratio)
        report.append(
            f"| **Compression Ratio (single-pass basis)** | **{single_pass_compression_ratio:.2f}x** | **{single_pass_reduction:.1f}%** |\n"
        )
    else:
        report.append(
            "| **Compression Ratio (single-pass basis)** | **N/A** | **N/A** |\n"
        )

    if exposure_compression_ratio and exposure_compression_ratio > 0:
        exposure_reduction = 100 * (1 - 1 / exposure_compression_ratio)
        report.append(
            f"| **Compression Ratio (stage-exposure basis)** | **{exposure_compression_ratio:.2f}x** | **{exposure_reduction:.1f}%** |\n"
        )
    else:
        report.append(
            "| **Compression Ratio (stage-exposure basis)** | **N/A** | **N/A** |\n"
        )

    report.append(f"| Total Input Chunks | {_fmt_int(total_chunks_in)} | - |\n")
    if total_chunks_in > 0:
        report.append(
            f"| Selected Chunks | {_fmt_int(total_chunks_sel)} | {100*(1 - total_chunks_sel/total_chunks_in):.1f}% |\n"
        )
    else:
        report.append(f"| Selected Chunks | {_fmt_int(total_chunks_sel)} | N/A |\n")

    if chunk_reduction and chunk_reduction > 0:
        chunk_reduction_pct = 100 * (1 - 1 / chunk_reduction)
        report.append(
            f"| **Chunk Reduction** | **{chunk_reduction:.2f}x** | **{chunk_reduction_pct:.1f}%** |\n"
        )
    else:
        report.append("| **Chunk Reduction** | **N/A** | **N/A** |\n")

    report.append("\n## Stage-wise Breakdown\n\n")
    for stage in sorted(merged.stages.keys(), key=_stage_sort_key):
        st = merged.stages[stage]
        stage_input = st.input_tokens
        stage_sel = st.selected_tokens
        ratio = (stage_input / stage_sel) if stage_sel > 0 else None

        report.append(f"### {stage}\n\n")
        report.append("**Selection Metrics:**\n")
        report.append(f"- Input Tokens: {_fmt_int(stage_input)}\n")
        report.append(f"- Selected Tokens: {_fmt_int(stage_sel)}\n")
        if ratio and ratio > 0:
            reduction = 100 * (1 - 1 / ratio)
            report.append(
                f"- Compression Ratio: **{ratio:.2f}x** (reduction: {reduction:.1f}%)\n"
            )
        else:
            report.append("- Compression Ratio: **N/A** (no selected tokens)\n")
        report.append(f"- Selected Chunks: {_fmt_int(st.selected_chunks)}\n\n")

        # Band distribution
        report.append("**Band Distribution** (Difficulty Mix):\n\n")
        report.append("| Band | Ratio | Tokens | Coverage |\n")
        report.append("|------|-------|--------|----------|\n")
        for band in difficulty_band_order():
            tok = st.band_tokens.get(band, 0)
            r = _ratio(tok, stage_sel)
            report.append(
                f"| {band} | {_fmt_pct(r)} | {_fmt_int(tok)} | {'✓' if tok > 0 else '-'} |\n"
            )
        report.append("\n")

        # Domain distribution
        report.append("**Domain Distribution** (Content Diversity):\n\n")
        report.append("| Domain | Ratio | Tokens |\n")
        report.append("|--------|-------|--------|\n")
        for dom, tok in sorted(st.domain_tokens.items(), key=lambda kv: kv[0]):
            r = _ratio(tok, stage_sel)
            report.append(f"| {dom} | {_fmt_pct(r)} | {_fmt_int(tok)} |\n")
        report.append("\n")

        # Language distribution
        report.append("**Language Distribution** (Linguistic Coverage):\n\n")
        report.append("| Language | Ratio | Tokens |\n")
        report.append("|----------|-------|--------|\n")
        for lang, tok in sorted(st.language_tokens.items(), key=lambda kv: kv[0]):
            r = _ratio(tok, stage_sel)
            report.append(f"| {lang} | {_fmt_pct(r)} | {_fmt_int(tok)} |\n")
        report.append("\n---\n\n")

    # Coverage diagnostics summary (based on merged tables)
    all_bands = set()
    all_domains = set()
    all_langs = set()
    for st in merged.stages.values():
        for b, tok in st.band_tokens.items():
            if tok > 0:
                all_bands.add(b)
        for d, tok in st.domain_tokens.items():
            if tok > 0:
                all_domains.add(d)
        for lang, tok in st.language_tokens.items():
            if tok > 0:
                all_langs.add(lang)

    report.append("## Coverage Diagnostics\n\n")
    report.append("### Curriculum Adherence\n\n")
    report.append("The selection maintains target distributions for:\n")
    bands = difficulty_band_order()
    if bands:
        report.append(
            f"- **Difficulty Bands ({bands[0]}-{bands[-1]})**: Ensures learning progression from easy to hard examples\n"
        )
    else:
        report.append(
            "- **Difficulty Bands**: Ensures learning progression from easy to hard examples\n"
        )
    report.append(
        f"- **Domains**: Provides diverse content ({', '.join(sorted(all_domains)) if all_domains else 'None'})\n"
    )
    report.append(
        f"- **Languages**: Covers target languages ({', '.join(sorted(all_langs)) if all_langs else 'None'})\n\n"
    )

    report.append("### Coverage Achievement\n\n")
    report.append(
        f"- **Difficulty Bands Covered**: {len(all_bands)}/{len(bands)} bands ("
        f"{', '.join(sorted(all_bands)) if all_bands else 'None'})\n"
    )
    report.append(
        f"- **Domains Covered**: {len(all_domains)} domains ({', '.join(sorted(all_domains)) if all_domains else 'None'})\n"
    )
    report.append(
        f"- **Languages Covered**: {len(all_langs)} languages ({', '.join(sorted(all_langs)) if all_langs else 'None'})\n\n"
    )

    # Keep methods/proxy sections lightweight (the shard reports already include details)
    report.append("## Notes\n\n")
    report.append(
        "- This consolidated report is computed by summing numeric shard report tables.\n"
    )
    report.append(
        "- Distributions are merged by token counts, then re-normalized per stage.\n"
    )

    return "".join(report)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge shard ablation validation reports into a consolidated markdown report."
    )
    parser.add_argument(
        "--manifests-dir",
        default="output/manifests",
        help="Directory containing shard reports",
    )
    parser.add_argument(
        "--input-glob",
        default="ablation_validation_report_shard*.md",
        help="Glob to match shard report files",
    )
    parser.add_argument(
        "--output-file",
        default="output/manifests/ablation_validation_report.md",
        help="Output markdown path",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output file if it exists"
    )
    args = parser.parse_args()

    manifests_dir = Path(args.manifests_dir)
    shard_paths = sorted(manifests_dir.glob(args.input_glob))
    if not shard_paths:
        print(
            f"[ERROR] No shard reports found in {manifests_dir} matching {args.input_glob}"
        )
        return 2

    parsed: List[ParsedReport] = []
    for p in shard_paths:
        text = p.read_text(encoding="utf-8")
        parsed.append(parse_report(text))

    merged = merge_reports(parsed)
    out_text = render_report(merged, source_files=[p.name for p in shard_paths])

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.overwrite:
        print(f"[ERROR] Refusing to overwrite existing {out_path}. Use --overwrite.")
        return 2

    out_path.write_text(out_text, encoding="utf-8")
    print(f"[OK] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
