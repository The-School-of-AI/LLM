#!/usr/bin/env python3
"""Generate verification artifacts for coreset outputs.

Produces a single Markdown report with:
- Per-stage validator summary (manifest + indices checks)
- Per-stage selected id counts
- Cross-stage overlap checks (non-overlap enforcement)

This is designed to be run after a pipeline execution.

Example:
  python tools/generate_verification_artifacts.py \
    --curriculum config/curriculum.yaml \
    --output-dir output/coresets \
    --stages 1B 3B 8B 70B \
    --report-path output/manifests/verification_artifacts.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Ensure repo root imports work when run from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _read_jsonl_ids(path: Path, id_fields: Sequence[str]) -> List[str]:
    ids: List[str] = []
    if not path.exists():
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            cid = None
            for k in id_fields:
                v = obj.get(k)
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    cid = s
                    break
            if cid is not None:
                ids.append(cid)
    return ids


def _read_parquet_ids(path: Path, id_fields: Sequence[str]) -> List[str]:
    try:
        import pyarrow.parquet as pq
    except Exception:
        return []

    ids: List[str] = []
    try:
        pf = pq.ParquetFile(str(path))
    except Exception:
        return ids

    for rg in range(pf.num_row_groups):
        try:
            table = pf.read_row_group(rg, columns=list(id_fields))
        except Exception:
            continue

        schema_names = set(table.schema.names)
        cols = [table.column(f).to_pylist() for f in id_fields if f in schema_names]
        if not cols:
            continue

        n = int(table.num_rows)
        for i in range(n):
            cid = None
            for col in cols:
                v = col[i]
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    cid = s
                    break
            if cid is not None:
                ids.append(cid)

    return ids


def _load_stage_ids(stage_dir: Path, id_fields: Sequence[str]) -> List[str]:
    parquet = stage_dir / "selected_indices.parquet"
    if parquet.exists():
        return _read_parquet_ids(parquet, id_fields)

    jsonl = stage_dir / "selected_indices.jsonl"
    if jsonl.exists():
        return _read_jsonl_ids(jsonl, id_fields)

    part_files = sorted(stage_dir.glob("selected_indices_part_*.parquet"))
    if part_files:
        out: List[str] = []
        for p in part_files:
            out.extend(_read_parquet_ids(p, id_fields))
        return out

    return []


def _pairwise_overlap_counts(
    stage_sets: Dict[str, Set[str]]
) -> Dict[Tuple[str, str], int]:
    stages = sorted(stage_sets.keys())
    pair_counts: Dict[Tuple[str, str], int] = {}
    for i in range(len(stages)):
        for j in range(i + 1, len(stages)):
            a, b = stages[i], stages[j]
            pair_counts[(a, b)] = len(stage_sets[a].intersection(stage_sets[b]))
    return pair_counts


def _format_validator_summary(
    curriculum_path: Path, output_dir: Path, stages: Sequence[str]
) -> str:
    from tools.validate_coreset_outputs import CoresetValidator

    validator = CoresetValidator(str(curriculum_path), output_base_dir=str(output_dir))

    lines: List[str] = []
    lines.append("## Validator Summary\n\n")

    for stage in stages:
        report = validator.validate_stage(stage)
        summary = report.get_summary()

        lines.append(f"### {stage}\n\n")
        lines.append(
            f"- Manifest: `{report.manifest_path}`\n"
            f"- Indices: `{report.indices_path}`\n"
            f"- Checks: {summary['total_checks']} (passed={summary['by_status']['passed']}, failed={summary['by_status']['failed']})\n"
            f"- Success rate: {summary['success_rate']:.1f}%\n\n"
        )

        failed = [c for c in report.checks if not c.passed]
        if failed:
            lines.append("Failed checks (top 10):\n\n")
            for c in failed[:10]:
                lines.append(
                    f"- `{c.check_id}` [{c.severity}] {c.category}: {c.name} — {c.message}\n"
                )
            lines.append("\n")

    return "".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate verification artifacts for coreset outputs"
    )
    parser.add_argument(
        "--curriculum", type=str, required=True, help="Path to curriculum.yaml"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/coresets",
        help="Base output directory",
    )
    parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        default=["1B", "3B", "8B", "70B"],
        help="Stages to check",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="output/manifests/verification_artifacts.md",
        help="Where to write the Markdown report",
    )
    parser.add_argument(
        "--id-fields",
        type=str,
        default="chunk_id,uid,guid,id",
        help="Comma-separated identifier fields to use when reading selected indices",
    )

    args = parser.parse_args(argv)

    curriculum_path = Path(args.curriculum)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)
    stages = list(args.stages)
    id_fields = [s.strip() for s in str(args.id_fields).split(",") if s.strip()]

    if not curriculum_path.exists():
        print(f"Curriculum not found: {curriculum_path}", file=sys.stderr)
        return 2

    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Load selected ids per stage
    stage_ids: Dict[str, List[str]] = {}
    stage_sets: Dict[str, Set[str]] = {}
    for stage in stages:
        stage_dir = output_dir / stage
        ids = _load_stage_ids(stage_dir, id_fields)
        stage_ids[stage] = ids
        stage_sets[stage] = set(ids)

    # Cross-stage overlaps
    all_counts: Counter[str] = Counter()
    for st, ids in stage_sets.items():
        for cid in ids:
            all_counts[cid] += 1

    multi = {cid: c for cid, c in all_counts.items() if c > 1}
    pair_counts = _pairwise_overlap_counts(stage_sets)

    md: List[str] = []
    md.append("# Verification Artifacts\n\n")
    md.append(f"Generated at: {datetime.now().isoformat()}\n\n")
    md.append(f"Output dir: `{output_dir}`\n\n")

    md.append("## Selected Indices Counts\n\n")
    for stage in stages:
        md.append(
            f"- {stage}: rows={len(stage_ids.get(stage, [])):,} unique_ids={len(stage_sets.get(stage, set())):,}\n"
        )
    md.append("\n")

    md.append("## Cross-stage Overlap Check\n\n")
    md.append(f"- Total unique ids across all stages: {len(all_counts):,}\n")
    md.append(f"- Ids appearing in >1 stage: {len(multi):,}\n\n")

    if multi:
        md.append("Pairwise overlaps:\n\n")
        for (a, b), c in sorted(
            pair_counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1])
        ):
            if c > 0:
                md.append(f"- {a} ∩ {b}: {c}\n")
        md.append("\nTop duplicated ids (up to 20):\n\n")
        for cid, c in sorted(multi.items(), key=lambda x: (-x[1], x[0]))[:20]:
            stages_for = [s for s, st in stage_sets.items() if cid in st]
            md.append(f"- {cid}: count={c}, stages={stages_for}\n")
        md.append("\n")
    else:
        md.append("No overlaps detected.\n\n")

    # Validator summary
    try:
        md.append(_format_validator_summary(curriculum_path, output_dir, stages))
    except Exception as e:
        md.append("## Validator Summary\n\n")
        md.append(f"Validator execution failed: {e}\n\n")

    report_path.write_text("".join(md), encoding="utf-8")
    print(str(report_path))

    # Exit non-zero if overlaps were detected
    return 3 if multi else 0


if __name__ == "__main__":
    raise SystemExit(main())
