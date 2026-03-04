"""Verify deterministic batch creation.

This script computes a compact signature per batch (SHA256 of chunk_id sequence)
using the same input file discovery, sharding logic, and batch iteration patterns
as the streaming pipeline.

Goal:
- If you run it multiple times against the same dataset and params, you should
  get identical per-batch signatures.

Notes:
- Batch membership is defined by *streaming order* (file order + line order).
  If you insert/delete a row earlier in the stream, later batches will shift.

This verifier supports:
- JSONL inputs (matches the streaming pipeline's JSONL batch processor)
- Parquet inputs (deterministic row-order iteration per file)

Stage-wise reporting:
- You can optionally ask for a stage-labeled report (1B/3B/8B/70B/etc).
    Stage does not affect batch creation; the signatures will be identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import xxhash
from src.io.batch_processor import BatchProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class OutputFingerprint:
    """Order-independent fingerprint of a set/multiset of chunk_ids.

    We avoid sorting potentially huge outputs. Instead we combine per-id hashes.
    This is strong enough for regression/determinism checking, but is not a
    cryptographic proof of equality.
    """

    count: int
    xor64: int
    sum64: int
    min_id: Optional[str]
    max_id: Optional[str]


def _update_fingerprint(fp: OutputFingerprint, chunk_id: str) -> OutputFingerprint:
    cid = str(chunk_id)
    h = xxhash.xxh64(cid.encode("utf-8")).intdigest()
    fp.count += 1
    fp.xor64 ^= int(h)
    fp.sum64 = (int(fp.sum64) + int(h)) & ((1 << 64) - 1)
    fp.min_id = cid if fp.min_id is None or cid < fp.min_id else fp.min_id
    fp.max_id = cid if fp.max_id is None or cid > fp.max_id else fp.max_id
    return fp


def _empty_fingerprint() -> OutputFingerprint:
    return OutputFingerprint(count=0, xor64=0, sum64=0, min_id=None, max_id=None)


def _iter_selected_indices_rows(stage_dir: Path) -> Iterator[Dict[str, Any]]:
    """Yield selected-indices rows from either streaming parts or legacy single file."""
    # Streaming parts
    part_files = sorted(stage_dir.glob("selected_indices_part_*"))
    if not part_files:
        # Legacy single-file layout
        for legacy_name in (
            "selected_indices.parquet",
            "selected_indices.jsonl",
            "selected_indices.csv",
        ):
            p = stage_dir / legacy_name
            if p.exists():
                part_files = [p]
                break

    for p in part_files:
        suf = p.suffix.lower()
        if suf == ".parquet":
            try:
                import pandas as pd
            except Exception as e:
                raise SystemExit("pandas is required to read parquet indices") from e
            df = pd.read_parquet(p)
            for row in df.to_dict(orient="records"):
                if isinstance(row, dict):
                    yield row
        elif suf == ".jsonl" or suf == ".json":
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
        elif suf == ".csv":
            try:
                import pandas as pd
            except Exception as e:
                raise SystemExit("pandas is required to read csv indices") from e
            df = pd.read_csv(p)
            for row in df.to_dict(orient="records"):
                if isinstance(row, dict):
                    yield row
        else:
            # Ignore unrelated files (e.g., manifest.json) that match glob in some environments
            continue


def fingerprint_stage_outputs(stage_dir: Path) -> Dict[str, Any]:
    fp = _empty_fingerprint()
    token_sum = 0
    token_field = None

    seen_any = False
    for row in _iter_selected_indices_rows(stage_dir):
        seen_any = True
        cid = row.get("chunk_id")
        if cid is None:
            continue
        _update_fingerprint(fp, str(cid))

        # Token totals are useful but optional.
        if token_field is None:
            if "token_count" in row:
                token_field = "token_count"
            elif "token_count_estimate" in row:
                token_field = "token_count_estimate"

        if token_field is not None:
            try:
                token_sum += int(row.get(token_field) or 0)
            except Exception:
                pass

    return {
        "has_indices": bool(seen_any),
        "fingerprint": asdict(fp),
        "token_sum": int(token_sum),
        "token_field": token_field,
    }


def _load_manifest_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _stable_manifest_subset(m: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract a stable subset of manifest fields for comparison.

    We intentionally ignore fields that commonly vary run-to-run (timestamps,
    environment metadata) and keep fields that should be deterministic.
    """
    if not isinstance(m, dict):
        return None

    comp = m.get("composition") if isinstance(m.get("composition"), dict) else None
    out = {
        "stage_name": m.get("stage_name") or m.get("stage"),
        "target_tokens": m.get("target_tokens"),
        "actual_tokens": m.get("actual_tokens") or m.get("selected_tokens"),
        "selected_chunks_count": m.get("selected_chunks_count")
        or m.get("selected_chunks"),
        "shard_id": m.get("shard_id"),
        "num_shards": m.get("num_shards"),
        "config_hash": m.get("config_hash"),
        "curriculum_hash": m.get("curriculum_hash"),
        "composition": {
            "band_distribution": (comp.get("band_distribution") if comp else None),
            "domain_distribution": (comp.get("domain_distribution") if comp else None),
            "language_distribution": (
                comp.get("language_distribution") if comp else None
            ),
        },
    }
    return out


def compare_output_dirs(
    *,
    outputs_a: str,
    outputs_b: str,
    stages: Optional[List[str]] = None,
    include_shard_manifests: bool = True,
) -> Dict[str, Any]:
    a_root = Path(outputs_a)
    b_root = Path(outputs_b)
    if not a_root.exists():
        raise SystemExit(f"outputs-a does not exist: {outputs_a}")
    if not b_root.exists():
        raise SystemExit(f"outputs-b does not exist: {outputs_b}")

    if stages:
        stage_list = [str(s) for s in stages]
    else:
        a_stages = {p.name for p in a_root.iterdir() if p.is_dir()}
        b_stages = {p.name for p in b_root.iterdir() if p.is_dir()}
        stage_list = sorted(a_stages | b_stages)

    stage_reports: Dict[str, Any] = {}
    ok_all = True

    for stage in stage_list:
        a_stage = a_root / stage
        b_stage = b_root / stage

        a_fp = (
            fingerprint_stage_outputs(a_stage)
            if a_stage.exists()
            else {"has_indices": False}
        )
        b_fp = (
            fingerprint_stage_outputs(b_stage)
            if b_stage.exists()
            else {"has_indices": False}
        )

        a_manifest = (
            _stable_manifest_subset(_load_manifest_json(a_stage / "manifest.json"))
            if a_stage.exists()
            else None
        )
        b_manifest = (
            _stable_manifest_subset(_load_manifest_json(b_stage / "manifest.json"))
            if b_stage.exists()
            else None
        )

        shard_manifests_a = {}
        shard_manifests_b = {}
        if include_shard_manifests:
            if a_stage.exists():
                for p in sorted(a_stage.glob("manifest_shard*.json")):
                    shard_manifests_a[p.name] = _stable_manifest_subset(
                        _load_manifest_json(p)
                    )
            if b_stage.exists():
                for p in sorted(b_stage.glob("manifest_shard*.json")):
                    shard_manifests_b[p.name] = _stable_manifest_subset(
                        _load_manifest_json(p)
                    )

        match = True
        reasons: List[str] = []

        if bool(a_fp.get("has_indices")) != bool(b_fp.get("has_indices")):
            match = False
            reasons.append("indices_presence_mismatch")

        if a_fp.get("has_indices") and b_fp.get("has_indices"):
            if a_fp.get("fingerprint") != b_fp.get("fingerprint"):
                match = False
                reasons.append("chunk_id_fingerprint_mismatch")
            if int(a_fp.get("token_sum") or 0) != int(b_fp.get("token_sum") or 0):
                match = False
                reasons.append("token_sum_mismatch")

        if a_manifest != b_manifest:
            # Not always fatal for users, but for determinism it generally should match.
            match = False
            reasons.append("manifest_mismatch")

        if include_shard_manifests and shard_manifests_a != shard_manifests_b:
            match = False
            reasons.append("shard_manifests_mismatch")

        ok_all = ok_all and match
        stage_reports[stage] = {
            "match": bool(match),
            "reasons": reasons,
            "a": {
                "stage_dir": str(a_stage),
                "indices": a_fp,
                "manifest": a_manifest,
                "shard_manifests": shard_manifests_a,
            },
            "b": {
                "stage_dir": str(b_stage),
                "indices": b_fp,
                "manifest": b_manifest,
                "shard_manifests": shard_manifests_b,
            },
        }

    return {
        "mode": "compare_outputs",
        "meta": {
            "outputs_a": str(a_root),
            "outputs_b": str(b_root),
            "stages": stage_list,
            "include_shard_manifests": bool(include_shard_manifests),
        },
        "ok": bool(ok_all),
        "stages": stage_reports,
    }


@dataclass
class BatchSignature:
    batch_idx: int
    num_chunks: int
    first_chunk_id: Optional[str]
    last_chunk_id: Optional[str]
    sha256: str


def _sha256_of_chunk_ids(chunk_ids: List[str]) -> str:
    h = hashlib.sha256()
    for cid in chunk_ids:
        h.update(cid.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _normalize_chunk_id_from_mapping(data: Dict[str, Any]) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    return data.get("chunk_id") or data.get("uid") or data.get("guid") or data.get("id")


def _iter_parquet_rows(
    filepath: str,
    *,
    max_rows: Optional[int],
) -> Iterator[Dict[str, Any]]:
    """Iterate Parquet rows in stable, file-defined order.

    This intentionally avoids pandas DataFrame iteration over concatenated files,
    and instead reads in record batches in the physical order stored in the file.

    S3 parquet streaming is intentionally not supported here (requires additional
    filesystem dependencies); use local parquet for this verifier.
    """
    if str(filepath).lower().startswith("s3://"):
        raise SystemExit(
            "Parquet input via s3:// is not supported by this verifier yet; "
            "download locally or use jsonl inputs."
        )

    try:
        import pyarrow.parquet as pq
    except Exception as e:
        raise SystemExit(
            "pyarrow is required to verify parquet determinism; install pyarrow or use --input-format jsonl"
        ) from e

    pf = pq.ParquetFile(filepath)
    emitted = 0
    for record_batch in pf.iter_batches():
        if max_rows is not None and emitted >= int(max_rows):
            return

        # Convert to python objects deterministically.
        # (to_pylist preserves batch row order)
        for row in record_batch.to_pylist():
            if max_rows is not None and emitted >= int(max_rows):
                return
            emitted += 1
            if isinstance(row, dict):
                yield row
            else:
                # Shouldn't happen with to_pylist(), but be defensive.
                yield {"_value": row}


def _iter_batches_from_parquet_like_pipeline(
    *,
    input_path: str,
    batch_size: int,
    max_rows: Optional[int],
    shard_id: int,
    num_shards: int,
    shard_key: str,
) -> Iterator[Tuple[int, List[Tuple[str, Dict[str, Any]]]]]:
    bp = BatchProcessor(batch_size=int(batch_size))
    files = bp.list_input_files(input_path, "parquet")
    if not files:
        raise SystemExit(f"No Parquet files found under: {input_path}")

    row_level_shard = int(num_shards) > 1 and len(files) == 1
    if not row_level_shard:
        files = bp.shard_files(files, int(shard_id), int(num_shards))

    emitted = 0
    batch_idx = 0
    batch: List[Tuple[str, Dict[str, Any]]] = []

    for f in files:
        for row in _iter_parquet_rows(
            f, max_rows=(None if max_rows is None else int(max_rows) - emitted)
        ):
            if max_rows is not None and emitted >= int(max_rows):
                break

            if not isinstance(row, dict):
                continue

            chunk_id = _normalize_chunk_id_from_mapping(row)

            if row_level_shard and int(num_shards) > 1:
                key_val = row.get(shard_key)
                if shard_key == "chunk_id" and not key_val:
                    key_val = chunk_id
                if not key_val:
                    # Deterministic fallback for missing shard key: use overall emitted row index.
                    key_bytes = str(emitted).encode("utf-8")
                else:
                    key_bytes = str(key_val).encode("utf-8")
                h = xxhash.xxh64(key_bytes).intdigest()
                if int(h % int(num_shards)) != int(shard_id):
                    emitted += 1
                    continue

            batch.append((chunk_id, row))
            emitted += 1

            if len(batch) >= int(batch_size):
                yield batch_idx, batch
                batch_idx += 1
                batch = []

        if max_rows is not None and emitted >= int(max_rows):
            break

    if batch:
        yield batch_idx, batch


def iter_batches_like_pipeline(
    *,
    input_path: str,
    input_format: str,
    batch_size: int,
    max_rows: Optional[int],
    shard_id: int,
    num_shards: int,
    shard_key: str,
) -> Iterator[Tuple[int, List[Tuple[str, Dict[str, Any]]]]]:
    bp = BatchProcessor(batch_size=int(batch_size))

    fmt = input_format.lower()
    if fmt == "auto":
        lower = str(input_path).lower()
        if lower.endswith(".jsonl"):
            fmt = "jsonl"
        elif lower.endswith(".parquet"):
            fmt = "parquet"
        else:
            jsonl_files = bp.list_input_files(input_path, "jsonl")
            parquet_files = bp.list_input_files(input_path, "parquet")
            if jsonl_files:
                fmt = "jsonl"
            elif parquet_files:
                fmt = "parquet"
            else:
                raise SystemExit(f"No JSONL or Parquet files found under: {input_path}")

    if fmt == "jsonl":
        files = bp.list_input_files(input_path, "jsonl")
        if not files:
            raise SystemExit(f"No JSONL files found under: {input_path}")

        row_level_shard = int(num_shards) > 1 and len(files) == 1
        if not row_level_shard:
            files = bp.shard_files(files, int(shard_id), int(num_shards))

        emitted = 0
        batch_idx = 0
        for f in files:
            for batch in bp.batch_iterator(
                str(f),
                max_chunks=max_rows,
                shard_id=(int(shard_id) if row_level_shard else 0),
                num_shards=(int(num_shards) if row_level_shard else 1),
                shard_key=shard_key,
            ):
                if max_rows is not None:
                    remaining = int(max_rows) - emitted
                    if remaining <= 0:
                        return
                    if len(batch) > remaining:
                        batch = batch[:remaining]
                emitted += len(batch)
                yield batch_idx, batch
                batch_idx += 1
        return

    if fmt == "parquet":
        yield from _iter_batches_from_parquet_like_pipeline(
            input_path=input_path,
            batch_size=batch_size,
            max_rows=max_rows,
            shard_id=shard_id,
            num_shards=num_shards,
            shard_key=shard_key,
        )
        return

    raise SystemExit(
        f"Unsupported input_format for this verifier: {input_format} (use auto|jsonl|parquet)"
    )


def compute_signatures(
    *,
    input_path: str,
    input_format: str,
    batch_size: int,
    max_rows: Optional[int],
    shard_id: int,
    num_shards: int,
    shard_key: str,
) -> Dict[str, Any]:
    signatures: List[BatchSignature] = []
    total_chunks = 0

    for batch_idx, batch in iter_batches_like_pipeline(
        input_path=input_path,
        input_format=input_format,
        batch_size=batch_size,
        max_rows=max_rows,
        shard_id=shard_id,
        num_shards=num_shards,
        shard_key=shard_key,
    ):
        chunk_ids = [cid for cid, _ in batch if cid is not None]
        total_chunks += len(chunk_ids)
        sig = BatchSignature(
            batch_idx=int(batch_idx),
            num_chunks=len(chunk_ids),
            first_chunk_id=(chunk_ids[0] if chunk_ids else None),
            last_chunk_id=(chunk_ids[-1] if chunk_ids else None),
            sha256=_sha256_of_chunk_ids(chunk_ids),
        )
        signatures.append(sig)

    out: Dict[str, Any] = {
        "meta": {
            "input_path": input_path,
            "input_format": input_format,
            "batch_size": int(batch_size),
            "max_rows": (None if max_rows is None else int(max_rows)),
            "shard_id": int(shard_id),
            "num_shards": int(num_shards),
            "shard_key": shard_key,
        },
        "total_batches": len(signatures),
        "total_chunks": int(total_chunks),
        "batches": [asdict(s) for s in signatures],
    }

    return out


def add_stage_wise_labels(report: Dict[str, Any], stages: List[str]) -> Dict[str, Any]:
    """Return a copy of the report with stage-wise labels added."""
    stages_norm = [str(s).strip() for s in (stages or []) if str(s).strip()]
    if not stages_norm:
        return report

    stage_reports: Dict[str, Any] = {}
    for stage in stages_norm:
        stage_reports[stage] = {
            "meta": {**(report.get("meta") or {}), "stage_name": stage},
            "total_batches": report.get("total_batches"),
            "total_chunks": report.get("total_chunks"),
            "batches": report.get("batches"),
        }

    return {
        **report,
        "stages": stages_norm,
        "stage_reports": stage_reports,
    }


def render_stage_wise_markdown(report: Dict[str, Any]) -> str:
    stages = report.get("stages") or []
    batches = report.get("batches") or []

    lines: List[str] = []
    lines.append("# Batch Determinism Signatures\n")

    meta = report.get("meta") or {}
    lines.append("## Meta\n")
    lines.append(f"- input_path: {meta.get('input_path')}\n")
    lines.append(f"- input_format: {meta.get('input_format')}\n")
    lines.append(f"- batch_size: {meta.get('batch_size')}\n")
    lines.append(f"- max_rows: {meta.get('max_rows')}\n")
    lines.append(f"- shard_id: {meta.get('shard_id')}\n")
    lines.append(f"- num_shards: {meta.get('num_shards')}\n")
    lines.append(f"- shard_key: {meta.get('shard_key')}\n")

    if stages:
        lines.append("\n## Stages\n")
        lines.append("- " + ", ".join(stages) + "\n")

    lines.append("\n## Batches\n")
    lines.append(
        "| batch_idx | num_chunks | first_chunk_id | last_chunk_id | sha256 |\n"
    )
    lines.append("|---:|---:|---|---|---|\n")
    for b in batches:
        lines.append(
            "| %s | %s | %s | %s | %s |\n"
            % (
                b.get("batch_idx"),
                b.get("num_chunks"),
                b.get("first_chunk_id") or "",
                b.get("last_chunk_id") or "",
                b.get("sha256"),
            )
        )

    if stages:
        lines.append("\n## Stage-Wise Labels\n")
        for stage in stages:
            lines.append(f"### {stage}\n")
            lines.append(
                "These signatures are identical across stages; stage labeling is for convenience.\n"
            )
            lines.append("- total_batches: %s\n" % (report.get("total_batches"),))
            lines.append("- total_chunks: %s\n" % (report.get("total_chunks"),))

    return "".join(lines)


def render_output_compare_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Stage Output Comparison\n")

    meta = report.get("meta") or {}
    lines.append("\n## Meta\n")
    lines.append(f"- outputs_a: {meta.get('outputs_a')}\n")
    lines.append(f"- outputs_b: {meta.get('outputs_b')}\n")
    lines.append(f"- include_shard_manifests: {meta.get('include_shard_manifests')}\n")

    lines.append("\n## Results\n")
    lines.append(f"- ok: {report.get('ok')}\n")

    stages = report.get("stages") or {}
    for stage_name in sorted(stages.keys()):
        s = stages[stage_name] or {}
        lines.append(f"\n### {stage_name}\n")
        lines.append(f"- match: {s.get('match')}\n")
        reasons = s.get("reasons") or []
        if reasons:
            lines.append(f"- reasons: {', '.join(reasons)}\n")

        a_idx = (s.get("a") or {}).get("indices") or {}
        b_idx = (s.get("b") or {}).get("indices") or {}
        if a_idx.get("has_indices") or b_idx.get("has_indices"):
            lines.append("- indices:\n")
            lines.append(
                f"  - a_count: {((a_idx.get('fingerprint') or {}).get('count'))} token_sum: {a_idx.get('token_sum')}\n"
            )
            lines.append(
                f"  - b_count: {((b_idx.get('fingerprint') or {}).get('count'))} token_sum: {b_idx.get('token_sum')}\n"
            )

    return "".join(lines)


def compare_runs(baseline: Dict[str, Any], current: Dict[str, Any]) -> Tuple[bool, str]:
    b = baseline.get("batches") or []
    c = current.get("batches") or []

    if len(b) != len(c):
        return False, f"Batch count differs: baseline={len(b)} current={len(c)}"

    for i, (bb, cc) in enumerate(zip(b, c)):
        if bb.get("sha256") != cc.get("sha256"):
            return (
                False,
                "First mismatch at batch_idx=%s: baseline_sha=%s current_sha=%s"
                % (i, bb.get("sha256"), cc.get("sha256")),
            )
        if bb.get("num_chunks") != cc.get("num_chunks"):
            return (
                False,
                "Chunk count mismatch at batch_idx=%s: baseline_n=%s current_n=%s"
                % (i, bb.get("num_chunks"), cc.get("num_chunks")),
            )

    return True, "OK"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify deterministic batch membership (JSONL or Parquet), optionally with stage-wise labels."
    )

    ap.add_argument(
        "--compare-outputs",
        action="store_true",
        help="Compare stage-wise coreset outputs between two runs (selected_indices + manifests).",
    )
    ap.add_argument(
        "--outputs-a",
        default=None,
        help="Run A output_coreset_path (directory containing stage folders like 1B/3B/...).",
    )
    ap.add_argument(
        "--outputs-b",
        default=None,
        help="Run B output_coreset_path (directory containing stage folders like 1B/3B/...).",
    )
    ap.add_argument(
        "--no-shard-manifests",
        action="store_true",
        help="Skip comparing manifest_shard*.json files.",
    )
    ap.add_argument("--input-path", required=False, help="Dataset file or directory.")
    ap.add_argument(
        "--input-format",
        default="auto",
        choices=["auto", "jsonl", "parquet"],
        help="Input format (auto detects by extension or by files under input-path).",
    )
    ap.add_argument("--batch-size", type=int, default=10_000, help="Rows per batch.")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit rows processed.")
    ap.add_argument("--shard-id", type=int, default=0, help="Shard id.")
    ap.add_argument("--num-shards", type=int, default=1, help="Num shards.")
    ap.add_argument(
        "--shard-key", default="chunk_id", help="Shard key for row-level sharding."
    )

    ap.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help="Optional stage labels to attach (e.g., --stages 1B 3B 8B 70B).",
    )
    ap.add_argument(
        "--stage-report-md",
        default=None,
        help="Optional path to write a stage-wise Markdown report.",
    )

    ap.add_argument(
        "--outputs-report-md",
        default=None,
        help="Optional path to write an output-compare Markdown report (when --compare-outputs is used).",
    )

    ap.add_argument("--out", default=None, help="Write signatures JSON to this path.")
    ap.add_argument(
        "--baseline", default=None, help="Baseline signatures JSON to compare against."
    )

    args = ap.parse_args()

    if args.compare_outputs:
        if not args.outputs_a or not args.outputs_b:
            raise SystemExit("--compare-outputs requires --outputs-a and --outputs-b")

        report = compare_output_dirs(
            outputs_a=args.outputs_a,
            outputs_b=args.outputs_b,
            stages=(list(args.stages) if args.stages else None),
            include_shard_manifests=(not bool(args.no_shard_manifests)),
        )

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )

        if args.outputs_report_md:
            md_path = Path(args.outputs_report_md)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(render_output_compare_markdown(report), encoding="utf-8")

        print("OK" if report.get("ok") else "MISMATCH")
        return 0 if report.get("ok") else 2

    if not args.input_path:
        raise SystemExit("--input-path is required unless --compare-outputs is used")

    current = compute_signatures(
        input_path=args.input_path,
        input_format=args.input_format,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        shard_key=args.shard_key,
    )

    if args.stages:
        current = add_stage_wise_labels(current, list(args.stages))

    if args.stage_report_md:
        md_path = Path(args.stage_report_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_stage_wise_markdown(current), encoding="utf-8")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(current, indent=2, sort_keys=True), encoding="utf-8"
        )

    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        ok, msg = compare_runs(baseline, current)
        print(msg)
        return 0 if ok else 2

    print(
        f"Wrote {current['total_batches']} batch signatures for {current['total_chunks']} chunks"
        if args.out
        else f"Computed {current['total_batches']} batch signatures for {current['total_chunks']} chunks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
