#!/usr/bin/env python3
"""Rebalance a JSONL chunk dataset to be curriculum-feasible.

This utility is intended for *sample datasets* (like large_sample_chunks.jsonl)
that are used for end-to-end testing. It rewrites band/domain/language fields so
that the selection engine can satisfy curriculum constraints (band-domain policy
and language policy) and hit scaled token targets.

It is deterministic: assignments are based only on chunk_id and token_count.

Example:
  python tools/rebalance_sample_dataset.py \
    --input data/datasets/large_sample_chunks.jsonl \
    --output data/datasets/large_sample_chunks.jsonl \
    --backup data/datasets/large_sample_chunks.orig.jsonl \
    --curriculum config/curriculum.yaml \
    --band-profile final_adaptive \
    --hi-share 0.05
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import yaml


def _hash_float(key: str) -> float:
    """Deterministic float in [0,1)."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    # Use first 8 bytes as uint64
    value = int.from_bytes(digest[:8], "big", signed=False)
    return value / 2**64


def _load_curriculum_band_domain_policy(
    curriculum_path: Path,
) -> Dict[str, Tuple[str, ...]]:
    obj = yaml.safe_load(curriculum_path.read_text(encoding="utf-8"))
    policy = obj.get("domains", {}).get("band_domain_policy", {})
    out: Dict[str, Tuple[str, ...]] = {}
    for band, domains in (policy or {}).items():
        out[str(band)] = tuple(str(d) for d in (domains or []))
    return out


def _load_stage_profile_band_weights(
    curriculum_path: Path, profile: str
) -> Dict[str, float]:
    obj = yaml.safe_load(curriculum_path.read_text(encoding="utf-8"))
    stage_profiles = (obj.get("growth_schedule", {}) or {}).get(
        "stage_profiles", {}
    ) or {}
    prof = stage_profiles.get(profile)
    if not prof:
        raise SystemExit(
            f"Unknown profile '{profile}'. Available: {sorted(stage_profiles.keys())}"
        )
    weights = prof.get("band_weights") or {}
    out: Dict[str, float] = {}
    for band, w in weights.items():
        out[str(band)] = float(w)
    total = sum(out.values())
    if total <= 0:
        raise SystemExit(f"Profile '{profile}' band_weights sum to {total}")
    # Normalize defensively
    out = {k: v / total for k, v in out.items()}
    return out


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _get_token_count(row: dict) -> int:
    try:
        return int(row.get("token_count_estimate", 0) or 0)
    except Exception:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str)
    ap.add_argument("--output", required=True, type=str)
    ap.add_argument("--backup", default=None, type=str)
    ap.add_argument("--curriculum", default="config/curriculum.yaml", type=str)
    ap.add_argument(
        "--band-profile",
        default="final_adaptive",
        type=str,
        help="growth_schedule.stage_profiles.<profile>.band_weights to target",
    )
    ap.add_argument(
        "--hi-share",
        default=0.08,
        type=float,
        help="Target Hindi token share in the output (kept under 0.08 cap)",
    )
    ap.add_argument(
        "--drop-token-ids",
        action="store_true",
        help="Remove 'token_ids' from rows so diversity scoring doesn't over-prefer rare-token chunks in sample data",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    curriculum_path = Path(args.curriculum)

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")
    if not curriculum_path.exists():
        raise SystemExit(f"Curriculum not found: {curriculum_path}")

    band_domain_policy = _load_curriculum_band_domain_policy(curriculum_path)
    band_weights = _load_stage_profile_band_weights(curriculum_path, args.band_profile)

    # First pass: total tokens
    total_tokens = 0
    rows_count = 0
    for row in _iter_jsonl(input_path):
        total_tokens += _get_token_count(row)
        rows_count += 1

    if rows_count == 0:
        raise SystemExit("Input JSONL is empty")

    # Token quotas by band (token-mass, not row-count)
    band_remaining: Dict[str, int] = {
        b: int(total_tokens * float(w)) for b, w in band_weights.items()
    }
    # Ensure all canonical bands exist
    for b in ["B0", "B1", "B2", "B3", "B4", "B5"]:
        band_remaining.setdefault(b, 0)

    hi_share = float(args.hi_share)
    if hi_share < 0.0 or hi_share > 0.08:
        raise SystemExit("--hi-share must be in [0, 0.08] to satisfy curriculum caps")
    hi_assigned_tokens = 0

    # Backup original if requested
    backup_path = Path(args.backup) if args.backup else None
    if backup_path is not None and not backup_path.exists():
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_bytes(input_path.read_bytes())

    # Second pass: rewrite (safe for in-place writes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    if output_path.resolve() == input_path.resolve():
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        write_path = temp_path
        read_path = input_path
    else:
        write_path = output_path
        read_path = input_path

    written_tokens = 0
    written_rows = 0
    with write_path.open("w", encoding="utf-8") as out:
        for row in _iter_jsonl(read_path):
            chunk_id = str(
                row.get("chunk_id") or row.get("id") or f"row_{written_rows}"
            )
            tc = _get_token_count(row)

            # Band: greedy fill by remaining quota
            band = max(band_remaining.items(), key=lambda kv: kv[1])[0]
            band_remaining[band] = int(band_remaining.get(band, 0)) - int(tc)
            row["band"] = band

            # Domain: enforce band-domain policy deterministically
            allowed_domains = band_domain_policy.get(band)
            if not allowed_domains:
                # Fallback to curriculum defaults
                allowed_domains = ("clean_web",)

            if len(allowed_domains) == 1:
                domain = allowed_domains[0]
            else:
                h = _hash_float(f"{chunk_id}:domain")
                idx = int(h * len(allowed_domains))
                if idx >= len(allowed_domains):
                    idx = len(allowed_domains) - 1
                domain = allowed_domains[idx]

            row["domain"] = domain
            row["dataset_id"] = f"ds_{domain}"

            # Language: stable per-row hash threshold so each streaming batch
            # has roughly the same hi share (avoids early-batch skew).
            lang_h = _hash_float(f"{chunk_id}:lang")
            if tc > 0 and lang_h < hi_share:
                row["language"] = "hi"
                hi_assigned_tokens += tc
            else:
                row["language"] = "en"

            if args.drop_token_ids:
                row.pop("token_ids", None)

            out.write(json.dumps(row, ensure_ascii=False) + "\n")

            written_tokens += tc
            written_rows += 1

    if written_rows == 0:
        raise SystemExit(
            "No rows were written. If you attempted an in-place rewrite and the input was empty, "
            "restore from the backup and retry."
        )

    if temp_path is not None:
        temp_path.replace(output_path)

    # Basic sanity print
    actual_hi_share = (
        (float(hi_assigned_tokens) / float(written_tokens)) if written_tokens else 0.0
    )
    print(
        f"Rebalanced {written_rows:,} rows ({written_tokens:,} tokens) -> {output_path}. "
        f"hi_share_target={hi_share:.4f}, hi_share_actual={actual_hi_share:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
