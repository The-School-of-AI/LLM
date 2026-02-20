"""Merge sharded stage manifests into a single stage-level manifest.

Sharded runs write per-stage manifest files like:
  output/coresets/<stage>/manifest_shard000.json

This tool aggregates shard-level stats (tokens, chunk counts, distributions,
availability stats, etc.) into a single manifest (default: manifest.json)
so downstream validators/tools can reason about the full stage output.

Usage:
  python tools/merge_sharded_manifests.py \
    --coreset-root output/coresets --stages 1B 3B 8B 70B --overwrite

Windows note:
  Use `py -3` instead of `python` if needed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_UTC = _dt.timezone.utc


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest is not a JSON object: {path}")
    return data


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_iso(ts: Any) -> Optional[_dt.datetime]:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        # Accept both '...Z' and '+00:00' forms.
        return _dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _flatten_total_distribution(d: Any) -> Dict[str, float]:
    """Return a flat mapping (key->ratio) from possibly nested distribution dicts."""
    if d is None:
        return {}
    if isinstance(d, dict):
        total = d.get("total")
        if isinstance(total, dict):
            return {str(k): float(v) for k, v in total.items() if _is_number(v)}
        # Already flat
        return {str(k): float(v) for k, v in d.items() if _is_number(v)}
    return {}


def _is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _weighted_merge_ratios(
    shard_actual_tokens: List[int], shard_ratio_maps: List[Dict[str, float]]
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Convert per-shard ratio maps into token counts then back into ratios."""
    token_counts: Dict[str, int] = {}
    total_tokens = sum(shard_actual_tokens)
    for tokens, ratio_map in zip(shard_actual_tokens, shard_ratio_maps, strict=False):
        for key, ratio in ratio_map.items():
            try:
                r = float(ratio)
            except Exception:
                continue
            token_counts[key] = token_counts.get(key, 0) + int(round(tokens * r))

    ratios: Dict[str, float] = {}
    if total_tokens > 0:
        for key, tok in token_counts.items():
            ratios[key] = tok / total_tokens
    return token_counts, ratios


def _merge_by_band_domain(
    shard_band_token_counts: List[Dict[str, int]],
    shard_by_band_domain_ratio: List[Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """Merge domain distribution by band.

    Each shard provides a mapping:
      by_band[band][domain] = ratio_within_band

    We convert these to domain token counts per band using band token totals.
    """

    band_domain_tokens: Dict[str, Dict[str, int]] = {}
    band_totals: Dict[str, int] = {}

    for band_tokens, by_band in zip(
        shard_band_token_counts, shard_by_band_domain_ratio, strict=False
    ):
        if not isinstance(by_band, dict):
            continue
        for band, dom_ratios in by_band.items():
            if not isinstance(dom_ratios, dict):
                continue
            band_tok = int(band_tokens.get(band, 0))
            band_totals[band] = band_totals.get(band, 0) + band_tok
            for dom, ratio in dom_ratios.items():
                if not _is_number(ratio):
                    continue
                tok = int(round(band_tok * float(ratio)))
                band_domain_tokens.setdefault(band, {})
                band_domain_tokens[band][dom] = (
                    band_domain_tokens[band].get(dom, 0) + tok
                )

    out: Dict[str, Dict[str, float]] = {}
    for band, dom_tokens in band_domain_tokens.items():
        total = band_totals.get(band, 0)
        if total <= 0:
            continue
        out[band] = {
            dom: tok / total for dom, tok in sorted(dom_tokens.items()) if tok > 0
        }
    return out


def _merge_availability_stats(shards: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    merged: Dict[str, Any] = {}
    has_any = False

    total_tokens = 0
    total_chunks = 0
    tokens_by_band: Dict[str, int] = {}
    chunks_by_band: Dict[str, int] = {}
    definition: Optional[str] = None

    for m in shards:
        a = m.get("availability_stats")
        if not isinstance(a, dict):
            continue
        has_any = True

        total_tokens += _safe_int(a.get("eligible_unused_tokens_total"), 0)
        total_chunks += _safe_int(a.get("eligible_unused_chunks_total"), 0)

        tb = a.get("eligible_unused_tokens_by_band")
        if isinstance(tb, dict):
            for k, v in tb.items():
                tokens_by_band[str(k)] = tokens_by_band.get(str(k), 0) + _safe_int(v, 0)

        cb = a.get("eligible_unused_chunks_by_band")
        if isinstance(cb, dict):
            for k, v in cb.items():
                chunks_by_band[str(k)] = chunks_by_band.get(str(k), 0) + _safe_int(v, 0)

        if definition is None and isinstance(a.get("definition"), str):
            definition = a.get("definition")

    if not has_any:
        return None

    merged["eligible_unused_tokens_total"] = total_tokens
    merged["eligible_unused_chunks_total"] = total_chunks
    if tokens_by_band:
        merged["eligible_unused_tokens_by_band"] = dict(sorted(tokens_by_band.items()))
    if chunks_by_band:
        merged["eligible_unused_chunks_by_band"] = dict(sorted(chunks_by_band.items()))
    if definition:
        merged["definition"] = definition
    return merged


def _merge_dedup_stats(shards: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    # Sum counts where present.
    has_any = False
    out = {
        "exact_duplicates_removed": 0,
        "near_duplicates_removed": 0,
        "total_chunks_before": 0,
        "total_chunks_after": 0,
        "total_tokens_before": 0,
        "total_tokens_after": 0,
    }

    for m in shards:
        d = m.get("dedup_stats")
        if not isinstance(d, dict):
            continue
        has_any = True
        for k in list(out.keys()):
            out[k] += _safe_int(d.get(k), 0)

    if not has_any:
        return None

    before = out["total_tokens_before"]
    after = out["total_tokens_after"]
    out["dedup_ratio"] = (before - after) / before if before else 0.0
    return out


def _merge_coverage_audit(
    shards: List[Dict[str, Any]], shard_actual_tokens: List[int]
) -> Optional[Dict[str, Any]]:
    audits = [m.get("coverage_audit") for m in shards]
    if not all(isinstance(a, dict) for a in audits if a is not None):
        # If some shards have audits and others do not, drop (avoid misleading data).
        if any(a is not None for a in audits):
            return None
        return None

    dict_audits = [a for a in audits if isinstance(a, dict)]
    if not dict_audits:
        return None

    # Conservative: passed only if all passed.
    passed = all(bool(a.get("passed")) for a in dict_audits)

    # expected_coverage and tolerance should be identical; if not, we keep the first.
    expected = dict_audits[0].get("expected_coverage")
    tolerance = dict_audits[0].get("tolerance")

    # Weighted merge of actual_coverage.
    total_tokens = sum(shard_actual_tokens)
    actual_weighted: Dict[str, float] = {}
    for a, tok in zip(dict_audits, shard_actual_tokens, strict=False):
        ac = a.get("actual_coverage")
        if not isinstance(ac, dict) or total_tokens <= 0:
            continue
        for k, v in ac.items():
            if not _is_number(v):
                continue
            actual_weighted[str(k)] = actual_weighted.get(str(k), 0.0) + (
                float(v) * (tok / total_tokens)
            )

    violations: List[str] = []
    for a in dict_audits:
        v = a.get("violations")
        if isinstance(v, list):
            violations.extend([str(x) for x in v if x is not None])

    return {
        "passed": passed,
        "expected_coverage": expected if isinstance(expected, dict) else {},
        "actual_coverage": dict(sorted(actual_weighted.items())),
        "tolerance": tolerance,
        "violations": sorted(set(violations)),
    }


def merge_stage_manifests(
    stage_dir: Path, *, overwrite: bool, output_name: str, strict: bool
) -> Optional[Path]:
    shard_paths = sorted(stage_dir.glob("manifest_shard*.json"))
    if not shard_paths:
        return None

    manifests = [_read_json(p) for p in shard_paths]

    stage_name = manifests[0].get("stage_name") or stage_dir.name
    # Sanity check stage names
    for m, p in zip(manifests, shard_paths, strict=False):
        m_stage = m.get("stage_name")
        if m_stage and str(m_stage) != str(stage_name):
            msg = f"Stage mismatch in {p}: stage_name={m_stage} expected={stage_name}"
            if strict:
                raise ValueError(msg)
            print(f"[WARN] {msg}")

    shard_actual_tokens = [_safe_int(m.get("actual_tokens"), 0) for m in manifests]

    out: Dict[str, Any] = {}
    out["stage_name"] = stage_name

    # Stable merged coreset id derived from shard coreset ids + config hash + stage.
    shard_coreset_ids = [str(m.get("coreset_id") or "") for m in manifests]
    shard_config_hashes = [str(m.get("config_hash") or "") for m in manifests]
    merged_id_material = json.dumps(
        {
            "stage": stage_name,
            "coreset_ids": shard_coreset_ids,
            "config_hashes": shard_config_hashes,
            "num_shards": len(manifests),
        },
        sort_keys=True,
    )
    out["coreset_id"] = _sha256_text("merged:" + merged_id_material)

    # Target tokens: prefer a computed global effective target if present.
    stage_target_scale = manifests[0].get("stage_target_scale")
    num_shards = int(manifests[0].get("num_shards") or len(manifests))
    out["num_shards"] = len(manifests)
    out["stage_target_scale"] = stage_target_scale

    target_tokens_global = manifests[0].get("target_tokens_global")
    if target_tokens_global is not None and _is_number(target_tokens_global):
        out["target_tokens_global"] = int(float(target_tokens_global))
        if _is_number(stage_target_scale):
            out["target_tokens"] = int(
                round(float(target_tokens_global) * float(stage_target_scale))
            )
        else:
            out["target_tokens"] = int(float(target_tokens_global))
    else:
        # Fallback: sum per-shard targets
        per_shard_targets = [
            _safe_int(m.get("target_tokens") or m.get("target_tokens_shard"), 0)
            for m in manifests
        ]
        out["target_tokens"] = sum(per_shard_targets)

    out["actual_tokens"] = sum(shard_actual_tokens)
    out["selected_chunks_count"] = sum(
        _safe_int(m.get("selected_chunks_count"), 0) for m in manifests
    )

    out["pipeline_version"] = manifests[0].get("pipeline_version")
    out["curriculum_version"] = manifests[0].get("curriculum_version")
    out["seed"] = manifests[0].get("seed")
    # config_hash should be stable across shards, but if it isn't, compute a stable merged hash.
    shard_config_hashes = [str(m.get("config_hash") or "") for m in manifests]
    if len({h for h in shard_config_hashes if h}) == 1:
        out["config_hash"] = manifests[0].get("config_hash")
    else:
        out["config_hash"] = _sha256_text(
            "merged_config_hash:" + json.dumps(shard_config_hashes, sort_keys=True)
        )
    out["algorithm_version"] = manifests[0].get("algorithm_version")
    out["deterministic"] = all(bool(m.get("deterministic", True)) for m in manifests)

    # created_at: use max (latest) timestamp if possible
    created_times = [_parse_iso(m.get("created_at")) for m in manifests]
    created_times_valid = [t for t in created_times if t is not None]
    if created_times_valid:
        out["created_at"] = max(created_times_valid).isoformat()
    else:
        out["created_at"] = (
            _dt.datetime.now(_UTC)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )

    # Selected chunks file reference: stage directory
    out["selected_chunks_file"] = str(Path("output") / "coresets" / str(stage_name))

    # Compose distributions
    compositions = [m.get("composition") for m in manifests]
    if any(c is not None for c in compositions):
        # Band
        shard_band_ratios = []
        shard_lang_ratios = []
        shard_domain_total_ratios = []
        shard_domain_by_band_ratios = []
        shard_band_token_counts: List[Dict[str, int]] = []

        for c in compositions:
            c = c if isinstance(c, dict) else {}
            band_ratio_map = _flatten_total_distribution(c.get("band_distribution"))
            shard_band_ratios.append(band_ratio_map)
            shard_lang_ratios.append(
                _flatten_total_distribution(c.get("language_distribution"))
            )

            dom = c.get("domain_distribution")
            shard_domain_total_ratios.append(_flatten_total_distribution(dom))

            by_band: Dict[str, Dict[str, float]] = {}
            if isinstance(dom, dict) and isinstance(dom.get("by_band"), dict):
                for band, dom_map in dom.get("by_band", {}).items():
                    if not isinstance(dom_map, dict):
                        continue
                    by_band[str(band)] = {
                        str(k): float(v) for k, v in dom_map.items() if _is_number(v)
                    }
            shard_domain_by_band_ratios.append(by_band)

        # Per-shard band token counts (needed for domain-by-band merge).
        for tok, ratio_map in zip(shard_actual_tokens, shard_band_ratios, strict=False):
            shard_band_token_counts.append(
                {
                    b: int(round(tok * float(r)))
                    for b, r in ratio_map.items()
                    if _is_number(r)
                }
            )

        band_tokens, band_ratios = _weighted_merge_ratios(
            shard_actual_tokens, shard_band_ratios
        )
        _, lang_ratios = _weighted_merge_ratios(shard_actual_tokens, shard_lang_ratios)
        _, dom_total_ratios = _weighted_merge_ratios(
            shard_actual_tokens, shard_domain_total_ratios
        )
        dom_by_band = _merge_by_band_domain(
            shard_band_token_counts, shard_domain_by_band_ratios
        )

        out["composition"] = {
            "band_distribution": dict(sorted(band_ratios.items())),
            "domain_distribution": {
                "total": dict(sorted(dom_total_ratios.items())),
                "by_band": dom_by_band,
            },
            "language_distribution": dict(sorted(lang_ratios.items())),
        }

    # Protected slices: conservative merge (min across shards)
    protected = [m.get("protected_slices_preserved") for m in manifests]
    protected_dicts = [p for p in protected if isinstance(p, dict)]
    if protected_dicts:
        keys = sorted({k for p in protected_dicts for k in p.keys()})
        merged_protected: Dict[str, float] = {}
        for k in keys:
            vals = [p.get(k) for p in protected_dicts]
            numeric = [float(v) for v in vals if _is_number(v)]
            if numeric:
                merged_protected[k] = min(numeric)
        out["protected_slices_preserved"] = merged_protected

    out["dedup_stats"] = _merge_dedup_stats(manifests)
    out["coverage_audit"] = _merge_coverage_audit(manifests, shard_actual_tokens)

    # Rolling window stats: take worst-case max deltas
    rolling = [
        m.get("rolling_window_stats")
        for m in manifests
        if isinstance(m.get("rolling_window_stats"), dict)
    ]
    if rolling:
        window_tokens = rolling[0].get("window_tokens")
        out["rolling_window_stats"] = {
            "window_tokens": window_tokens,
            "max_band_delta": max(
                float(r.get("max_band_delta") or 0.0) for r in rolling
            ),
            "max_domain_delta": max(
                float(r.get("max_domain_delta") or 0.0) for r in rolling
            ),
        }

    out["availability_stats"] = _merge_availability_stats(manifests)

    # Merge metadata
    out["shard_merge"] = {
        "merged_at": _dt.datetime.now(_UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "shard_manifests": [p.name for p in shard_paths],
        "shard_actual_tokens": shard_actual_tokens,
        "num_shards_expected": num_shards,
        "config_hash_shards": shard_config_hashes,
    }

    # Consistency checks
    def _expect_same(key: str):
        vals = [m.get(key) for m in manifests]
        unique = {json.dumps(v, sort_keys=True, default=str) for v in vals}
        if len(unique) > 1:
            msg = f"Inconsistent '{key}' across shards in {stage_dir.name}: {vals}"
            if strict:
                raise ValueError(msg)
            print(f"[WARN] {msg}")

    for k in [
        "pipeline_version",
        "curriculum_version",
        "seed",
        "config_hash",
        "stage_target_scale",
        "target_tokens_global",
    ]:
        _expect_same(k)

    # Write output
    out_path = stage_dir / output_name
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing {out_path}. Use --overwrite."
        )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=False)
        f.write("\n")

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge sharded manifests per stage into a stage-level manifest."
    )
    parser.add_argument(
        "--coreset-root",
        default="output/coresets",
        help="Root directory containing stage subfolders",
    )
    parser.add_argument(
        "--stages",
        nargs="*",
        default=None,
        help="Stages to merge (default: all stage subfolders found)",
    )
    parser.add_argument(
        "--output-name",
        default="manifest.json",
        help="Output manifest filename (default: manifest.json)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output file if it exists"
    )
    parser.add_argument(
        "--strict", action="store_true", help="Fail on inconsistent shard metadata"
    )
    args = parser.parse_args()

    root = Path(args.coreset_root)
    if not root.exists():
        raise FileNotFoundError(f"coreset root not found: {root}")

    if args.stages is None or len(args.stages) == 0:
        stage_dirs = [
            p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
        ]
    else:
        stage_dirs = [root / s for s in args.stages]

    wrote_any = False
    for stage_dir in sorted(stage_dirs, key=lambda p: p.name):
        if not stage_dir.exists():
            print(f"[WARN] Missing stage dir: {stage_dir}")
            continue
        try:
            out_path = merge_stage_manifests(
                stage_dir,
                overwrite=args.overwrite,
                output_name=args.output_name,
                strict=args.strict,
            )
        except Exception as e:
            print(f"[ERROR] Failed merging stage {stage_dir.name}: {e}")
            return 2

        if out_path is None:
            print(f"[SKIP] {stage_dir.name}: no shard manifests found")
            continue

        wrote_any = True
        print(f"[OK] {stage_dir.name}: wrote {out_path}")

    if not wrote_any:
        print("[INFO] No sharded manifests found to merge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
