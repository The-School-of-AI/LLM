"""Post-extraction curriculum gate enforcing stage- and band-aware rules.

This module reads the metadata layer (pyarrow Table or parquet path), the
curriculum YAML, and enforces rules that require band or stage context:

- indic_not_allowed_at_stage: rejects samples in Indic languages before the
  curriculum-allowed earliest stage for that language.
- agentic_not_allowed_in_band_at_stage: rejects samples that contain agentic
  traces when the assigned band forbids agentic content.

The output is a list of rejection dicts compatible with `RejectionWriter`.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def _load_curriculum(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _stage_order_map(curriculum: Dict[str, Any]) -> Dict[str, int]:
    stages = curriculum.get("growth_schedule", {}).get("stages", []) or []
    order_map = {}
    for s in stages:
        name = s.get("name")
        order = s.get("order")
        if name is not None and order is not None:
            order_map[name] = int(order)
    return order_map


def _get_earliest_stage_for_lang(curriculum: Dict[str, Any], lang: str) -> Optional[str]:
    sec = curriculum.get("language_and_context", {}).get("language_policy", {}).get("secondary_languages", []) or []
    for entry in sec:
        if isinstance(entry, dict) and entry.get("lang") and entry.get("lang").lower() == lang.lower():
            return entry.get("earliest_stage")
    # If primary languages include it, earliest stage is first stage
    prim = curriculum.get("language_and_context", {}).get("language_policy", {}).get("primary_languages", []) or []
    for entry in prim:
        if isinstance(entry, dict) and entry.get("lang") and entry.get("lang").lower() == lang.lower():
            # default to earliest defined stage in growth schedule
            stages = curriculum.get("growth_schedule", {}).get("stages", []) or []
            if stages:
                return stages[0].get("name")
    return None


def _band_agentic_policy(curriculum: Dict[str, Any], band: str) -> Optional[Any]:
    bands = curriculum.get("difficulty_system", {}).get("bands", {}) or {}
    band_cfg = bands.get(band) or {}
    reasoning_policy = band_cfg.get("reasoning_policy", {}) or {}
    return reasoning_policy.get("agentic")


def run_gate_on_table(
    table: pa.Table,
    curriculum_path: str,
    target_stage: str,
) -> List[Dict[str, Any]]:
    """Run the curriculum gate on an in-memory pyarrow Table.

    Args:
        table: pyarrow.Table containing metadata rows (must include columns:
               'uuid', 'id', 'file_path', 'language', 'band_assignment_band',
               'modality_has_agentic' or 'modality_agentic_density').
        curriculum_path: Path to `curriculum.yaml`.
        target_stage: Stage name (e.g., '1B', '3B') that we're gating for.

    Returns:
        List of rejection dicts with keys: uuid, id, file_path, rejected_reason, rejected_at
    """
    curriculum = _load_curriculum(curriculum_path)
    stage_map = _stage_order_map(curriculum)
    target_order = stage_map.get(target_stage)

    rows = table.to_pylist()
    rejections: List[Dict[str, Any]] = []

    for r in rows:
        uuid = r.get("uuid")
        rec_id = r.get("id")
        file_path = r.get("file_path", "unknown")

        # Language-based gating (Indic earliest stage)
        lang = r.get("language")
        if isinstance(lang, str) and lang.strip():
            lang_norm = lang.strip().lower()
        else:
            lang_norm = None

        if lang_norm:
            earliest = _get_earliest_stage_for_lang(curriculum, lang_norm)
            if earliest and target_order is not None:
                earliest_order = stage_map.get(earliest)
                if earliest_order is not None and target_order < earliest_order:
                    # reject
                    rejections.append(
                        {
                            "uuid": uuid,
                            "id": rec_id,
                            "file_path": file_path,
                            "rejected_reason": "indic_not_allowed_at_stage",
                            "rejected_at": "curriculum_gate",
                        }
                    )
                    # skip other checks for this record
                    continue

        # Agentic-in-band gating
        band = r.get("band_assignment_band") or r.get("band")
        has_agentic = r.get("modality_has_agentic")
        if has_agentic is None:
            # fallback: check density
            has_agentic = bool(r.get("modality_agentic_density"))

        if band and has_agentic:
            policy = _band_agentic_policy(curriculum, band)
            # If policy is 'forbidden' or explicitly disallowing
            if policy in ("forbidden", "exclude", False):
                rejections.append(
                    {
                        "uuid": uuid,
                        "id": rec_id,
                        "file_path": file_path,
                        "rejected_reason": "agentic_not_allowed_in_band_at_stage",
                        "rejected_at": "curriculum_gate",
                    }
                )

    return rejections


def run_gate_on_parquet(
    metadata_parquet_path: str,
    curriculum_path: str,
    target_stage: str,
    output_rejections_parquet: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run the gate on a parquet file and optionally write rejections to disk.
    """
    table = pq.read_table(metadata_parquet_path)
    rejections = run_gate_on_table(table, curriculum_path, target_stage)

    if output_rejections_parquet and rejections:
        # Write as parquet
        rows = [
            {
                "uuid": r["uuid"],
                "id": r["id"],
                "file_path": r["file_path"],
                "rejected_reason": r["rejected_reason"],
                "rejected_at": r["rejected_at"],
            }
            for r in rejections
        ]
        out_table = pa.Table.from_pylist(rows)
        pq.write_table(out_table, output_rejections_parquet)

    return rejections
