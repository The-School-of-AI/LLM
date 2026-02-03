"""CSV writer: flatten curriculum_tags to flat columns and write main + rejected CSVs."""

import csv
import hashlib
import uuid as uuid_module
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema import MAIN_CSV_COLUMNS, REJECTED_CSV_COLUMNS, SCHEMA_VERSION, RejectionReason


def compute_checksum(text: str) -> str:
    """SHA-256 of normalized text (strip, UTF-8). Empty string -> hash of empty."""
    normalized = (text or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_rejected(record: Dict[str, Any]) -> bool:
    """True if this record has an error or no band (should go to rejected log only)."""
    tags = record.get("curriculum_tags") or {}
    if tags.get("error") is not None:
        return True
    band_assignment = tags.get("band_assignment") or {}
    band = band_assignment.get("band")
    if band is None or band == "":
        return True
    return False


def _get_rejection_reason_and_details(record: Dict[str, Any]) -> tuple[str, str]:
    """Infer reason and details from curriculum_tags.error or band_assignment."""
    tags = record.get("curriculum_tags") or {}
    err = tags.get("error")
    if err is not None:
        return RejectionReason.METRIC_FAILED.value, str(err)
    band_assignment = tags.get("band_assignment") or {}
    if band_assignment.get("band") is None:
        return RejectionReason.BAND_ASSIGNMENT_FAILED.value, "band missing"
    return RejectionReason.METRIC_FAILED.value, "unknown"


def flatten_record(record: Dict[str, Any], file_path: str) -> Dict[str, str]:
    """Build one flat main-CSV row from a tagged record. Assumes record is not rejected."""
    tags = record.get("curriculum_tags") or {}
    text = record.get("text") or ""

    band_assignment = tags.get("band_assignment") or {}
    difficulty = tags.get("difficulty") or {}
    readability = tags.get("readability") or {}
    modality = tags.get("modality") or {}
    tokenizer_difficulty = tags.get("tokenizer_difficulty") or {}
    entropy = tags.get("entropy") or {}
    structural_density = tags.get("structural_density") or {}
    cot_scanner = tags.get("cot_scanner") or {}

    def b(v: Any) -> str:
        if v is True:
            return "true"
        if v is False:
            return "false"
        return str(v) if v != "" and v is not None else ""

    return {
        "uuid": str(uuid_module.uuid4()),
        "id": str(record.get("id", "")),
        "file_path": file_path,
        "band": str(band_assignment.get("band", "")),
        "band_reason": str(band_assignment.get("reason", "")),
        "difficulty_level": str(difficulty.get("level", "")),
        "difficulty_score": str(difficulty.get("score", "")),
        "readability_fk_grade": str(readability.get("flesch_kincaid_grade", "")),
        "primary_modality": str(modality.get("primary_modality", "")),
        "tokenizer_level": str(tokenizer_difficulty.get("level", "")),
        "entropy_score": str(entropy.get("score", "")),
        "structural_density": str(structural_density.get("structural_density", "")),
        "has_cot": b(cot_scanner.get("has_cot")),
        "has_agentic": b(cot_scanner.get("has_agentic")),
        "checksum": compute_checksum(text),
        "minhash": "",
        "optional_1": "",
        "optional_2": "",
        "optional_3": "",
        "schema_version": SCHEMA_VERSION,
    }


def build_rejected_row(
    record: Dict[str, Any],
    file_path: str,
    reason: str,
    details: str,
) -> Dict[str, str]:
    """Build one rejected-log row."""
    return {
        "uuid": str(uuid_module.uuid4()),
        "id": str(record.get("id", "")),
        "file_path": file_path,
        "reason": reason,
        "details": details,
        "schema_version": SCHEMA_VERSION,
    }


def write_csv_output(
    tagged_records: List[Dict[str, Any]],
    file_path: str,
    output_csv_path: str | Path,
    rejected_csv_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Write main CSV and rejected CSV from tagged records.

    Args:
        tagged_records: List of records with curriculum_tags (and id, text).
        file_path: Source file path to put in file_path column (same for all rows in this batch).
        output_csv_path: Path for main CSV.
        rejected_csv_path: Path for rejected log; if None, derived from output_csv_path.

    Returns:
        Dict with keys: main_csv_path, rejected_csv_path, main_row_count, rejected_row_count.
    """
    output_csv_path = Path(output_csv_path)
    if rejected_csv_path is None:
        stem = output_csv_path.stem
        parent = output_csv_path.parent
        rejected_csv_path = parent / f"{stem}_rejected.csv"
    else:
        rejected_csv_path = Path(rejected_csv_path)

    main_rows: List[Dict[str, str]] = []
    rejected_rows: List[Dict[str, str]] = []

    for record in tagged_records:
        if is_rejected(record):
            reason, details = _get_rejection_reason_and_details(record)
            rejected_rows.append(build_rejected_row(record, file_path, reason, details))
        else:
            main_rows.append(flatten_record(record, file_path))

    def write_csv(path: Path, columns: List[str], rows: List[Dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_csv_path, MAIN_CSV_COLUMNS, main_rows)
    write_csv(rejected_csv_path, REJECTED_CSV_COLUMNS, rejected_rows)

    return {
        "main_csv_path": str(output_csv_path),
        "rejected_csv_path": str(rejected_csv_path),
        "main_row_count": len(main_rows),
        "rejected_row_count": len(rejected_rows),
    }
