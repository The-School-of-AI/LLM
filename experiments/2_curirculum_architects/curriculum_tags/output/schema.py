"""Output schema constants for main CSV and rejected log."""

from enum import Enum
from typing import List

SCHEMA_VERSION = "v1"

# Main CSV columns (order matters for header and row writing)
MAIN_CSV_COLUMNS: List[str] = [
    "uuid",
    "id",
    "file_path",
    "band",
    "band_reason",
    "difficulty_level",
    "difficulty_score",
    "readability_fk_grade",
    "primary_modality",
    "tokenizer_level",
    "entropy_score",
    "structural_density",
    "has_cot",
    "has_agentic",
    "checksum",
    "minhash",
    "optional_1",
    "optional_2",
    "optional_3",
    "schema_version",
]

# Rejected log CSV columns
REJECTED_CSV_COLUMNS: List[str] = [
    "uuid",
    "id",
    "file_path",
    "reason",
    "details",
    "schema_version",
]


class RejectionReason(str, Enum):
    """Standard reasons for rejecting a row (no band/tags written to main output)."""

    PARSE_ERROR = "parse_error"
    EMPTY_TEXT = "empty_text"
    METRIC_FAILED = "metric_failed"
    BAND_ASSIGNMENT_FAILED = "band_assignment_failed"
