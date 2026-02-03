"""Tests for CSV output (flat main + rejected log)."""

import csv
import tempfile
from pathlib import Path

from curriculum_tags.output import (
    MAIN_CSV_COLUMNS,
    REJECTED_CSV_COLUMNS,
    SCHEMA_VERSION,
    RejectionReason,
    is_rejected,
    write_csv_output,
)
from curriculum_tags.output.csv_writer import build_rejected_row, compute_checksum, flatten_record


def test_compute_checksum():
    assert compute_checksum("hello") == compute_checksum("hello")
    assert compute_checksum("hello") != compute_checksum("world")
    assert compute_checksum("") != ""
    assert compute_checksum("  hello  ") == compute_checksum("hello")


def test_is_rejected():
    assert is_rejected({"curriculum_tags": {"error": "fail"}}) is True
    assert is_rejected({"curriculum_tags": {"band_assignment": {}}}) is True
    assert is_rejected({"curriculum_tags": {"band_assignment": {"band": ""}}}) is True
    assert is_rejected({"curriculum_tags": {"band_assignment": {"band": "B1"}}}) is False
    assert is_rejected({"curriculum_tags": {}}) is True


def test_flatten_record():
    record = {
        "id": "sample_0",
        "text": "Hello world.",
        "curriculum_tags": {
            "version": "v1",
            "band_assignment": {"band": "B1", "reason": "constraint_match"},
            "difficulty": {"level": "L1", "score": 0.25},
            "readability": {"flesch_kincaid_grade": 5.2},
            "modality": {"primary_modality": "text"},
            "tokenizer_difficulty": {"level": "T1"},
            "entropy": {"score": 3.8},
            "structural_density": {"structural_density": 0.02},
            "cot_scanner": {"has_cot": False, "has_agentic": False},
        },
    }
    row = flatten_record(record, "path/to/file.parquet")
    assert row["id"] == "sample_0"
    assert row["file_path"] == "path/to/file.parquet"
    assert row["band"] == "B1"
    assert row["band_reason"] == "constraint_match"
    assert row["difficulty_level"] == "L1"
    assert row["difficulty_score"] == "0.25"
    assert row["readability_fk_grade"] == "5.2"
    assert row["primary_modality"] == "text"
    assert row["tokenizer_level"] == "T1"
    assert row["entropy_score"] == "3.8"
    assert row["structural_density"] == "0.02"
    assert row["has_cot"] == "false"
    assert row["has_agentic"] == "false"
    assert row["schema_version"] == SCHEMA_VERSION
    assert len(row["uuid"]) == 36
    assert len(row["checksum"]) == 64  # SHA-256 hex
    assert row["minhash"] == ""
    assert row["optional_1"] == ""


def test_build_rejected_row():
    record = {"id": "sample_42", "text": "x"}
    row = build_rejected_row(
        record,
        "s3://bucket/part.parquet",
        RejectionReason.METRIC_FAILED.value,
        "tokenizer not found",
    )
    assert row["id"] == "sample_42"
    assert row["file_path"] == "s3://bucket/part.parquet"
    assert row["reason"] == RejectionReason.METRIC_FAILED.value
    assert row["details"] == "tokenizer not found"
    assert row["schema_version"] == SCHEMA_VERSION
    assert len(row["uuid"]) == 36


def test_write_csv_output():
    tagged_ok = {
        "id": "ok_1",
        "text": "Hello.",
        "curriculum_tags": {
            "version": "v1",
            "band_assignment": {"band": "B0", "reason": "ok"},
            "difficulty": {"level": "L0", "score": 0.1},
            "readability": {},
            "modality": {"primary_modality": "text"},
            "tokenizer_difficulty": {},
            "entropy": {},
            "structural_density": {},
            "cot_scanner": {"has_cot": False, "has_agentic": False},
        },
    }
    tagged_rejected = {
        "id": "bad_1",
        "text": "x",
        "curriculum_tags": {"version": "v1", "error": "metric failed"},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        main_path = Path(tmpdir) / "main.csv"
        stats = write_csv_output(
            [tagged_ok, tagged_rejected],
            file_path="input.parquet",
            output_csv_path=main_path,
        )
        assert stats["main_row_count"] == 1
        assert stats["rejected_row_count"] == 1
        assert Path(stats["main_csv_path"]).exists()
        assert Path(stats["rejected_csv_path"]).exists()

        with open(main_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["id"] == "ok_1"
        assert rows[0]["band"] == "B0"
        assert rows[0]["file_path"] == "input.parquet"
        assert list(rows[0].keys()) == MAIN_CSV_COLUMNS

        rejected_path = Path(tmpdir) / "main_rejected.csv"
        with open(rejected_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rej_rows = list(reader)
        assert len(rej_rows) == 1
        assert rej_rows[0]["id"] == "bad_1"
        assert rej_rows[0]["reason"] == RejectionReason.METRIC_FAILED.value
        assert list(rej_rows[0].keys()) == REJECTED_CSV_COLUMNS


def test_write_csv_output_custom_rejected_path():
    tagged_ok = {
        "id": "ok",
        "text": "Hi",
        "curriculum_tags": {
            "version": "v1",
            "band_assignment": {"band": "B1", "reason": ""},
        },
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        main_path = Path(tmpdir) / "out.csv"
        rejected_path = Path(tmpdir) / "rejected_only.csv"
        write_csv_output(
            [tagged_ok],
            file_path="x.parquet",
            output_csv_path=main_path,
            rejected_csv_path=rejected_path,
        )
        assert main_path.exists()
        assert rejected_path.exists()
        with open(rejected_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 0
