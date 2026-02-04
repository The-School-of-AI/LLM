"""Tests for curriculum_gate postprocessing."""

import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from curriculum_extractor.postprocessing.curriculum_gate import (
    run_gate_on_table,
)
from curriculum_extractor.utils.curriculum_loader import CurriculumConfig


def make_temp_curriculum_yaml(tmpdir: Path) -> str:
    content = {
        "growth_schedule": {"stages": [{"name": "1B", "order": 1}, {"name": "3B", "order": 2}]},
        "language_and_context": {
            "language_policy": {
                "primary_languages": [{"lang": "en"}],
                "secondary_languages": [{"lang": "hi", "earliest_stage": "3B"}],
            }
        },
        "difficulty_system": {
            "bands": {
                "B0": {"reasoning_policy": {"agentic": "forbidden"}},
                "B3": {"reasoning_policy": {"agentic": "toy_only"}},
            }
        },
    }

    path = tmpdir / "curriculum_tmp.yaml"
    import yaml

    with open(path, "w") as f:
        yaml.safe_dump(content, f)
    return str(path)


def test_indic_not_allowed(tmp_path):
    # Create curriculum that allows hi starting at 3B
    cur_path = make_temp_curriculum_yaml(tmp_path)

    # Build a table with a Hindi record but target_stage=1B (should reject)
    rows = [
        {
            "uuid": "u1",
            "id": "1",
            "file_path": "f1",
            "language": "hi",
            "band_assignment_band": "B0",
            "modality_has_agentic": False,
        }
    ]
    table = pa.Table.from_pylist(rows)

    rejections = run_gate_on_table(table, cur_path, target_stage="1B")
    assert len(rejections) == 1
    assert rejections[0]["rejected_reason"] == "indic_not_allowed_at_stage"


def test_agentic_not_allowed_in_band(tmp_path):
    cur_path = make_temp_curriculum_yaml(tmp_path)

    # Create a record assigned to B0 which forbids agentic
    rows = [
        {
            "uuid": "u2",
            "id": "2",
            "file_path": "f2",
            "language": "en",
            "band_assignment_band": "B0",
            "modality_has_agentic": True,
        }
    ]
    table = pa.Table.from_pylist(rows)

    rejections = run_gate_on_table(table, cur_path, target_stage="3B")
    assert len(rejections) == 1
    assert rejections[0]["rejected_reason"] == "agentic_not_allowed_in_band_at_stage"
