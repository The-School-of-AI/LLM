"""Tests for metadata sampling logic in calculate_proportions."""

import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.calculate_proportions import sample_metadata


@pytest.fixture
def mock_parquet_file():
    """Create a temporary parquet file context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_metadata.parquet"


def create_table(data, path):
    """Helper to write data to parquet."""
    table = pa.Table.from_pylist(data)
    pq.write_table(table, path)


def test_legacy_difficulty_tag(mock_parquet_file):
    """Test sampling works with legacy difficulty tags."""
    data = [
        {"curriculum_tags": {"difficulty": {"band": "B0"}}},
        {"curriculum_tags": {"difficulty": {"band": "B0"}}},
        {"curriculum_tags": {"difficulty": {"band": "B1"}}},
    ]
    create_table(data, mock_parquet_file)

    dist = sample_metadata(str(mock_parquet_file), sample_rate=1.0)

    assert dist["B0"] == 2 / 3
    assert dist["B1"] == 1 / 3


def test_new_band_assignment_tag(mock_parquet_file):
    """Test sampling uses the new band_assignment tag."""
    data = [
        {"curriculum_tags": {"band_assignment": {"band": "B4"}}},
        {"curriculum_tags": {"band_assignment": {"band": "B5"}}},
    ]
    create_table(data, mock_parquet_file)

    dist = sample_metadata(str(mock_parquet_file), sample_rate=1.0)

    assert dist["B4"] == 0.5
    assert dist["B5"] == 0.5


def test_priority_logic(mock_parquet_file):
    """Test that band_assignment overrides difficulty."""
    data = [
        {
            "curriculum_tags": {
                # Legacy says B0, New says B5
                "difficulty": {"band": "B0"},
                "band_assignment": {"band": "B5"},
            }
        }
    ]
    create_table(data, mock_parquet_file)

    dist = sample_metadata(str(mock_parquet_file), sample_rate=1.0)

    # Should follow the new tag
    assert dist["B5"] == 1.0
    assert "B0" not in dist


def test_missing_tags_error(mock_parquet_file):
    """Test error raised when no valid tags present."""
    data = [{"curriculum_tags": {"other_metric": {"score": 1}}}]
    create_table(data, mock_parquet_file)

    with pytest.raises(ValueError, match="No valid band assignment or difficulty tags"):
        sample_metadata(str(mock_parquet_file), sample_rate=1.0)
