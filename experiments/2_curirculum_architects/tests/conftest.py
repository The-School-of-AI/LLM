"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_curriculum_config():
    """Create a sample curriculum config file."""
    config = {
        "version": "0.2",
        "difficulty_bands": {
            "bands": [
                {"id": "B0", "name": "Nursery", "description": "Basic content"},
                {"id": "B1", "name": "Elementary", "description": "Elementary content"},
                {
                    "id": "B2",
                    "name": "Middle School",
                    "description": "Middle school content",
                },
                {
                    "id": "B3",
                    "name": "High School",
                    "description": "High school content",
                },
                {
                    "id": "B4",
                    "name": "Undergraduate",
                    "description": "Undergraduate content",
                },
                {"id": "B5", "name": "Graduate", "description": "Graduate content"},
            ]
        },
        "modalities": {
            "general_text": {"weight": 0.3},
            "code": {"weight": 0.3},
            "math": {"weight": 0.2},
            "science": {"weight": 0.2},
        },
        "metrics_config": {
            "enabled_metrics": [
                "difficulty",
                "readability",
                "modality",
                "band_assignment",
            ]
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        path = Path(f.name)

    yield path
    path.unlink()
