"""Tests for difficulty metric plugin."""

import tempfile
from pathlib import Path

import pytest
import yaml

from curriculum_tags.metrics.difficulty import DifficultyMetric
from curriculum_tags.utils.curriculum_loader import CurriculumConfig


@pytest.fixture
def temp_config():
    """Create temporary config."""
    config_data = {
        "version": "0.1",
        "difficulty": {
            "bands": {
                "B0": 0.15,
                "B1": 0.30,
                "B2": 0.50,
                "B3": 0.70,
                "B4": 0.85,
                "B5": 1.00,
            }
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        path = Path(f.name)

    config = CurriculumConfig(path)
    yield config
    path.unlink()


def test_difficulty_simple_text(temp_config):
    """Test difficulty metric on simple text."""
    metric = DifficultyMetric(temp_config)

    sample = {"text": "Hello world"}
    result = metric.compute(sample)

    assert "band" in result
    assert "score" in result
    assert "features" in result
    assert result["band"] in ["B0", "B1", "B2", "B3", "B4", "B5"]


def test_difficulty_complex_text(temp_config):
    """Test difficulty metric on complex text."""
    metric = DifficultyMetric(temp_config)

    complex_text = """
    The implementation of quantum entanglement phenomena requires sophisticated
    mathematical frameworks incorporating Hilbert space representations and
    non-commutative operator algebras to adequately describe superposition states.
    """

    sample = {"text": complex_text}
    result = metric.compute(sample)

    # Complex text should score higher
    assert result["score"] > 0.3


def test_difficulty_short_text(temp_config):
    """Test difficulty metric on very short text."""
    metric = DifficultyMetric(temp_config)

    sample = {"text": "Hi"}
    result = metric.compute(sample)

    # Short text defaults to B0
    assert result["band"] == "B0"


def test_difficulty_custom_bands(temp_config):
    """Test difficulty metric with custom curriculum bands."""
    # Modify config to have different thresholds
    config_data = {
        "version": "0.1",
        "difficulty": {
            "bands": {
                "B0": 0.10,  # Lower threshold
                "B1": 0.20,
                "B2": 1.00,
            }
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        path = Path(f.name)

    config = CurriculumConfig(path)
    metric = DifficultyMetric(config)

    sample = {"text": "Hello world this is a test"}
    result = metric.compute(sample)

    assert result["band"] in ["B0", "B1", "B2"]
    path.unlink()
