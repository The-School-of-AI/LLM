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
    config_data = {"version": "0.1"}
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

    assert "level" in result
    assert "score" in result
    assert "features" in result
    assert result["level"] in ["L0", "L1", "L2", "L3", "L4", "L5"]


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

    # Short text defaults to L0
    assert result["level"] == "L0"


def test_difficulty_custom_levels(temp_config):
    """Test difficulty metric with custom curriculum levels."""
    # Modify config to have different thresholds
    metric = DifficultyMetric(temp_config)
    metric.levels = {
        "L0": 0.10,
        "L1": 0.20,
        "L2": 1.00,
    }

    sample = {"text": "Hello world this is a test"}
    result = metric.compute(sample)

    assert result["level"] in ["L0", "L1", "L2"]
