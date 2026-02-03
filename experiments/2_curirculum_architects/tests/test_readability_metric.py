"""Tests for readability metric plugin."""

import tempfile
from pathlib import Path

import pytest
import yaml
from curriculum_tags.metrics.readability import ReadabilityMetric
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


def test_readability_simple_text(temp_config):
    """Test readability on simple text."""
    metric = ReadabilityMetric(temp_config)

    sample = {"text": "Hello world. This is a test."}
    result = metric.compute(sample)

    assert "flesch_kincaid_grade" in result
    assert "avg_sentence_length" in result
    assert "avg_word_length" in result
    assert result["flesch_kincaid_grade"] >= 0


def test_readability_complex_text(temp_config):
    """Test readability on complex text."""
    metric = ReadabilityMetric(temp_config)

    complex_text = """
    The implementation of quantum entanglement phenomena requires sophisticated
    mathematical frameworks. These frameworks incorporate Hilbert space
    representations and non-commutative operator algebras. Such abstractions
    adequately describe superposition states in quantum mechanics.
    """

    sample = {"text": complex_text}
    result = metric.compute(sample)

    # Complex text should have higher grade level
    assert result["flesch_kincaid_grade"] > 5


def test_readability_empty_text(temp_config):
    """Test readability on empty text."""
    metric = ReadabilityMetric(temp_config)

    sample = {"text": ""}
    result = metric.compute(sample)

    assert result["flesch_kincaid_grade"] == 0.0
    assert result["avg_sentence_length"] == 0.0


def test_readability_single_word(temp_config):
    """Test readability on single word."""
    metric = ReadabilityMetric(temp_config)

    sample = {"text": "Hello"}
    result = metric.compute(sample)

    # Should handle edge case gracefully
    assert isinstance(result["flesch_kincaid_grade"], float)
