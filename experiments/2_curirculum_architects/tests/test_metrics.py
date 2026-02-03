"""Tests for built-in metric plugins."""

import tempfile
from pathlib import Path

import pytest
import yaml
from curriculum_tags.metrics.difficulty import DifficultyMetric
from curriculum_tags.metrics.modality import ModalityMetric
from curriculum_tags.metrics.readability import ReadabilityMetric
from curriculum_tags.utils.curriculum_loader import CurriculumConfig


@pytest.fixture
def temp_config():
    """Create temporary config."""
    config_data = {
        "version": "0.1",
        "difficulty_system": {
            "bands": {
                "B0": {"name": "Nursery"},
                "B1": {"name": "Primary"},
            }
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        path = Path(f.name)

    config = CurriculumConfig(path)
    yield config
    path.unlink()


def test_difficulty_metric_simple_text(temp_config):
    """Test difficulty metric on simple text."""
    metric = DifficultyMetric(temp_config)

    sample = {"text": "Hello world"}
    result = metric.compute(sample)

    assert "level" in result
    assert "score" in result
    assert "features" in result
    assert result["level"] in ["L0", "L1", "L2", "L3", "L4", "L5"]


def test_difficulty_metric_complex_text(temp_config):
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


def test_difficulty_metric_short_text(temp_config):
    """Test difficulty metric on very short text."""
    metric = DifficultyMetric(temp_config)

    sample = {"text": "Hi"}
    result = metric.compute(sample)

    # Short text defaults to L0
    assert result["level"] == "L0"


def test_modality_metric_code(temp_config):
    """Test modality detection for code."""
    metric = ModalityMetric(temp_config)

    code_sample = {"text": "def hello():\n    print('world')"}
    result = metric.compute(code_sample)

    assert result["has_code"] is True
    assert result["primary_modality"] == "code"


def test_modality_metric_math(temp_config):
    """Test modality detection for math."""
    metric = ModalityMetric(temp_config)

    math_sample = {"text": "The equation is: ∑ x² ≤ ∫ f(x) dx"}
    result = metric.compute(math_sample)

    assert result["has_math"] is True
    assert result["primary_modality"] == "math"


def test_modality_metric_reasoning(temp_config):
    """Test modality detection for reasoning."""
    metric = ModalityMetric(temp_config)

    reasoning_sample = {"text": "Let's think step by step. Therefore, we conclude..."}
    result = metric.compute(reasoning_sample)

    assert result["has_reasoning"] is True
    assert result["primary_modality"] == "reasoning"


def test_modality_metric_general_text(temp_config):
    """Test modality detection for general text."""
    metric = ModalityMetric(temp_config)

    general_sample = {"text": "This is a simple sentence."}
    result = metric.compute(general_sample)

    assert result["has_code"] is False
    assert result["has_math"] is False
    assert result["primary_modality"] == "general_text"


def test_readability_metric(temp_config):
    """Test readability metric."""
    metric = ReadabilityMetric(temp_config)

    sample = {"text": "The quick brown fox jumps over the lazy dog."}
    result = metric.compute(sample)

    assert "flesch_kincaid_grade" in result
    assert "avg_sentence_length" in result
    assert "avg_word_length" in result
    assert result["flesch_kincaid_grade"] >= 0


def test_readability_metric_empty_text(temp_config):
    """Test readability metric on empty text."""
    metric = ReadabilityMetric(temp_config)

    sample = {"text": ""}
    result = metric.compute(sample)

    assert result["flesch_kincaid_grade"] == 0.0
    assert result["avg_sentence_length"] == 0.0


def test_readability_metric_complex_text(temp_config):
    """Test readability metric on complex text."""
    metric = ReadabilityMetric(temp_config)

    complex_text = """
    Notwithstanding the aforementioned considerations regarding epistemological
    frameworks and hermeneutical methodologies, we must acknowledge the
    multifaceted nature of contemporary discourse.
    """

    sample = {"text": complex_text}
    result = metric.compute(sample)

    # Complex text should have higher FK grade
    assert result["flesch_kincaid_grade"] > 10
