"""Tests for modality metric plugin."""

import tempfile
from pathlib import Path

import pytest
import yaml
from curriculum_tags.metrics.modality import ModalityMetric
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


def test_modality_code_detection(temp_config):
    """Test modality detection for code."""
    metric = ModalityMetric(temp_config)

    code_sample = {"text": "def hello():\n    print('world')"}
    result = metric.compute(code_sample)

    assert result["has_code"] is True
    assert result["primary_modality"] == "code"


def test_modality_math_detection(temp_config):
    """Test modality detection for math."""
    metric = ModalityMetric(temp_config)

    math_sample = {"text": "The equation is: ∑ x² ≤ ∫ f(x) dx"}
    result = metric.compute(math_sample)

    assert result["has_math"] is True
    assert result["primary_modality"] == "math"


def test_modality_reasoning_detection(temp_config):
    """Test modality detection for reasoning."""
    metric = ModalityMetric(temp_config)

    reasoning_sample = {
        "text": "Let's think step by step. First, we need to analyze..."
    }
    result = metric.compute(reasoning_sample)

    assert result["has_reasoning"] is True
    assert result["primary_modality"] == "reasoning"


def test_modality_agentic_detection(temp_config):
    """Test modality detection for agentic traces."""
    metric = ModalityMetric(temp_config)

    agentic_sample = {"text": 'Thought: I should use a tool\nAction: search("query")'}
    result = metric.compute(agentic_sample)

    assert result["has_agentic"] is True
    assert result["primary_modality"] == "agentic_traces"


def test_modality_general_text(temp_config):
    """Test modality detection for general text."""
    metric = ModalityMetric(temp_config)

    general_sample = {"text": "This is just a regular sentence about nothing special."}
    result = metric.compute(general_sample)

    assert result["primary_modality"] == "general_text"
    assert not result["has_code"]
    assert not result["has_math"]
    assert not result["has_reasoning"]
    assert not result["has_agentic"]


def test_modality_research_paper_detection(temp_config):
    """Test modality detection for research papers."""
    metric = ModalityMetric(temp_config)

    # Test with Abstract
    assert (
        metric.compute({"text": "Abstract: This paper presents..."})[
            "has_research_paper"
        ]
        is True
    )

    # Test with References
    assert (
        metric.compute({"text": "References: 1. Smith et al."})["has_research_paper"]
        is True
    )

    # Test with arXiv/doi
    assert (
        metric.compute({"text": "See arXiv: 2101.12345 for details."})[
            "has_research_paper"
        ]
        is True
    )
    assert (
        metric.compute({"text": "doi: 10.1145/1234567.1234568"})["has_research_paper"]
        is True
    )

    # Test with et al.
    assert (
        metric.compute({"text": "As shown by Wang et al. (2023)..."})[
            "has_research_paper"
        ]
        is True
    )

    # Test with citations [1, 2]...[3] (Regex requires multiple blocks)
    assert (
        metric.compute({"text": "Recent works [1, 2] have shown results [3]."})[
            "has_research_paper"
        ]
        is True
    )

    # Test primary modality
    result = metric.compute({"text": "Abstract: Deep learning is hard. doi: 123"})
    assert result["has_research_paper"] is True
    assert result["primary_modality"] == "research_papers"


def test_modality_false_positive_prevention(temp_config):
    """Test that refined regexes do not match general text."""
    metric = ModalityMetric(temp_config)

    # Code negatives
    assert not metric.compute({"text": "My biology class is doing well."})["has_code"]
    assert not metric.compute({"text": "The function of the heart."})["has_code"]
    assert not metric.compute({"text": "We import goods."})["has_code"]
    assert not metric.compute({"text": "Where are you from?"})["has_code"]

    # Math negatives
    assert not metric.compute({"text": "It costs $50.00."})["has_math"]
    assert not metric.compute({"text": "The sum of 1 and 2."})["has_math"]
    assert not metric.compute({"text": "Using the alpha channel."})["has_math"]

    # Research negatives
    assert not metric.compute({"text": "Abstract art is fascinating."})[
        "has_research_paper"
    ]
    assert not metric.compute({"text": "I have good references."})["has_research_paper"]

    # Agentic negatives
    assert not metric.compute({"text": "My final thought: this is great."})[
        "has_agentic"
    ]
    assert not metric.compute({"text": "We need to take action:"})["has_agentic"]

    # Reasoning negatives
    assert not metric.compute({"text": "The reasoning: it was necessary."})[
        "has_reasoning"
    ]
    assert not metric.compute({"text": "This requires no explanation."})[
        "has_reasoning"
    ]


def test_modality_chaining_with_previous_tags(temp_config):
    """Test that modality can access previous plugin results."""
    metric = ModalityMetric(temp_config)

    # Simulate document with previous tags
    sample = {
        "text": "def fibonacci(n): return n",
        "curriculum_tags": {"difficulty": {"band": "B2", "score": 0.45}},
    }

    result = metric.compute(sample)

    # Modality should work regardless of previous tags
    assert result["has_code"] is True
    # The sample still has its curriculum_tags untouched
    assert "difficulty" in sample["curriculum_tags"]
