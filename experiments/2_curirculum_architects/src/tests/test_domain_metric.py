import pytest
from curriculum_tags.metrics.domain import DomainMetric


class MockConfig:
    pass


@pytest.fixture
def metric():
    return DomainMetric(MockConfig())


def test_domain_from_modality_agentic(metric):
    sample = {
        "text": "Some text",
        "curriculum_tags": {
            "modality": {"primary_modality": "agentic_traces", "has_agentic": True}
        },
    }
    result = metric.compute(sample)
    assert result["primary_domain"] == "planning_reasoning_curated"
    assert result["confidence"] == 1.0
    assert result["reason"] == "modality_signal"


def test_domain_from_modality_code(metric):
    sample = {
        "text": "def foo(): pass",
        "curriculum_tags": {"modality": {"primary_modality": "code", "has_code": True}},
    }
    result = metric.compute(sample)
    assert result["primary_domain"] == "code_repos"
    assert result["confidence"] == 0.95


def test_domain_from_modality_math(metric):
    sample = {
        "text": "1 + 1 = 2",
        "curriculum_tags": {"modality": {"primary_modality": "math", "has_math": True}},
    }
    result = metric.compute(sample)
    assert result["primary_domain"] == "math_science"
    assert result["confidence"] == 0.9


def test_domain_heuristic_dialogue(metric):
    # No modality, fallback to text regex
    sample = {
        "text": "User: Hello there.\nAssistant: Hi! How can I help?",
        "curriculum_tags": {"modality": {"primary_modality": "general_text"}},
    }
    result = metric.compute(sample)
    assert result["primary_domain"] == "dialogue_chat"
    assert result["confidence"] == 0.85


def test_domain_heuristic_encyclopedic(metric):
    sample = {
        "text": "Python is a programming language. [1] It was created by Guido van Rossum.",
        "curriculum_tags": {"modality": {"primary_modality": "general_text"}},
    }
    result = metric.compute(sample)
    assert result["primary_domain"] == "encyclopedic"
    assert result["confidence"] == 0.8


def test_domain_heuristic_technical(metric):
    sample = {
        "text": "The API requires the following arguments: x, y.",
        "curriculum_tags": {"modality": {"primary_modality": "general_text"}},
    }
    result = metric.compute(sample)
    assert result["primary_domain"] == "technical_docs"
    assert result["confidence"] == 0.6


def test_domain_default(metric):
    sample = {
        "text": "Just some random blog post text.",
        "curriculum_tags": {"modality": {"primary_modality": "general_text"}},
    }
    result = metric.compute(sample)
    assert result["primary_domain"] == "general_web_clean"
    assert result["reason"] == "default"
