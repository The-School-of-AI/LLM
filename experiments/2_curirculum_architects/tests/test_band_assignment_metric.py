"""Unit tests for BandAssignmentMetric."""

import pytest
from curriculum_tags.metrics.band_assignment import BandAssignmentMetric


@pytest.fixture
def metric():
    """Create metric instance."""
    return BandAssignmentMetric({})


def test_agentic_traces(metric):
    """Test agentic trace override."""
    sample = {
        "curriculum_tags": {
            "modality": {"has_agentic": True}
        }
    }
    result = metric.compute(sample)
    assert result["band"] == "B5"
    assert "agentic" in result["reason"].lower()


def test_research_paper(metric):
    """Test research paper logic."""
    # Standard research paper
    sample_b4 = {
        "curriculum_tags": {
            "modality": {"has_research_paper": True},
            "difficulty": {"score": 0.5},
            "readability": {"flesch_kincaid_grade": 12.0}
        }
    }
    assert metric.compute(sample_b4)["band"] == "B4"

    # Complex research paper
    sample_b5 = {
        "curriculum_tags": {
            "modality": {"has_research_paper": True},
            "difficulty": {"score": 0.9}, # Very high
            "readability": {"flesch_kincaid_grade": 18.0}
        }
    }
    assert metric.compute(sample_b5)["band"] == "B5"


def test_code_math_logic(metric):
    """Test code and math band progression."""
    base_sample = {
        "curriculum_tags": {
            "modality": {"has_code": True},
            "difficulty": {"score": 0.0}
        }
    }
    
    # B2
    base_sample["curriculum_tags"]["difficulty"]["score"] = 0.3
    assert metric.compute(base_sample)["band"] == "B2"
    
    # B3
    base_sample["curriculum_tags"]["difficulty"]["score"] = 0.5
    assert metric.compute(base_sample)["band"] == "B3"
    
    # B4
    base_sample["curriculum_tags"]["difficulty"]["score"] = 0.7
    assert metric.compute(base_sample)["band"] == "B4"
    
    # B5
    base_sample["curriculum_tags"]["difficulty"]["score"] = 0.9
    assert metric.compute(base_sample)["band"] == "B5"


def test_general_text_bands(metric):
    """Test mixed logic with Difficulty Mapping + Secondary Thresholds."""
    # L0 -> B0
    sample = {
        "curriculum_tags": {
            "difficulty": {"score": 0.1, "level": "L0"},
            "readability": {"flesch_kincaid_grade": 2.0},
            "entropy": {"score": 3.0},
            "diversity": {"rare_ratio": 0.05}
        }
    }
    assert metric.compute(sample)["band"] == "B0"
    
    # L2 -> B1 (Provided secondary stats pass)
    sample["curriculum_tags"]["difficulty"] = {"score": 0.4, "level": "L2"}
    sample["curriculum_tags"]["readability"]["flesch_kincaid_grade"] = 10.0 # Pass B1
    sample["curriculum_tags"]["entropy"]["score"] = 4.0 # Pass B1
    sample["curriculum_tags"]["diversity"]["rare_ratio"] = 0.20 # Pass B1
    assert metric.compute(sample)["band"] == "B1"
    
    # L4 -> B4 (Provided stats pass)
    sample["curriculum_tags"]["difficulty"] = {"score": 0.8, "level": "L4"}
    sample["curriculum_tags"]["readability"]["flesch_kincaid_grade"] = 15.0 # Pass B4
    sample["curriculum_tags"]["entropy"]["score"] = 5.2 # Pass B4
    sample["curriculum_tags"]["diversity"]["rare_ratio"] = 0.28 # Pass B4
    assert metric.compute(sample)["band"] == "B4"

def test_entropy_diversity_filtering(metric):
    """Test that low entropy/diversity limits the mapped band."""
    # L4 -> should be B4...
    sample = {
        "curriculum_tags": {
            "difficulty": {"score": 0.8, "level": "L4"},
            "readability": {"flesch_kincaid_grade": 15.0},
            # ...BUT Low Entropy/Diversity (only B0 level)
            "entropy": {"score": 2.0}, 
            "diversity": {"rare_ratio": 0.05}
        }
    }
    # Should fall back recursively L4(B4)->fail -> B3->fail -> ... -> B0
    assert metric.compute(sample)["band"] == "B0"
