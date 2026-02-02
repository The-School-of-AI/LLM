"""Unit tests for BandAssignmentMetric."""

import pytest
from curriculum_tags.metrics.band_assignment import BandAssignmentMetric


from pathlib import Path
from types import SimpleNamespace

@pytest.fixture
def metric():
    """Create metric instance with access to real band_assignment.yaml."""
    # We point to a fake curriculum.yaml in the project root so it finds band_assignment.yaml next to it
    root_dir = Path(__file__).parent.parent
    fake_config = SimpleNamespace(path=str(root_dir / "curriculum.yaml"))
    
    return BandAssignmentMetric(fake_config)


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
    """Test mixed logic with Difficulty Mapping + Secondary Ranges."""
    # L0 -> Fits B0 criteria
    sample = {
        "curriculum_tags": {
            "difficulty": {"score": 0.1, "level": "L0"},
            "readability": {"flesch_kincaid_grade": 2.0},
            "entropy": {"score": 3.0},
            "diversity": {"rare_ratio": 0.05}
        }
    }
    assert metric.compute(sample)["band"] == "B0"
    
    # L2 -> Fits B1 criteria
    sample["curriculum_tags"]["difficulty"] = {"score": 0.4, "level": "L2"}
    sample["curriculum_tags"]["readability"]["flesch_kincaid_grade"] = 9.0 # Fits B1 (4-10) and B2 (8-14)
    sample["curriculum_tags"]["entropy"]["score"] = 4.0 
    sample["curriculum_tags"]["diversity"]["rare_ratio"] = 0.20 
    
    # With L2, Diff 0.4, FK 9.0:
    # B1 Constraints: L2 ok, Read 4-10 ok, Diff 0.2-0.5 ok, Ent 3.5-5.5 ok, Div 0.10-0.25 ok. -> MATCH
    # B2 Constraints: L2 ok, Read 8-14 ok, Diff 0.4-0.7 ok, Ent 4.0-6.0 ok, Div 0.15-0.35 ok. -> MATCH
    # Overlap Policy: Highest -> B2
    assert metric.compute(sample)["band"] == "B2"

def test_hard_constraints_filtering(metric):
    """Test that strict ranges filter out bands."""
    # L2 (Allowed in B1, B2)
    # BUT FK Grade 20.0 (Too high for B1 [4-10] or B2 [8-14])
    sample = {
        "curriculum_tags": {
            "difficulty": {"score": 0.4, "level": "L2"},
            "readability": {"flesch_kincaid_grade": 20.0},
            "entropy": {"score": 4.0}, 
            "diversity": {"rare_ratio": 0.20}
        }
    }
    # Should fail B1 and B2 readability checks.
    # Fallback -> B0
    assert metric.compute(sample)["band"] == "B0"
    
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
def test_cot_scanner_integration(metric):
    """Test COT and Agentic scanner integration."""
    
    # Test COT Floor: Simple text (L0/B0) + COT -> Should be B3
    sample_cot = {
        "curriculum_tags": {
            "difficulty": {"score": 0.1, "level": "L0"},
            "readability": {"flesch_kincaid_grade": 2.0},
            "cot_scanner": {"has_cot": True, "has_agentic": False}
        }
    }
    # Even though difficulty is L0, COT implies reasoning -> B3
    assert metric.compute(sample_cot)["band"] == "B3"
    
    # Test Agentic Override: Simple text + Agentic -> Should be B5
    sample_agentic = {
        "curriculum_tags": {
            "difficulty": {"score": 0.1, "level": "L0"},
            "readability": {"flesch_kincaid_grade": 2.0},
            "cot_scanner": {"has_cot": False, "has_agentic": True}
        }
    }
    assert metric.compute(sample_agentic)["band"] == "B5"
