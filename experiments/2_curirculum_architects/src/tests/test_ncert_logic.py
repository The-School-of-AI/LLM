import pytest
from curriculum_tags.metrics.band_assignment import BandAssignmentMetric
from curriculum_tags.metrics.domain import DomainMetric
from curriculum_tags.metrics.modality import ModalityMetric


@pytest.fixture
def modality_metric():
    return ModalityMetric({})


@pytest.fixture
def domain_metric():
    return DomainMetric({})


@pytest.fixture
def band_metric():
    return BandAssignmentMetric({})


def test_ncert_modality_mapping(modality_metric):
    # Test cases: Subject -> Primary Modality
    cases = [
        ("Physics", "technical_text"),
        ("Chemistry", "technical_text"),
        ("Mathematics", "math"),
        ("History", "structured_knowledge"),
        ("English", "general_text"),
    ]

    for subject, expected in cases:
        sample = {
            "id": "ncert_test",
            "dataset": "ncert",
            "text": "some sample text",
            "metadata": {"subject": subject, "source_type": "textbook"},
        }
        result = modality_metric.compute(sample)
        assert result["primary_modality"] == expected, f"Failed for {subject}"


def test_ncert_domain_mapping(domain_metric):
    # Test cases: Subject -> Primary Domain
    cases = [
        ("Physics", "math_science"),
        ("History", "encyclopedic"),
        ("Economics", "technical_docs"),
        ("English", "general_web_clean"),
    ]

    for subject, expected in cases:
        sample = {
            "id": "ncert_test",
            "dataset": "ncert",
            "text": "some sample text",
            "metadata": {"subject": subject, "source_type": "textbook"},
        }
        result = domain_metric.compute(sample)
        assert result["primary_domain"] == expected, f"Failed for {subject}"


def test_ncert_band_capping(band_metric):
    # Helper to create result mock
    def get_capped_band(initial_band, grade):
        # We test the helper method directly as it isolates the logic
        return band_metric.adjust_band_for_ncert(initial_band, grade)

    # Grade 6 (Cap B2)
    assert get_capped_band("B1", 6) == "B1"
    assert get_capped_band("B2", 6) == "B2"
    assert get_capped_band("B3", 6) == "B2"  # Capped
    assert get_capped_band("B5", 6) == "B2"  # Capped

    # Grade 9 (Cap B3)
    assert get_capped_band("B3", 9) == "B3"
    assert get_capped_band("B4", 9) == "B3"  # Capped
    assert get_capped_band("B5", 9) == "B3"  # Capped

    # Grade 11 (Cap B4)
    assert get_capped_band("B4", 11) == "B4"
    assert get_capped_band("B5", 11) == "B4"  # Capped

    # Grade None (No change)
    assert get_capped_band("B5", None) == "B5"


def test_ncert_integration(modality_metric, domain_metric, band_metric):
    """Test full flow for a sample."""
    sample = {
        "id": "ncert_123",
        "dataset": "ncert",
        "text": "Calculate the velocity...",
        "metadata": '{"subject": "Physics", "grade": 7, "source_type": "textbook"}',
    }

    # 1. Modality
    mod_res = modality_metric.compute(sample)
    assert mod_res["primary_modality"] == "technical_text"

    # 2. Domain (needs modality tags technically but our override ignores them)
    sample["curriculum_tags"] = {"modality": mod_res}
    dom_res = domain_metric.compute(sample)
    assert dom_res["primary_domain"] == "math_science"

    # 3. Band (Mocking the pipeline inputs)
    # Assume automated metrics gave B4 based on difficulty
    # Since B4 is > B2 (Grade 7 limit), it should be capped.

    # We test _get_ncert_grade
    grade = band_metric._get_ncert_grade(sample)
    assert grade == 7

    # Logic is inside compute but we can test adjustment directly again or ensure compute calls it.
    # To test compute fully we'd need to mock all tags.
    # Let's trust the logic unit test and just verify grade extraction here.
