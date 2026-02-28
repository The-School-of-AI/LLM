from curriculum_tags.metrics.band_assignment import BandAssignmentMetric


def test_fallback():
    class Config:
        path = "d:/ERA_CAPSTONE/LLM/experiments/2_curirculum_architects/curriculum.yaml"

    metric = BandAssignmentMetric(Config())

    # Sample WITHOUT "diversity" top-level key, but WITH "difficulty.features.rare_ratio"
    sample = {
        "text": "This is a test sample.",
        "curriculum_tags": {
            "difficulty": {
                "level": "L2",
                "score": 0.45,
                "features": {"rare_ratio": 0.25},  # This should be picked up
            },
            "readability": {"flesch_kincaid_grade": 9.0},
            "entropy": {"score": 3.0},
        },
    }

    # Logic:
    # B1 allows L2, Read 2-14, Diff 0.1-0.6, Entropy 1.5-6.0, Diversity 0.0-0.4.
    # With rare_ratio 0.25, it should pass B1.

    result = metric.compute(sample)
    print(f"Assigned Band: {result['band']}")
    print(f"Reason: {result['reason']}")

    assert result["band"] in ["B1", "B2"]  # B2 might also match if constraints allow
    assert "B1" in result["reason"] or "B2" in result["reason"]
    print("Fallback Check PASSED!")


if __name__ == "__main__":
    test_fallback()
