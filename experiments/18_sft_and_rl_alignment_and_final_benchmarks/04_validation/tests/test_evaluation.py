"""
Unit Tests — SFT Validation Framework
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from instruction_following import InstructionFollowingScorer
from hallucination_detector import HallucinationDetector
from metrics import MetricsCollector


# ===========================================================================
# Instruction-Following Tests
# ===========================================================================

class TestInstructionFollowingScorer:

    def setup_method(self):
        self.scorer = InstructionFollowingScorer()

    def test_numbered_list_detected(self):
        output = "1. First item\n2. Second item\n3. Third item"
        rubric = {"must_have": ["numbered list"], "must_not_have": []}
        result = self.scorer.score(output, rubric, {})
        assert result["followed"] is True

    def test_forbidden_word_detected(self):
        output = "The water cycle involves evaporation of water clouds rain."
        rubric = {"must_have": [], "must_not_have": []}
        constraints = {"forbidden_words": ["water", "rain", "cloud"]}
        result = self.scorer.score(output, rubric, constraints)
        assert result["followed"] is False

    def test_exact_sentence_count_pass(self):
        output = "First sentence here. Second sentence here. Third sentence here."
        result = self.scorer.score(output, {}, {"sentence_count": 3})
        assert result["checks_passed"] >= 1

    def test_exact_sentence_count_fail(self):
        output = "Only one sentence."
        result = self.scorer.score(output, {}, {"sentence_count": 3})
        # should fail the sentence count check
        relevant_check = next(
            (c for c in result["check_details"] if "sentence_count" in c["name"]), None
        )
        assert relevant_check is not None
        assert relevant_check["passed"] is False

    def test_valid_json_check_pass(self):
        output = '{"name": "Alice", "age": 30}'
        result = self.scorer.score(output, {}, {"valid_json": True})
        json_check = next(
            (c for c in result["check_details"] if "json" in c["name"].lower()), None
        )
        assert json_check is not None
        assert json_check["passed"] is True

    def test_valid_json_check_fail(self):
        output = '{name: Alice, age: 30}'  # invalid JSON
        result = self.scorer.score(output, {}, {"valid_json": True})
        json_check = next(
            (c for c in result["check_details"] if "json" in c["name"].lower()), None
        )
        assert json_check is not None
        assert json_check["passed"] is False

    def test_required_keywords_all_present(self):
        output = "The anthropogenic causes require mitigation strategies to build resilience."
        result = self.scorer.score(
            output, {}, {"required_keywords": ["anthropogenic", "mitigation", "resilience"]}
        )
        kw_check = next(
            (c for c in result["check_details"] if "required_keywords" in c["name"]), None
        )
        assert kw_check is not None
        assert kw_check["passed"] is True

    def test_required_keywords_missing(self):
        output = "Climate change is a serious problem requiring action."
        result = self.scorer.score(
            output, {}, {"required_keywords": ["anthropogenic", "mitigation"]}
        )
        kw_check = next(
            (c for c in result["check_details"] if "required_keywords" in c["name"]), None
        )
        assert kw_check is not None
        assert kw_check["passed"] is False

    def test_score_range(self):
        result = self.scorer.score("Some response.", {}, {})
        assert 0.0 <= result["score"] <= 1.0

    def test_empty_rubric_full_score(self):
        result = self.scorer.score("Any response.", {}, {})
        assert result["score"] == 1.0

    def test_word_count_constraint(self):
        output = " ".join(["word"] * 50)  # exactly 50 words
        result = self.scorer.score(output, {}, {"word_count_exact": 50})
        wc_check = next(
            (c for c in result["check_details"] if "word_count" in c["name"]), None
        )
        assert wc_check is not None
        assert wc_check["passed"] is True

    def test_markdown_h2_format(self):
        output = "## Section One\nContent here.\n## Section Two\nMore content."
        result = self.scorer.score(output, {}, {"format": "markdown_h2"})
        fmt_check = next(
            (c for c in result["check_details"] if "format" in c["name"].lower()), None
        )
        assert fmt_check is not None
        assert fmt_check["passed"] is True


# ===========================================================================
# Hallucination Detector Tests
# ===========================================================================

class TestHallucinationDetector:

    def setup_method(self):
        self.detector = HallucinationDetector()

    def test_clean_output_low_risk(self):
        output = "Mitochondria produce ATP through cellular respiration in eukaryotic cells."
        result = self.detector.detect(
            output=output,
            anchors=["mitochondria", "ATP", "organelle"],
            ground_truth={"organelle": "mitochondria", "function": "produces ATP"},
        )
        assert result["risk_score"] < 0.5
        assert result["detected"] is False

    def test_error_marker_high_risk(self):
        output = "[ERROR] API call failed after 3 attempts: timeout"
        result = self.detector.detect(output=output, anchors=[], ground_truth={})
        assert result["detected"] is True
        assert result["risk_score"] >= 0.5

    def test_missing_anchors_flagged(self):
        output = "The sky is blue and the grass is green, completely unrelated to the topic."
        result = self.detector.detect(
            output=output,
            anchors=["quantum", "entanglement", "physics", "particles"],
            ground_truth={},
        )
        # Should flag low anchor coverage
        assert result["flag_count"] > 0

    def test_full_anchor_coverage_no_flag(self):
        output = "Quantum entanglement links particles across distances through physics laws."
        result = self.detector.detect(
            output=output,
            anchors=["quantum", "entanglement", "particles"],
            ground_truth={},
        )
        # anchor coverage check should not trigger
        anchor_flags = [f for f in result["flags"] if "ANCHOR" in f["type"]]
        assert len(anchor_flags) == 0

    def test_no_anchors_no_anchor_flags(self):
        output = "Some creative writing output with no factual claims."
        result = self.detector.detect(output=output, anchors=[], ground_truth={})
        anchor_flags = [f for f in result["flags"] if "ANCHOR" in f["type"]]
        assert len(anchor_flags) == 0

    def test_risk_score_range(self):
        result = self.detector.detect(
            output="Test output.", anchors=[], ground_truth={}
        )
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_fabrication_signal_detected(self):
        output = (
            "According to studies show research proves that it is well-known "
            "experts agree this works in 1842."
        )
        result = self.detector.detect(output=output, anchors=[], ground_truth={})
        flag_types = [f["type"] for f in result["flags"]]
        assert "FABRICATION_LANGUAGE" in flag_types


# ===========================================================================
# Metrics Collector Tests
# ===========================================================================

def make_result(
    prompt_id: str,
    category: str = "test",
    difficulty: str = "easy",
    base_if_score: float = 0.5,
    sft_if_score: float = 0.8,
    base_hall_risk: float = 0.1,
    sft_hall_risk: float = 0.2,
) -> dict:
    return {
        "prompt_id": prompt_id,
        "category": category,
        "difficulty": difficulty,
        "prompt_text": "Test prompt",
        "base_output": "Base output",
        "sft_output": "SFT output",
        "annotator_note": "",
        "base": {
            "instruction_following": {
                "score": base_if_score,
                "followed": base_if_score >= 0.75,
                "checks_passed": 3,
                "checks_total": 4,
                "check_details": [],
            },
            "hallucination": {
                "risk_score": base_hall_risk,
                "detected": base_hall_risk >= 0.5,
                "flag_count": 0,
                "flags": [],
            },
        },
        "sft": {
            "instruction_following": {
                "score": sft_if_score,
                "followed": sft_if_score >= 0.75,
                "checks_passed": 4,
                "checks_total": 4,
                "check_details": [],
            },
            "hallucination": {
                "risk_score": sft_hall_risk,
                "detected": sft_hall_risk >= 0.5,
                "flag_count": 0,
                "flags": [],
            },
        },
        "delta": {
            "if_score_change": round(sft_if_score - base_if_score, 4),
            "hallucination_risk_change": round(sft_hall_risk - base_hall_risk, 4),
        },
    }


class TestMetricsCollector:

    def test_summary_keys_present(self):
        collector = MetricsCollector()
        collector.add_result(make_result("P1"))
        summary = collector.summary()
        required_keys = [
            "total_prompts", "base_if_rate", "sft_if_rate",
            "if_improvement", "base_hallucination_rate",
            "sft_hallucination_rate", "validation_pass",
        ]
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"

    def test_if_improvement_calculation(self):
        collector = MetricsCollector()
        # base follows 0 out of 2, sft follows 2 out of 2
        collector.add_result(make_result("P1", base_if_score=0.5, sft_if_score=0.9))
        collector.add_result(make_result("P2", base_if_score=0.6, sft_if_score=0.8))
        summary = collector.summary()
        # base_if_rate should be 0% (neither >=0.75), sft 100% (both >=0.75)
        assert summary["base_if_rate"] == 0.0
        assert summary["sft_if_rate"] == 1.0
        assert summary["if_improvement"] > 0

    def test_new_hallucination_detection(self):
        collector = MetricsCollector()
        # base clean (risk 0.1), SFT hallucinates (risk 0.8)
        collector.add_result(make_result("P1", base_hall_risk=0.1, sft_hall_risk=0.8))
        summary = collector.summary()
        assert summary["new_hallucination_count"] == 1
        assert "P1" in summary["new_hallucination_prompts"]
        assert summary["validation_criteria"]["no_new_hallucinations"] is False

    def test_validation_pass_when_both_criteria_met(self):
        collector = MetricsCollector()
        collector.add_result(make_result("P1", base_if_score=0.6, sft_if_score=0.9,
                                         base_hall_risk=0.1, sft_hall_risk=0.2))
        summary = collector.summary()
        assert summary["validation_pass"] is True

    def test_empty_collector_returns_error(self):
        collector = MetricsCollector()
        summary = collector.summary()
        assert "error" in summary

    def test_per_category_breakdown(self):
        collector = MetricsCollector()
        collector.add_result(make_result("P1", category="factual_qa"))
        collector.add_result(make_result("P2", category="reasoning"))
        summary = collector.summary()
        assert "factual_qa" in summary["per_category_if_rate"]
        assert "reasoning" in summary["per_category_if_rate"]

    def test_csv_export(self, tmp_path):
        collector = MetricsCollector()
        collector.add_result(make_result("P1"))
        csv_path = str(tmp_path / "test_output.csv")
        collector.to_csv(csv_path)
        assert Path(csv_path).exists()
        content = Path(csv_path).read_text()
        assert "prompt_id" in content
        assert "P1" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
