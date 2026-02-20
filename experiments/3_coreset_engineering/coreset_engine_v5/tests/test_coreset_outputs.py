"""
Test suite for coreset output validation
Tests the CoresetValidator against generated coreset outputs
"""

import os
from pathlib import Path

import pytest
from tools.validate_coreset_outputs import CoresetValidator

# These tests validate on-disk artifacts under output/coresets.
# They are intentionally opt-in because the repo may contain stale example outputs
# that do not correspond to the active curriculum schema.
RUN_EXISTING_OUTPUT_TESTS = os.environ.get("CORESET_TEST_EXISTING_OUTPUTS", "0") == "1"


class TestCoresetValidator:
    """Test coreset output validation"""

    @pytest.fixture
    def validator(self):
        """Create validator instance"""
        return CoresetValidator(
            curriculum_path="config/curriculum.yaml", output_base_dir="output/coresets"
        )

    def test_validator_initialization(self, validator):
        """Test validator initializes correctly"""
        assert validator.curriculum is not None
        assert validator.curriculum.version is not None
        assert len(validator.curriculum.stages) > 0

    def test_curriculum_loaded(self, validator):
        """Test curriculum is loaded"""
        assert "1B" in validator.curriculum.stages
        assert "3B" in validator.curriculum.stages
        assert "8B" in validator.curriculum.stages
        assert "70B" in validator.curriculum.stages

    def test_bands_available(self, validator):
        """Test bands are available from curriculum"""
        assert len(validator.curriculum.bands) >= 6
        from src.core.types import DifficultyBand, difficulty_band_order

        # Curriculum may add additional bands (e.g., B6). All must be valid DifficultyBand values.
        valid_band_names = set(difficulty_band_order())
        for band_enum in validator.curriculum.bands.keys():
            assert band_enum.value in valid_band_names

        # Base curricula are expected to define at least B0..B5.
        for band_str in ["B0", "B1", "B2", "B3", "B4", "B5"]:
            band = DifficultyBand(band_str)
            assert band in validator.curriculum.bands

    @pytest.mark.skipif(
        (not RUN_EXISTING_OUTPUT_TESTS)
        or (not (Path("output/coresets/1B/manifest.json").exists())),
        reason="1B coreset outputs not generated",
    )
    def test_validate_1b_stage(self, validator):
        """Test validation of 1B stage"""
        report = validator.validate_stage("1B")
        assert report is not None
        assert report.stage_name == "1B"
        assert len(report.checks) > 0

    @pytest.mark.skipif(
        (not RUN_EXISTING_OUTPUT_TESTS)
        or (not (Path("output/coresets/1B/manifest.json").exists())),
        reason="Coreset outputs not generated",
    )
    def test_manifest_exists_1b(self, validator):
        """Test manifest file exists for 1B"""
        report = validator.validate_stage("1B")
        manifest_checks = [c for c in report.checks if c.check_id == "FILE_MANIFEST"]
        assert len(manifest_checks) > 0
        assert manifest_checks[0].passed

    @pytest.mark.skipif(
        (not RUN_EXISTING_OUTPUT_TESTS)
        or (not (Path("output/coresets/1B/selected_indices.jsonl").exists())),
        reason="Indices file not generated",
    )
    def test_indices_exist_1b(self, validator):
        """Test indices file exists for 1B"""
        report = validator.validate_stage("1B")
        indices_checks = [c for c in report.checks if c.check_id == "FILE_INDICES"]
        assert len(indices_checks) > 0
        assert indices_checks[0].passed

    @pytest.mark.skipif(
        (not RUN_EXISTING_OUTPUT_TESTS)
        or (not (Path("output/coresets/1B/manifest.json").exists())),
        reason="Manifest not generated",
    )
    def test_manifest_structure_1b(self, validator):
        """Test manifest has required fields"""
        report = validator.validate_stage("1B")
        structure_checks = [c for c in report.checks if "MANIFEST_" in c.check_id]
        assert len(structure_checks) > 0
        for check in structure_checks:
            # Should have most required fields
            if check.severity == "high":
                assert check.passed, f"Missing field: {check.name}"

    @pytest.mark.skipif(
        (not RUN_EXISTING_OUTPUT_TESTS)
        or (not (Path("output/coresets/1B/manifest.json").exists())),
        reason="Manifest not generated",
    )
    def test_band_distribution_1b(self, validator):
        """Test band distribution against curriculum"""
        report = validator.validate_stage("1B")
        band_checks = [c for c in report.checks if c.category == "band_ratios"]
        assert len(band_checks) > 0

    @pytest.mark.skipif(
        (not RUN_EXISTING_OUTPUT_TESTS)
        or (not (Path("output/coresets/1B/manifest.json").exists())),
        reason="Outputs not generated",
    )
    def test_domain_distribution_valid_1b(self, validator):
        """Test domain distribution is valid"""
        report = validator.validate_stage("1B")
        domain_checks = [
            c for c in report.checks if c.category == "domain_distribution"
        ]
        # Should have at least some domain checks
        if domain_checks:
            for check in domain_checks:
                if check.severity == "high":
                    assert check.passed, f"Invalid domain: {check.name}"

    @pytest.mark.skipif(
        (not RUN_EXISTING_OUTPUT_TESTS)
        or (not (Path("output/coresets/1B/manifest.json").exists())),
        reason="Outputs not generated",
    )
    def test_language_policy_compliance_1b(self, validator):
        """Test language policy compliance"""
        report = validator.validate_stage("1B")
        lang_checks = [c for c in report.checks if c.category == "language_policy"]
        # Critical checks should all pass
        for check in lang_checks:
            if check.severity == "critical":
                assert check.passed, f"Language policy violation: {check.name}"

    @pytest.mark.skipif(
        (not RUN_EXISTING_OUTPUT_TESTS)
        or (not (Path("output/coresets/1B/manifest.json").exists())),
        reason="Outputs not generated",
    )
    def test_report_generation_1b(self, validator):
        """Test report can be generated"""
        validator.validate_stage("1B")
        report = validator.generate_report("1B")
        assert report is not None
        assert "1B" in report
        assert "VERIFICATION REPORT" in report

    @pytest.mark.skipif(
        (not RUN_EXISTING_OUTPUT_TESTS)
        or (not (Path("output/coresets/1B/manifest.json").exists())),
        reason="Outputs not generated",
    )
    def test_checklist_generation_1b(self, validator):
        """Test checklist can be generated"""
        validator.validate_stage("1B")
        checklist = validator.generate_checklist("1B")
        assert checklist is not None
        assert "CHECKLIST" in checklist
        assert "1B" in checklist

    @pytest.mark.skipif(
        (not RUN_EXISTING_OUTPUT_TESTS)
        or (not (Path("output/coresets/1B/manifest.json").exists())),
        reason="Outputs not generated",
    )
    def test_validation_summary(self, validator):
        """Test validation summary computation"""
        report = validator.validate_stage("1B")
        summary = report.get_summary()
        assert "total_checks" in summary
        assert "by_severity" in summary
        assert "by_status" in summary
        assert "success_rate" in summary
        assert summary["total_checks"] > 0


class TestValidationOutput:
    """Test validation output formats"""

    @pytest.fixture
    def validator(self):
        """Create validator"""
        return CoresetValidator(
            curriculum_path="config/curriculum.yaml", output_base_dir="output/coresets"
        )

    @pytest.mark.skipif(
        not (Path("output/coresets/1B/manifest.json").exists()),
        reason="Outputs not generated",
    )
    def test_checklist_format(self, validator):
        """Test checklist has proper format"""
        validator.validate_stage("1B")
        checklist = validator.generate_checklist("1B")

        # Should have sections
        assert "FILES" in checklist or "files" in checklist.lower()
        # Should have checkmarks or X marks
        assert ("✓" in checklist or "PASS" in checklist) or (
            "✗" in checklist or "FAIL" in checklist
        )
        # Should be readable text
        assert len(checklist) > 100

    @pytest.mark.skipif(
        not (Path("output/coresets/1B/manifest.json").exists()),
        reason="Outputs not generated",
    )
    def test_report_format(self, validator):
        """Test report has proper structure"""
        validator.validate_stage("1B")
        report = validator.generate_report("1B")

        # Should have summary section
        assert "SUMMARY" in report or "Summary" in report
        # Should have findings
        assert "FINDING" in report or "findings" in report.lower()
        # Should have breakdown
        assert "BREAKDOWN" in report or "breakdown" in report.lower()
        # Should be readable
        assert len(report) > 200


def test_integration_validate_and_report():
    """Integration test: validate outputs and generate reports"""
    if not Path("output/coresets/1B/manifest.json").exists():
        pytest.skip("Coreset outputs not generated")

    validator = CoresetValidator(
        curriculum_path="config/curriculum.yaml", output_base_dir="output/coresets"
    )

    # Validate stage
    report = validator.validate_stage("1B")
    assert report is not None

    # Generate outputs
    checklist = validator.generate_checklist("1B")
    detailed_report = validator.generate_report("1B")

    assert checklist is not None
    assert detailed_report is not None
    assert len(checklist) > 0
    assert len(detailed_report) > 0

    # Both should mention the stage
    assert "1B" in checklist
    assert "1B" in detailed_report
