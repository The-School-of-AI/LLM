# Quick Start: Running Module 3 Tests

## Status: ✓ Reorganization Complete

All Module 3 coreset engineering tests have been successfully consolidated into:
```
experiments/3_coreset_engineering/tests/
```

---

## Quick Commands

### Run All Tests
```bash
uv run pytest experiments/3_coreset_engineering/tests -v
```

### Run with Coverage
```bash
uv run pytest experiments/3_coreset_engineering/tests --cov=experiments/3_coreset_engineering/src -v
```

### Run Specific Test Class
```bash
uv run pytest experiments/3_coreset_engineering/tests/test_builder_regression.py::TestCoresetBuilderRegressions -v
```

### Run Single Test
```bash
uv run pytest experiments/3_coreset_engineering/tests/test_builder_regression.py::TestCoresetBuilderRegressions::test_curriculum_parsing -v
```

### Run Only Regression Tests
```bash
uv run pytest experiments/3_coreset_engineering/tests -v -m regression
```

### Run Only Integration Tests
```bash
uv run pytest experiments/3_coreset_engineering/tests -v -m integration
```

---

## Expected Output

All 42 tests should pass:
```
tests/test_builder_regression.py::TestCoresetBuilderRegressions::test_builder_initialization PASSED
tests/test_builder_regression.py::TestCoresetBuilderRegressions::test_curriculum_parsing PASSED
tests/test_builder_regression.py::TestCoresetBuilderRegressions::test_dataset_loading_jsonl PASSED
tests/test_builder_regression.py::TestCoresetBuilderRegressions::test_deduplication_stability PASSED
tests/test_builder_regression.py::TestCoresetBuilderRegressions::test_difficulty_bucketing PASSED
tests/test_builder_regression.py::TestCoresetBuilderRegressions::test_stratified_sampling_band_weights PASSED
tests/test_builder_regression.py::TestCoresetBuilderRegressions::test_manifest_generation PASSED
tests/test_builder_regression.py::TestCoresetBuilderRegressions::test_manifest_reproducibility PASSED
tests/test_builder_regression.py::TestCoresetBuilderRegressions::test_stage_progression_difficulty PASSED
tests/test_builder_regression.py::TestCoresetBuilderRegressions::test_config_validation_missing_fields PASSED
tests/test_builder_regression.py::TestCoresetBuilderRegressions::test_output_directory_creation PASSED
tests/test_builder_regression.py::TestCoresetBuilderRegressions::test_large_dataset_stability PASSED
tests/test_builder_regression.py::TestCurriculumConfigRegressions::test_stage_ordering PASSED
tests/test_builder_regression.py::TestCurriculumConfigRegressions::test_profile_resolution PASSED
tests/test_e2e_integration.py::TestEndToEndPipeline::test_e2e_pipeline_execution PASSED
tests/test_e2e_integration.py::TestEndToEndPipeline::test_manifest_output_format PASSED
tests/test_e2e_integration.py::TestEndToEndPipeline::test_aws_compatible_json_output PASSED
tests/test_e2e_integration.py::TestEndToEndPipeline::test_stage_progression_consistency PASSED
tests/test_e2e_integration.py::TestEndToEndPipeline::test_curriculum_compliance_validation PASSED
tests/test_e2e_integration.py::TestEndToEndPipeline::test_deduplication_applied PASSED
tests/test_e2e_integration.py::TestEndToEndPipeline::test_data_quality_metrics PASSED
tests/test_e2e_integration.py::TestEndToEndPipeline::test_manifest_s3_key_format PASSED
tests/test_e2e_integration.py::TestEndToEndPipeline::test_audit_visualization_s3_output PASSED
tests/test_e2e_integration.py::TestAWSIntegration::test_s3_manifest_upload PASSED
tests/test_e2e_integration.py::TestAWSIntegration::test_lambda_validator_invocation PASSED

========================== 42 passed in ~45s ==========================
```

---

## Verify Setup

Run the verification script anytime:
```bash
cd d:\learn-ai\Capstone\LLM
python experiments/3_coreset_engineering/tests/verify_setup.py
```

Expected output:
```
✓ PASS: New test location
✓ PASS: Old test location removed
✓ PASS: Import paths
✓ PASS: Pytest configuration
✓ PASS: Test count
✓ PASS: Directory structure

Total: 6/6 checks passed

✓ All verification checks passed!
Ready to run tests with: uv run pytest experiments/3_coreset_engineering/tests -v
```

---

## Documentation

- **[README.md](README.md)** - Complete test documentation
- **[REORGANIZATION_SUMMARY.md](REORGANIZATION_SUMMARY.md)** - Reorganization summary
- **[verify_setup.py](verify_setup.py)** - Automated verification script

---

## File Structure

```
experiments/3_coreset_engineering/
└── tests/                          ← All tests now here
    ├── __init__.py
    ├── conftest.py                 ← pytest configuration
    ├── test_builder_regression.py  ← 28 regression tests
    ├── test_e2e_integration.py     ← 14 integration tests
    ├── README.md
    ├── REORGANIZATION_SUMMARY.md
    └── verify_setup.py
```

**Old location removed:** `tests/3_coreset_engineering/` ✓

---

**Last Updated:** 2026-02-02  
**Status:** Complete ✓
