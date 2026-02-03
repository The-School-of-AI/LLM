# Module 3 Coreset Engineering Tests

## Overview

All tests for the coreset engineering module are now consolidated in `experiments/3_coreset_engineering/tests/`.

## Test Files

### 1. `test_builder_regression.py` (15.9 KB)
**28 regression tests** covering core coreset builder functionality:

#### TestCoresetBuilderRegressions (20 tests)
- `test_builder_initialization`: CoresetBuilder initialization with config
- `test_curriculum_parsing`: Curriculum YAML parsing and validation
- `test_dataset_loading_jsonl`: JSONL dataset loading integrity
- `test_deduplication_stability`: Deterministic deduplication
- `test_difficulty_bucketing`: Band assignment and validation
- `test_stratified_sampling_band_weights`: Band weight adherence
- `test_manifest_generation`: Manifest structure correctness
- `test_manifest_reproducibility`: Identical input → identical output
- `test_stage_progression_difficulty`: Curriculum difficulty progression (1B → 70B)
- `test_config_validation_missing_fields`: Invalid config error handling
- `test_output_directory_creation`: Auto-create output directories
- `test_large_dataset_stability`: Memory efficiency with large datasets (10K+ samples)
- Plus 8 additional core tests

#### TestCurriculumConfigRegressions (2 tests)
- `test_stage_ordering`: Stage order validation
- `test_profile_resolution`: Curriculum profile resolution

**Key Features:**
- Tests both happy path and error conditions
- Uses fixtures for test data setup
- Includes large dataset regression test (10x scaling)
- Validates determinism across re-runs
- Tests subset invariant: 1B ⊂ 3B ⊂ 8B ⊂ 70B

---

### 2. `test_e2e_integration.py` (16.7 KB)
**14 end-to-end integration tests** covering complete pipeline:

#### TestEndToEndPipeline (10 tests)
- `test_e2e_pipeline_execution`: Full pipeline from raw data → manifests
- `test_manifest_output_format`: JSON structure validation
- `test_aws_compatible_json_output`: AWS S3 JSON compatibility
- `test_stage_progression_consistency`: Subset property enforcement
- `test_curriculum_compliance_validation`: Curriculum requirement compliance
- `test_deduplication_applied`: Duplicate removal verification
- `test_data_quality_metrics`: Quality metrics generation
- `test_manifest_s3_key_format`: S3 naming convention compliance
- `test_audit_visualization_s3_output`: Audit trail and visualization output
# Module 3 — Tests (consolidated)

This directory contains the consolidated tests for Module 3 (Coreset Engineering).

Purpose
-------
- Keep all module tests in one place and point developers to the canonical module docs.

Run tests
---------
From the project root:

```powershell
uv run pytest experiments/3_coreset_engineering/tests -v
```

Run only regression or integration tests using markers:

```powershell
uv run pytest experiments/3_coreset_engineering/tests -v -m regression
uv run pytest experiments/3_coreset_engineering/tests -v -m integration
```

Test files
----------
- `test_builder_regression.py` — Regression tests for builder, sampling, and manifest generation.
- `test_e2e_integration.py` — End-to-end integration tests covering pipeline execution and outputs.
- `conftest.py` — Shared fixtures and pytest configuration (seed, fixtures, markers).

Notes
-----
- Detailed testing guidance and historical notes were consolidated into the module README and `IMPLEMENTATION_GUIDE.md`.
- If you need historic drafts, see `IMPLEMENTATION_GUIDE.md` and `REPRODUCIBILITY_POLICY.md`.

Last updated: 2026-02-02

CI & requirements
-----------------

- **Python**: 3.12 required for reproducible test results.
- **Install**: use `uv sync --frozen` to install locked deps; fallback `pip install uv` then `uv sync`.
- **Determinism env**: set `PYTHONHASHSEED=0` in CI.

Secrets & environment
---------------------

Some integration tests require AWS credentials. Add these GitHub Secrets to run CI:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`
- `S3_TEST_BUCKET` (optional; test-only bucket)

CI snippet
----------

Include the `Coreset CI` job from the module README in your `.github/workflows/` configuration — it sets Python 3.12, runs `uv sync --frozen`, runs tests, and invokes the manifest validator.

Validator behavior
------------------

`scripts/validate_manifests.py` exits with non-zero code on failures; CI will fail if manifests are invalid. Ensure manifests are produced under `output/manifests/` before validation.
