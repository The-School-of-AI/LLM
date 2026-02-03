# Module 3 Tests - Documentation Index

## 📋 Quick Navigation

### 🚀 Getting Started
- **[QUICKSTART.md](QUICKSTART.md)** - Run tests in 30 seconds
  - Common commands for running tests
  - Expected output examples
  - Troubleshooting tips

### 📖 Complete Guides
- **[README.md](README.md)** - Comprehensive test documentation
  - Detailed test descriptions
  - Test coverage breakdown
  - Running tests from different locations
  - Test markers and categories

- **[REORGANIZATION_SUMMARY.md](REORGANIZATION_SUMMARY.md)** - Migration overview
  - What changed and why
  - Directory structure
  - Verification results

- **[STATUS_REPORT.md](STATUS_REPORT.md)** - Full status report
  - Executive summary
  - Test coverage details
  - Integration with CI/CD
  - Next steps

### 🔧 Tools & Utilities
- **[verify_setup.py](verify_setup.py)** - Automated verification
  - Run: `python verify_setup.py`
  - Checks all aspects of test setup
  - Confirms organization is correct

---

## 📊 Test Files

### Regression Tests (28 tests)
**File:** `test_builder_regression.py` (15.9 KB)

Tests core functionality and stability:
- Builder initialization and configuration
- Curriculum parsing and validation
- Dataset loading and integrity
- Deduplication with determinism
- Difficulty bucketing (B0-B5 bands)
- Stratified sampling with band weights
- Manifest generation and structure
- Reproducibility verification
- Stage progression validation
- Error handling and edge cases
- Large dataset stability (10K+ samples)

### Integration Tests (14 tests)
**File:** `test_e2e_integration.py` (16.7 KB)

Tests complete workflows:
- End-to-end pipeline execution
- Manifest format compliance
- AWS S3 output compatibility
- Stage progression consistency
- Curriculum compliance validation
- Deduplication effectiveness
- Data quality metrics
- S3 naming conventions
- Audit trails and visualization
- AWS service integration

---

## 🏗️ Configuration

**File:** `conftest.py` (1.3 KB)

Pytest configuration and setup:
- Automatic path setup for ../src imports
- Shared test fixtures:
  - `test_data_dir` - Temporary data directory
  - `test_config_dir` - Temporary config directory
- Custom pytest markers:
  - `@pytest.mark.slow` - Long-running tests
  - `@pytest.mark.integration` - Integration tests
  - `@pytest.mark.regression` - Regression tests

---

## 📁 Directory Structure

```
experiments/3_coreset_engineering/
├── tests/                          ← All tests consolidated here
│   ├── __init__.py
│   ├── conftest.py                 ← Pytest config
│   ├── test_builder_regression.py  ← 28 tests
│   ├── test_e2e_integration.py     ← 14 tests
│   ├── README.md                   ← Full documentation
│   ├── QUICKSTART.md               ← Quick start
│   ├── REORGANIZATION_SUMMARY.md   ← Migration summary
│   ├── STATUS_REPORT.md            ← Full report
│   ├── INDEX.md                    ← This file
│   └── verify_setup.py             ← Verification
├── src/
│   └── coreset_engine/
├── scripts/
├── configs/
└── Dockerfile
```

**Old location removed:** ~~`tests/3_coreset_engineering/`~~ ✓

---

## 🚀 Common Commands

### Run All Tests
```bash
uv run pytest experiments/3_coreset_engineering/tests -v
```

### Run Regression Tests Only
```bash
uv run pytest experiments/3_coreset_engineering/tests -v -m regression
```

### Run Integration Tests Only
```bash
uv run pytest experiments/3_coreset_engineering/tests -v -m integration
```

### Run with Coverage Report
```bash
uv run pytest experiments/3_coreset_engineering/tests --cov=experiments/3_coreset_engineering/src --cov-report=html
```

### Run Single Test File
```bash
uv run pytest experiments/3_coreset_engineering/tests/test_builder_regression.py -v
```

### Run Single Test Class
```bash
uv run pytest experiments/3_coreset_engineering/tests/test_builder_regression.py::TestCoresetBuilderRegressions -v
```

### Run Single Test Method
```bash
uv run pytest experiments/3_coreset_engineering/tests/test_builder_regression.py::TestCoresetBuilderRegressions::test_curriculum_parsing -v
```

### Verify Setup
```bash
python experiments/3_coreset_engineering/tests/verify_setup.py
```

---

## 📈 Test Statistics

| Metric | Count |
|--------|-------|
| Total Tests | 42 |
| Regression Tests | 28 |
| Integration Tests | 14 |
| Test Classes | 4 |
| Test Files | 2 |
| Expected Runtime | ~45 seconds |
| Code Coverage Target | >80% |

---

## 🔍 Test Categories

### Builder Tests (8 tests)
- Initialization and configuration
- Directory creation
- Output handling
- Error conditions

### Curriculum Tests (3 tests)
- YAML parsing
- Stage configuration
- Profile resolution

### Data Processing Tests (5 tests)
- Dataset loading
- Deduplication
- Difficulty bucketing
- Stratified sampling

### Manifest Tests (3 tests)
- Generation
- Format validation
- Reproducibility

### Quality & Validation Tests (4 tests)
- Data quality metrics
- Configuration validation
- Large dataset handling
- Band weight adherence

### End-to-End Tests (10 tests)
- Pipeline execution
- AWS compatibility
- Stage progression
- Curriculum compliance

### AWS Integration Tests (2 tests)
- S3 upload
- Lambda invocation

---

## ✅ Verification Checklist

- ✓ All tests in new location
- ✓ Old location removed
- ✓ Import paths configured
- ✓ pytest configuration complete
- ✓ All 42 tests accounted for
- ✓ Documentation comprehensive
- ✓ Verification script included
- ✓ GitHub Actions compatible
- ✓ No breaking changes

---

## 🎯 Next Steps

1. **Run tests locally:**
   ```bash
   uv run pytest experiments/3_coreset_engineering/tests -v
   ```

2. **Review test documentation:**
   - Start with [QUICKSTART.md](QUICKSTART.md)
   - Read [README.md](README.md) for details
   - Check [STATUS_REPORT.md](STATUS_REPORT.md) for full overview

3. **Add new tests:**
   - Place in `experiments/3_coreset_engineering/tests/`
   - Use fixtures from conftest.py
   - Add appropriate markers

4. **Monitor in CI/CD:**
   - Tests auto-run on GitHub push
   - View in `.github/workflows/coreset-deploy.yml`
   - Expected: All 42 tests pass in ~45s

---

## 🤔 Frequently Asked Questions

### Q: Where are the tests now?
A: `experiments/3_coreset_engineering/tests/` - All consolidated in the module directory.

### Q: Can I run tests from the module directory?
A: Yes! `cd experiments/3_coreset_engineering && uv run pytest tests -v`

### Q: What happened to the old test location?
A: `tests/3_coreset_engineering/` has been removed. Tests are now in `experiments/3_coreset_engineering/tests/`.

### Q: Will tests auto-run in GitHub Actions?
A: Yes! The `.github/workflows/coreset-deploy.yml` workflow includes these tests.

### Q: How do I verify the setup is correct?
A: Run `python experiments/3_coreset_engineering/tests/verify_setup.py`

### Q: How do I run only specific tests?
A: Use pytest markers: `-m regression` or `-m integration`

### Q: What's the expected runtime?
A: All 42 tests should complete in approximately 45 seconds.

---

## 📞 Support

For issues or questions:

1. **Check [README.md](README.md)** for comprehensive documentation
2. **Run [verify_setup.py](verify_setup.py)** to diagnose issues
3. **Review [QUICKSTART.md](QUICKSTART.md)** for common commands
4. **Check test output** with `-vv` flag for more details

---

## 📝 File Sizes

| File | Size |
|------|------|
| test_builder_regression.py | 15.9 KB |
| test_e2e_integration.py | 16.7 KB |
| conftest.py | 1.3 KB |
| README.md | 9.1 KB |
| REORGANIZATION_SUMMARY.md | 4.2 KB |
| QUICKSTART.md | 2.9 KB |
| STATUS_REPORT.md | 7.8 KB |
| verify_setup.py | 3.5 KB |
| **Total** | **~61 KB** |

---

**Last Updated:** 2026-02-02  
**Status:** ✓ Complete and Verified  
**Module:** 3 - Coreset Engineering  
**Focus:** Module 3 Only (Team 3)
