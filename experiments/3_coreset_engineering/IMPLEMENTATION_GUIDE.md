# Module 3 - Reproducibility & Determinism Implementation Guide

**Date:** 2026-02-02  
**Scope:** Coreset Engineering Module (Team 3)  
**Aligned with:** Team 2 (Curriculum), Team 19 (Reproducibility)

---

## Overview

Module 3 has been updated to implement comprehensive reproducibility and determinism policies across all stages of coreset engineering. This document maps the L1 (foundations) and L2 (implementation) requirements to concrete code, configs, and tests.

---

## L1: Foundations - What Was Implemented

### 1. Repo Structure & Configs

**Files Created/Updated:**

```
experiments/3_coreset_engineering/
├── configs/
│   ├── seed_policy.yaml              ✓ NEW - Seed management policy
│   ├── manifest_schema.json          ✓ NEW - Manifest validation schema
│   ├── curriculum.yaml               ✓ UPDATED - Added checksums
│   └── sampling_policy.yaml          (reserved for future use)
│
├── src/coreset_engine/
│   ├── reproducibility.py            ✓ NEW - Core reproducibility module
│   └── __init__.py                   (existing)
│
├── scripts/
│   └── validate_manifests.py         ✓ NEW - CI/CD validation tool
│
├── tests/
│   ├── test_reproducibility.py       ✓ NEW - Reproducibility tests
│   ├── conftest.py                   ✓ UPDATED - Added reproducibility fixtures
│   └── (other tests unchanged)
│
├── Dockerfile                        ✓ UPDATED - Deterministic build environment
└── REPRODUCIBILITY_POLICY.md         ✓ NEW - Policy documentation
```

---

### 2. Deterministic Execution Guarantees

**Core Principle:** Same seed + same config → identical output every time.

#### Seed Management
```python
# From: experiments/3_coreset_engineering/src/coreset_engine/reproducibility.py

class SeedPolicy:
    """Enforces single seed initialization at pipeline start."""
    
    CANONICAL_SEED = 42  # Immutable
    
    def initialize(self):
        """Set seed exactly once. Raises error if called twice."""
        if self._seed_initialized:
            raise RuntimeError("Seed already initialized!")
        
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        self._seed_initialized = True
```

**Usage Pattern:**
```python
# At pipeline entry point
policy = SeedPolicy('configs/seed_policy.yaml')
policy.initialize()  # Called exactly once

# All downstream randomness is deterministic
indices = random.sample(population, k=1000)  # Seeded
```

#### Deterministic Iteration
```python
# Good (deterministic):
for item in sorted(items, key=lambda x: x['id']):
    process(item)

# Bad (non-deterministic):
for item in set_of_items:  # Sets are unordered!
    process(item)

# Use helper:
for item in DeterminismChecker.sorted_items(items):
    process(item)
```

---

### 3. Manifest Schemas & Fingerprints

**Canonical Manifest Structure:**
```json
{
  "schema_version": "0.2",
  "manifest_type": "coreset_engineering",
  "frozen_on": "2026-02-02",
  "metadata": {
    "curriculum_version": "0.2",
    "curriculum_checksum": "sha256:abc123...",
    "seed": 42,
    "execution_date": "2026-02-02T10:30:00Z",
    "execution_id": "run_abc123_stage_1B"
  },
  "stages": {
    "1B": {
      "manifest_fingerprint": "sha256:xyz789...",
      "indices": [0, 5, 12, 45, ...],  # MUST BE SORTED
      "statistics": { ... }
    }
  },
  "audit_trail": {
    "reproducibility_checksum": "sha256:..."
  }
}
```

**Fingerprinting Algorithm:**
```python
def compute_indices_fingerprint(indices: List[int]) -> str:
    """Deterministic fingerprint of manifest indices."""
    # 1. Sort (enforces canonical ordering)
    sorted_indices = sorted(indices)
    
    # 2. Serialize to JSON deterministically
    json_str = json.dumps(sorted_indices, separators=(',', ':'), sort_keys=True)
    
    # 3. Hash with SHA256
    hash_value = hashlib.sha256(json_str.encode()).hexdigest()
    
    # 4. Return in canonical format
    return f"sha256:{hash_value}"
```

**Schema Validation:** See [manifest_schema.json](configs/manifest_schema.json) - validates all manifests against canonical schema.

---

### 4. Replay Verification

**Core Concept:** Identical seed + config should produce identical manifests.

```python
# Run 1
builder_1 = CoresetBuilder(seed=42, config_path='configs/curriculum.yaml')
manifest_1 = builder_1.build_manifest(dataset)
fp_1 = compute_fingerprint(manifest_1['1B']['indices'])

# Run 2 (identical inputs)
builder_2 = CoresetBuilder(seed=42, config_path='configs/curriculum.yaml')
manifest_2 = builder_2.build_manifest(dataset)
fp_2 = compute_fingerprint(manifest_2['1B']['indices'])

# MUST be identical
assert fp_1 == fp_2  # Proves determinism
```

**Test Coverage:** See [test_reproducibility.py](tests/test_reproducibility.py) - 20+ tests verifying replay determinism.

---

### 5. CI & Regression Tests

**GitHub Actions Integration:**
```yaml
# In .github/workflows/coreset-deploy.yml

- name: Run Reproducibility Tests
  run: |
    uv run pytest experiments/3_coreset_engineering/tests/test_reproducibility.py -v

- name: Validate Manifest Schema
  run: |
    python experiments/3_coreset_engineering/scripts/validate_manifests.py

- name: Check Determinism
  run: |
    uv run pytest experiments/3_coreset_engineering/tests -k determinism -v
```

---

### 6. Audit Tooling

**ManifestAuditor Tool:**
```bash
# Validate a single manifest
python scripts/validate_manifests.py --manifest-file output/manifest_1B.json

# Validate entire directory
python scripts/validate_manifests.py --manifest-dir output/manifests/

# Output results as JSON
python scripts/validate_manifests.py --manifest-dir output/manifests/ \
  --json-output audit_results.json
```

**Audit Checks:**
1. ✓ Seed correctness (must be 42)
2. ✓ Indices are sorted
3. ✓ Fingerprints match computed values
4. ✓ Schema compliance
5. ✓ Audit trail completeness

---

## L2: Implementation Tasks - Code References

### Task 1: Config + Seed Policy ✓

**Files:**
- [configs/seed_policy.yaml](configs/seed_policy.yaml) - Full seed policy with rules
- [src/coreset_engine/reproducibility.py](src/coreset_engine/reproducibility.py) - SeedPolicy class

**Key Rules:**
```yaml
global_seed: 42
initialization:
  timing: "before_any_randomness"
  frequency: "exactly once per execution"
  checkpoint: "Verify before manifest generation"

reset_policy:
  - rule_name: "no_mid_pipeline_reset"
    violation_severity: "CRITICAL"
```

**Usage:**
```python
from coreset_engine.reproducibility import SeedPolicy

policy = SeedPolicy('configs/seed_policy.yaml')
policy.initialize()  # Once at pipeline start
policy.validate_at_checkpoint('manifest_generation')  # Before critical ops
```

---

### Task 2: Manifest JSON Schema ✓

**File:** [configs/manifest_schema.json](configs/manifest_schema.json)

**Key Requirements:**
- `schema_version`: "0.2"
- `metadata.seed`: 42 (must be exactly 42)
- `stages.*.indices`: Strictly sorted array
- `stages.*.manifest_fingerprint`: SHA256 hash
- `audit_trail.reproducibility_checksum`: Present and valid

**Validation in Tests:**
```python
import jsonschema

with open('configs/manifest_schema.json') as f:
    schema = json.load(f)

jsonschema.validate(instance=manifest, schema=schema)
```

---

### Task 3: Deterministic Ordering Enforcement ✓

**File:** [src/coreset_engine/reproducibility.py](src/coreset_engine/reproducibility.py) - `DeterminismChecker` class

**Enforcement Patterns:**
```python
# Use this helper for all iterations
from coreset_engine.reproducibility import DeterminismChecker

for item in DeterminismChecker.sorted_items(items):
    process(item)

# Verify lists are sorted
DeterminismChecker.assert_sorted(indices, name="band_indices")
```

---

### Task 4: Replay Test Scripts ✓

**File:** [tests/test_reproducibility.py](tests/test_reproducibility.py)

**Key Tests:**
1. `TestSeedPolicy` - Seed initialization and validation (4 tests)
2. `TestManifestFingerprinting` - Fingerprint computation (6 tests)
3. `TestReproducibilityValidator` - Manifest validation (7 tests)
4. `TestDeterministicManifestGeneration` - Replay tests (3 tests)
5. `TestEndToEndReproducibility` - Full workflow (2 tests)

**Total:** 22 reproducibility tests

**Run Tests:**
```bash
# All reproducibility tests
uv run pytest experiments/3_coreset_engineering/tests/test_reproducibility.py -v

# Only determinism tests
uv run pytest experiments/3_coreset_engineering/tests -k determinism -v

# Specific test class
uv run pytest experiments/3_coreset_engineering/tests/test_reproducibility.py::TestSeedPolicy -v
```

---

### Task 5: CI Automation ✓

**Updated File:** [.github/workflows/coreset-deploy.yml](.github/workflows/coreset-deploy.yml)

**New CI Steps:**
1. Run reproducibility tests
2. Validate manifest schema
3. Audit manifest fingerprints
4. Check seed policy compliance

**Expected CI Output:**
```
- Reproducibility Tests: PASS (22/22)
- Manifest Validation: PASS (all schemas compliant)
- Determinism Checks: PASS (fingerprints correct)
- Seed Policy: PASS (seed=42 verified)
```

---

### Task 6: Audit Tooling ✓

**File:** [scripts/validate_manifests.py](scripts/validate_manifests.py)

**Features:**
- Single file validation
- Directory batch validation
- JSON output for CI/CD systems
- Detailed issue reporting

**CLI Usage:**
```bash
# Validate single manifest
python scripts/validate_manifests.py --manifest-file manifest.json

# Validate directory
python scripts/validate_manifests.py --manifest-dir output/manifests/

# Output JSON
python scripts/validate_manifests.py \
  --manifest-dir output/manifests/ \
  --json-output results.json
```

---

## Complete List of Changes

### New Files (9 total)

1. ✓ `REPRODUCIBILITY_POLICY.md` - Policy documentation
2. ✓ `configs/seed_policy.yaml` - Seed management config
3. ✓ `configs/manifest_schema.json` - Manifest validation schema
4. ✓ `src/coreset_engine/reproducibility.py` - Core reproducibility module
5. ✓ `scripts/validate_manifests.py` - CI/CD validation tool
6. ✓ `tests/test_reproducibility.py` - Reproducibility tests (22 tests)

### Updated Files (2 total)

1. ✓ `Dockerfile` - Deterministic build environment
2. ✓ `tests/conftest.py` - Added reproducibility fixtures

---

## Quick Start

### 1. Initialize Seed (in your code)
```python
from coreset_engine.reproducibility import SeedPolicy

policy = SeedPolicy('configs/seed_policy.yaml')
policy.initialize()  # Must be called exactly once at start
```

### 2. Generate Manifests (deterministically)
```python
# With seeded randomness, manifests are deterministic
builder = CoresetBuilder()
manifests = builder.build(dataset)
# output/manifests/1B/manifest.json, 3B/..., etc.
```

### 3. Validate Manifests (in CI)
```bash
python scripts/validate_manifests.py --manifest-dir output/manifests/
# ✓ All manifests passed reproducibility validation!
```

### 4. Run Tests
```bash
# All reproducibility tests
uv run pytest experiments/3_coreset_engineering/tests/test_reproducibility.py -v

# Only determinism tests
uv run pytest experiments/3_coreset_engineering/tests -k determinism -v
```

---

## Policy Compliance Checklist

- [x] **Seed Management**
  - Global seed set to 42 (canonical value)
  - Initialized exactly once at pipeline start
  - Never reset mid-pipeline
  - Validated at critical checkpoints

- [x] **Deterministic Iteration**
  - All sets/dicts sorted before iteration
  - Using `DeterminismChecker.sorted_items()` helper
  - Code review verifies no unseeded randomness

- [x] **Manifest Generation**
  - Indices strictly sorted
  - Fingerprints computed deterministically
  - Schema validated before output
  - Audit trail includes reproducibility checksum

- [x] **Configuration**
  - `seed_policy.yaml` defines seed rules
  - `manifest_schema.json` validates output format
  - `curriculum.yaml` includes schema checksum
  - All configs versioned and immutable

- [x] **Testing**
  - 22 reproducibility tests covering all aspects
  - Replay tests verify identical input → identical output
  - Schema validation tests
  - Fingerprint correctness tests
  - Integrated in CI/CD pipeline

- [x] **CI/CD Integration**
  - Reproducibility tests run on every commit
  - Manifest validation in deployment step
  - Seed policy compliance checked
  - Audit tooling available for debugging

- [x] **Documentation**
  - Policy documented in REPRODUCIBILITY_POLICY.md
  - Code examples in docstrings
  - CI/CD integration documented
  - Quick start guide provided

---

## Troubleshooting

### Issue: "Seed not initialized at checkpoint"
**Solution:** Call `policy.initialize()` at pipeline start (exactly once).

### Issue: "Indices not sorted" warning
**Solution:** Use `sorted(indices)` before storing in manifest.

### Issue: "Fingerprint mismatch"
**Solution:** Verify indices are in canonical sorted order and algorithm hasn't changed.

### Issue: CI validation fails
**Solution:** Run `python scripts/validate_manifests.py --manifest-dir output/manifests/` locally to debug.

---

## Related Documentation

- [REPRODUCIBILITY_POLICY.md](REPRODUCIBILITY_POLICY.md) - Full policy details
- [configs/seed_policy.yaml](configs/seed_policy.yaml) - Seed management rules
- [configs/manifest_schema.json](configs/manifest_schema.json) - Manifest format spec
- [tests/test_reproducibility.py](tests/test_reproducibility.py) - Test implementation
- [scripts/validate_manifests.py](scripts/validate_manifests.py) - Validation tool

---

## References

- **Team 2 (Curriculum):** Curriculum frozen and checksummed
- **Team 19 (Reproducibility):** Reproducibility guidelines and standards
- **Team 3 (Coreset):** Implementation and maintenance of policies

---

**Status:** ✓ Complete - All L1 and L2 requirements implemented  
**Date:** 2026-02-02  
**Approval:** Ready for Team 2 + Team 19 review
