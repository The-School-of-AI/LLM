# Module 3 Reproducibility & Determinism Policy

**Date:** 2026-02-02  
**Scope:** Coreset Engineering (Team 3)  
**Owner:** Team 2 (Curriculum Architects) + Team 3 (Coreset Engineering)

---

## Level 1: Foundations

### Repo Structure & Configs

#### 1.1 Config-as-Code Policy
```
experiments/3_coreset_engineering/configs/
├── curriculum.yaml          # Frozen curriculum schema (Team 2)
├── sampling_policy.yaml     # Deterministic sampling rules (Team 3)
├── manifest_schema.json     # Output manifest schema
└── seed_policy.yaml         # Seed management across stages
```

**Rule:** No hardcoded values in code. All parameters driven by YAML configs.

#### 1.2 Versioning & Checksums
```yaml
# curriculum.yaml
version: "0.2"
frozen_on: "2026-02-02"
schema_checksum: "sha256:abc123..."  # Immutable fingerprint
owner_team: "Team 2: Curriculum Architects"
```

**Rule:** Breaking changes require version bump + migration guide.

---

### Deterministic Execution Guarantees

#### 2.1 Seeding Policy
```python
# CANONICAL: experiments/3_coreset_engineering/src/reproducibility.py

import random
import numpy as np
import torch

GLOBAL_SEED = 42  # Frozen, immutable

def set_global_seed(seed: int = GLOBAL_SEED):
    """Set seed across all libraries for determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

**Requirement:** Every execution with same seed → identical output.

#### 2.2 Deterministic Ordering
```python
# Bad (non-deterministic):
for item in set_of_items:  # Sets are unordered!
    process(item)

# Good (deterministic):
for item in sorted(set_of_items, key=lambda x: x['id']):
    process(item)
```

**Rule:** Use `sorted()` everywhere. Never iterate over sets/dicts without sorting.

#### 2.3 Randomness Sealing
```python
# Before generating manifests
random.seed(42)

# All downstream randomness is deterministic
sample_idx = random.randint(0, len(dataset))  # Seeded
indices = random.sample(population, k=1000)    # Seeded
```

**Rule:** Seed is set exactly once before manifest generation, never re-seeded mid-pipeline.

---

### Manifest Schemas & Fingerprints

#### 3.1 Canonical Manifest Schema
```json
{
  "schema_version": "0.2",
  "manifest_type": "coreset_engineering",
  "frozen_on": "2026-02-02",
  "metadata": {
    "curriculum_version": "0.2",
    "curriculum_checksum": "sha256:...",
    "seed": 42,
    "execution_date": "2026-02-02T10:30:00Z",
    "execution_id": "run_abc123_stage_1B"
  },
  "stages": {
    "1B": {
      "manifest_fingerprint": "sha256:xyz789",
      "indices": [0, 5, 12, 45, ...],
      "statistics": {
        "total_samples": 1000,
        "band_distribution": {
          "B0": 400,
          "B1": 300,
          "B2": 200,
          "B3": 100,
          "B4": 0,
          "B5": 0
        },
        "modality_distribution": {
          "text": 800,
          "code": 200
        }
      }
    },
    "3B": { ... },
    "8B": { ... },
    "70B": { ... }
  },
  "audit_trail": {
    "version_history": ["0.1", "0.2"],
    "config_diffs": [],
    "reproducibility_checksum": "sha256:..."
  }
}
```

#### 3.2 Manifest Fingerprinting
```python
import hashlib
import json

def compute_manifest_fingerprint(indices: List[int]) -> str:
    """Deterministic fingerprint of manifest indices."""
    # Sort indices first (determinism)
    sorted_indices = sorted(indices)
    # Hash the JSON representation
    json_str = json.dumps(sorted_indices, separators=(',', ':'), sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()

# Example
fingerprint = compute_manifest_fingerprint([0, 5, 12, ...])
# fingerprint = "sha256:xyz789"
```

**Rule:** Fingerprints must match across identical inputs, exactly.

---

## Level 2: Implementation Tasks

### Task 1: Config + Seed Policy

#### 1.1 Create `seed_policy.yaml`
```yaml
# experiments/3_coreset_engineering/configs/seed_policy.yaml
version: "0.2"
owner_team: "Team 3: Coreset Engineering"
frozen_on: "2026-02-02"

global_seed: 42

seed_reset_rules:
  # Never reset seed mid-pipeline
  - rule: "no_mid_pipeline_reset"
    description: "Seed is set once at pipeline start, never reset"
    applies_to: ["sampling", "deduplication", "bucketing"]

  # Deterministic ordering for all sets
  - rule: "deterministic_iteration"
    description: "All iterations must use sorted() for reproducibility"
    applies_to: ["band_assignment", "modality_selection", "stage_progression"]

  # Seed validation checkpoints
  - rule: "seed_checkpoint_validation"
    description: "Verify seed is set before each major operation"
    applies_to: ["manifest_generation", "deduplication_init", "sampling_start"]

modality_randomization:
  # Reproducible modality selection
  strategy: "seeded_weights"
  method: "Use configured weights, never random modality assignment"

band_assignment_randomness:
  # Difficulty bucketing uses deterministic scoring, not randomness
  strategy: "deterministic_scoring"
  method: "Difficulty scores from heuristic/tokenizer, not random assignment"
```

#### 1.2 Implement `seed_policy.py`
```python
# experiments/3_coreset_engineering/src/reproducibility/seed_policy.py
import yaml
import random
import numpy as np
from typing import Dict, Any
from pathlib import Path

class SeedPolicy:
    """Manages seed initialization and validation."""
    
    def __init__(self, config_path: str = "configs/seed_policy.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.global_seed = self.config['global_seed']
        self._seed_initialized = False
    
    def initialize(self):
        """Set seed at pipeline start (called exactly once)."""
        if self._seed_initialized:
            raise RuntimeError("Seed already initialized! Cannot reset mid-pipeline.")
        
        random.seed(self.global_seed)
        np.random.seed(self.global_seed)
        self._seed_initialized = True
    
    def validate_at_checkpoint(self, checkpoint_name: str):
        """Verify seed was initialized before critical operations."""
        if not self._seed_initialized:
            raise RuntimeError(f"Seed not initialized at checkpoint '{checkpoint_name}'")
    
    @property
    def is_initialized(self) -> bool:
        return self._seed_initialized
```

---

### Task 2: Manifest JSON Schema

#### 2.1 Create `manifest_schema.json`
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Coreset Engineering Manifest Schema",
  "description": "Canonical schema for coreset manifests produced by Team 3",
  "version": "0.2",
  "type": "object",
  "required": ["schema_version", "metadata", "stages", "audit_trail"],
  "properties": {
    "schema_version": {
      "type": "string",
      "pattern": "^[0-9]+\\.[0-9]+$",
      "description": "Manifest schema version (e.g., '0.2')"
    },
    "metadata": {
      "type": "object",
      "required": ["curriculum_version", "seed", "execution_date"],
      "properties": {
        "curriculum_version": {
          "type": "string",
          "pattern": "^[0-9]+\\.[0-9]+$"
        },
        "curriculum_checksum": {
          "type": "string",
          "pattern": "^sha256:[a-f0-9]{64}$",
          "description": "SHA256 checksum of curriculum.yaml"
        },
        "seed": {
          "type": "integer",
          "description": "Global seed used for determinism (must be 42)"
        },
        "execution_date": {
          "type": "string",
          "format": "date-time"
        },
        "execution_id": {
          "type": "string",
          "pattern": "^run_[a-z0-9]+_stage_[0-9AB]+$"
        }
      }
    },
    "stages": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "1B": { "$ref": "#/definitions/stage" },
        "3B": { "$ref": "#/definitions/stage" },
        "8B": { "$ref": "#/definitions/stage" },
        "70B": { "$ref": "#/definitions/stage" }
      }
    },
    "audit_trail": {
      "type": "object",
      "properties": {
        "reproducibility_checksum": {
          "type": "string",
          "pattern": "^sha256:[a-f0-9]{64}$"
        }
      }
    }
  },
  "definitions": {
    "stage": {
      "type": "object",
      "required": ["manifest_fingerprint", "indices", "statistics"],
      "properties": {
        "manifest_fingerprint": {
          "type": "string",
          "pattern": "^sha256:[a-f0-9]{64}$"
        },
        "indices": {
          "type": "array",
          "items": { "type": "integer", "minimum": 0 },
          "description": "Sorted indices into dataset"
        },
        "statistics": {
          "type": "object",
          "properties": {
            "total_samples": { "type": "integer" },
            "band_distribution": {
              "type": "object",
              "properties": {
                "B0": { "type": "integer" },
                "B1": { "type": "integer" },
                "B2": { "type": "integer" },
                "B3": { "type": "integer" },
                "B4": { "type": "integer" },
                "B5": { "type": "integer" }
              }
            },
            "modality_distribution": {
              "type": "object",
              "properties": {
                "text": { "type": "integer" },
                "code": { "type": "integer" }
              }
            }
          }
        }
      }
    }
  }
}
```

---

### Task 3: Deterministic Ordering Enforcement

#### 3.1 Create `determinism.py`
```python
# experiments/3_coreset_engineering/src/reproducibility/determinism.py
from typing import List, Set, Dict, Any
from functools import wraps

class DeterminismError(Exception):
    """Raised when non-deterministic operation is detected."""
    pass

def enforce_sorted_iteration(func):
    """Decorator: Ensure all list/set iterations are sorted."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Note: This is a pattern/guideline; actual enforcement in code review
        return func(*args, **kwargs)
    return wrapper

def sorted_items(items: Any, key=None) -> List:
    """Always use this to iterate over sets/dicts for determinism."""
    if isinstance(items, dict):
        return sorted(items.items(), key=lambda x: x[0])
    elif isinstance(items, set):
        return sorted(items)
    else:
        return sorted(items, key=key) if key else sorted(items)

# Example: Band assignment (always sorted)
def assign_bands_deterministic(samples):
    """Assign bands to samples in deterministic order."""
    sorted_samples = sorted_items(samples, key=lambda x: x['id'])
    for sample in sorted_samples:
        band = compute_band(sample['difficulty_score'])
        sample['band'] = band
```

---

### Task 4: Replay Verification Test Scripts

#### 4.1 Create `test_replay.py`
```python
# experiments/3_coreset_engineering/tests/test_replay.py
import pytest
import json
import hashlib
from pathlib import Path

class TestReplayVerification:
    """Test that manifests can be replayed identically."""
    
    def test_deterministic_manifest_generation(self):
        """Same input + seed → identical manifest."""
        from coreset_engine.builder import CoresetBuilder
        
        # Run 1: Generate manifest
        builder_1 = CoresetBuilder(seed=42)
        manifest_1 = builder_1.build_manifest(dataset)
        fingerprint_1 = compute_fingerprint(manifest_1['1B']['indices'])
        
        # Run 2: Regenerate with same seed
        builder_2 = CoresetBuilder(seed=42)
        manifest_2 = builder_2.build_manifest(dataset)
        fingerprint_2 = compute_fingerprint(manifest_2['1B']['indices'])
        
        # Fingerprints must match exactly
        assert fingerprint_1 == fingerprint_2
    
    def test_manifest_schema_compliance(self, manifest_json_path):
        """Manifest complies with canonical schema."""
        import jsonschema
        
        with open(manifest_json_path) as f:
            manifest = json.load(f)
        
        with open('configs/manifest_schema.json') as f:
            schema = json.load(f)
        
        # Should not raise
        jsonschema.validate(instance=manifest, schema=schema)
    
    def test_replay_indices_are_sorted(self, manifest_json_path):
        """All indices in manifest must be sorted."""
        with open(manifest_json_path) as f:
            manifest = json.load(f)
        
        for stage_name, stage_data in manifest['stages'].items():
            indices = stage_data['indices']
            assert indices == sorted(indices), f"Indices not sorted in stage {stage_name}"
    
    def test_fingerprint_computation(self):
        """Test fingerprinting algorithm."""
        indices = [0, 5, 10, 15, 20]
        fingerprint = compute_fingerprint(indices)
        
        # Re-compute: should be identical
        fingerprint_2 = compute_fingerprint(indices)
        assert fingerprint == fingerprint_2
        
        # Different indices: different fingerprint
        fingerprint_3 = compute_fingerprint([0, 5, 10, 15, 21])
        assert fingerprint != fingerprint_3

def compute_fingerprint(indices):
    """Compute SHA256 fingerprint of indices."""
    json_str = json.dumps(sorted(indices), separators=(',', ':'))
    return hashlib.sha256(json_str.encode()).hexdigest()
```

---

### Task 5: CI Regression Tests

#### 5.1 Update `.github/workflows/coreset-deploy.yml`
```yaml
# Add reproducibility checks to pipeline
name: Coreset Engineering - Build, Test, Deploy

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      # ... existing steps ...
      
      # New: Reproducibility tests
      - name: Run Reproducibility Tests
        run: |
          uv run pytest experiments/3_coreset_engineering/tests/test_replay.py -v
      
      # New: Manifest schema validation
      - name: Validate Manifest Schema
        run: |
          python experiments/3_coreset_engineering/scripts/validate_manifests.py
      
      # New: Seed policy validation
      - name: Check Seed Policy Compliance
        run: |
          python experiments/3_coreset_engineering/scripts/check_seed_policy.py
      
      # New: Determinism regression
      - name: Run Determinism Regression Tests
        run: |
          uv run pytest experiments/3_coreset_engineering/tests -k determinism -v
```

---

### Task 6: Audit Tooling

#### 6.1 Create `audit_tool.py`
```python
# experiments/3_coreset_engineering/scripts/audit_manifest.py
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any

class ManifestAuditor:
    """Audit manifests for reproducibility and compliance."""
    
    def __init__(self, manifest_path: str):
        with open(manifest_path) as f:
            self.manifest = json.load(f)
    
    def audit(self) -> Dict[str, Any]:
        """Run comprehensive audit."""
        return {
            'schema_compliance': self.check_schema(),
            'seed_correctness': self.check_seed(),
            'fingerprint_validity': self.check_fingerprints(),
            'index_sorting': self.check_sorted_indices(),
            'determinism_markers': self.check_determinism_markers(),
        }
    
    def check_schema(self) -> bool:
        """Verify manifest schema compliance."""
        required = ['schema_version', 'metadata', 'stages']
        return all(key in self.manifest for key in required)
    
    def check_seed(self) -> bool:
        """Verify seed is set to canonical value (42)."""
        return self.manifest['metadata']['seed'] == 42
    
    def check_fingerprints(self) -> bool:
        """Verify manifest fingerprints are correct."""
        for stage_name, stage in self.manifest['stages'].items():
            indices = stage['indices']
            computed_fp = hashlib.sha256(
                json.dumps(sorted(indices)).encode()
            ).hexdigest()
            if stage['manifest_fingerprint'] != f"sha256:{computed_fp}":
                return False
        return True
    
    def check_sorted_indices(self) -> bool:
        """Ensure all indices are strictly sorted."""
        for stage_name, stage in self.manifest['stages'].items():
            indices = stage['indices']
            if indices != sorted(indices):
                return False
        return True
    
    def check_determinism_markers(self) -> bool:
        """Verify audit trail indicates deterministic execution."""
        audit = self.manifest.get('audit_trail', {})
        return 'reproducibility_checksum' in audit

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest_path')
    args = parser.parse_args()
    
    auditor = ManifestAuditor(args.manifest_path)
    result = auditor.audit()
    
    print(json.dumps(result, indent=2))
    
    if all(result.values()):
        print("✓ Manifest audit passed!")
        exit(0)
    else:
        print("✗ Manifest audit failed!")
        exit(1)
```

---

## Checklist: L2 Task Completion

- [ ] **Config + Seed Policy**
  - [ ] Created `seed_policy.yaml` with canonical seed (42)
  - [ ] Implemented `SeedPolicy` class enforcing single initialization
  - [ ] Updated curriculum.yaml with schema_checksum

- [ ] **Manifest JSON Schema**
  - [ ] Created `manifest_schema.json` with all required fields
  - [ ] Documented fingerprinting algorithm
  - [ ] Added jsonschema validation to tests

- [ ] **Deterministic Ordering**
  - [ ] Created `determinism.py` with sorting enforcement patterns
  - [ ] Updated sampling code to use `sorted()` everywhere
  - [ ] Code review: confirmed no set/dict iterations without sorting

- [ ] **Replay Test Scripts**
  - [ ] Created `test_replay.py` with determinism tests
  - [ ] Tests verify identical input + seed → identical output
  - [ ] Added schema compliance validation

- [ ] **CI Automation**
  - [ ] Updated `.github/workflows/coreset-deploy.yml` with reproducibility checks
  - [ ] Added manifest validation step
  - [ ] Added seed policy compliance check

- [ ] **Audit Tooling**
  - [ ] Implemented `ManifestAuditor` class
  - [ ] Created audit script checking 5 categories
  - [ ] Integrated with CI/CD pipeline

---

**Status:** Policy document complete  
**Implementation:** See tasks 1-6 above  
**Approval:** Requires Team 2 (Curriculum) + Team 19 (Reproducibility) sign-off
