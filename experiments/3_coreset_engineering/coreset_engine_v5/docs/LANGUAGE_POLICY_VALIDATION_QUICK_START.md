# Language Policy Validation - Quick Start

## Overview

This guide explains how to validate language policy compliance in generated coresets using the enhanced validation framework.

## Language Policy Basics

The curriculum defines language constraints for each stage:

```yaml
language_policy:
  definition_method: "hard_cap_with_stage_gating"
  primary_languages:
    - lang: "en"
      max_share: 0.92          # English: maximum 92% of tokens
  secondary_languages:
    - lang: "hi"
      max_share: 0.08          # Hindi: maximum 8% of tokens
      earliest_stage: "1B"     # Hindi allowed from 1B onwards
  excluded_languages:          # These languages must NOT appear
    - "zh"    # Chinese
    - "ja"    # Japanese
    - "ko"    # Korean
    - "fr"    # French
    - "de"    # German
    - "es"    # Spanish
  violation_action: "DROP_SAMPLE"
```

## Quick Commands

### Generate Validation Report with Language Metrics

```bash
# Generate reports for all stages
python tools/validate_coreset_outputs.py --stages 1B 3B 8B 70B --format both

# Check only language compliance
python tools/validate_coreset_outputs.py --stages 1B --format report | grep -A 20 "LANGUAGE POLICY"
```

### Interpret Results

**Perfect compliance:**
```
Primary languages:
  Compliant: 1/1
  Violations: (none)

Secondary languages:
  Compliant: 1/1
  Violations: (none)

Compliance Score: 100/100 (Excellent)
```

**With violations:**
```
Primary languages:
  Compliant: 0/1
  Violations:
    en: 0.95 (max: 0.92, excess: 0.03)

Compliance Score: 75/100 (At threshold)
```

This means:
- English share is 95% in the coreset
- Policy allows max 92% (+ 1% tolerance = 93%)
- Actual exceeds limit by 3% (95% - 92%)

### Key Metrics Explained

| Metric | Meaning | Good Value |
|--------|---------|-----------|
| `Excluded found` | Count of disallowed languages | 0 |
| `Unrecognized` | Languages not in policy | 0 |
| `Primary compliant` | Ratio like 1/1 | All pass (denominator=1) |
| `Secondary compliant` | Ratio like 1/1 | All pass (denominator=1) |
| `Compliance score` | 0-100 overall | ≥ 75 |

## Common Issues & Solutions

### Issue: English Share Exceeds 92%

**Symptom:**
```
Primary languages:
  Compliant: 0/1
  Violations:
    en: 0.94 (max: 0.92, excess: 0.02)
```

**Causes:**
1. Input data has higher English ratio than 92%
2. SelectionEngine not properly filtering chunks
3. Rounding effects in selection process

**Solutions:**
1. Check input data distribution: `python tools/check_lang_policy.py`
2. Review SelectionEngine logs for dropped chunks
3. Verify curriculum.yaml language_policy is loaded correctly

### Issue: Hindi Not Present (Empty Secondary)

**Symptom:**
```
Secondary languages:
  Compliant: 0/1
  Violations:
    hi: 0.00 (max: 0.08, excess: 0.00)
```

**Causes:**
1. Hindi data not in input dataset
2. SelectionEngine removed all Hindi chunks
3. Stage constraints prevent Hindi (e.g., using pre-1B stage)

**Solutions:**
1. Verify Hindi data exists: `ls data/datasets/ | grep -i hi`
2. Check earliest_stage constraint: should allow stage you're generating
3. Run with relaxed constraints to debug: update curriculum.yaml temporarily

### Issue: Excluded Language Found

**Symptom:**
```
Excluded languages found: 1

Violations:
  zh: 0.02 (language: Chinese in excluded list)
```

**Causes:**
1. Language detection incorrectly identified language
2. Input data contains disallowed language
3. SelectionEngine failed to filter excluded languages

**Solutions:**
1. Verify language detection: check ChunkMetadata.language field
2. Ensure curriculum.yaml excluded_languages list is complete
3. Check SelectionEngine logs for excluded language handling

## Debugging with Validator API

```python
from tools.validate_coreset_outputs import CoresetValidator

# Initialize validator
validator = CoresetValidator(
    curriculum_path="config/curriculum.yaml",
    output_base_dir="output/coresets"
)

# Validate specific stage
report = validator.validate_stage("1B")

# Access language metrics directly
lang_metrics = report.language_metrics
print(f"Excluded languages: {lang_metrics['excluded_found']}")
print(f"Primary violations: {lang_metrics['primary_violations']}")
print(f"Secondary violations: {lang_metrics['secondary_violations']}")

# Check individual violations
for violation in lang_metrics['primary_violations']:
    print(f"{violation['language']}: {violation['actual_share']:.2%} (max: {violation['max_share']:.2%})")
```

## Validation Tolerance

The validator uses **1% tolerance** for max_share constraints to account for:
- Token boundary effects
- Rounding in calculations
- Floating-point precision

**Example:**
- Max share: 92%
- Tolerance: 1%
- Passes if actual ≤ 93%
- Fails if actual > 93%

## Compliance Score Breakdown

Total: **100 points** distributed as:
- **25 points:** No excluded languages found
- **25 points:** No unrecognized languages
- **25 points:** All primary languages compliant
- **25 points:** All secondary languages compliant

**Pass threshold: 75/100**

**Score Interpretation:**
- **100:** Perfect compliance
- **75:** At threshold (acceptable)
- **50:** Missing 2 categories (concerning)
- **25:** Missing 3 categories (critical)
- **0:** Complete failure (all categories violated)

## Language Distribution Example

### Valid Distribution (1B Stage)
```
Language Distribution:
  en: 92%  ✓ (primary, within max 92%)
  hi: 8%   ✓ (secondary, within max 8%, after 1B)

Result: COMPLIANT (100/100)
```

### Invalid Distribution #1 (Exceeds Primary)
```
Language Distribution:
  en: 95%  ✗ (exceeds max 92% + 1% tolerance)
  hi: 5%   ✓ (within max 8%)

Result: COMPLIANT (75/100) - primary violation
```

### Invalid Distribution #2 (Excluded Language)
```
Language Distribution:
  en: 90%  ✓ (within max 92%)
  fr: 5%   ✗ (EXCLUDED language present)
  hi: 5%   ✓ (within max 8%)

Result: VIOLATION (50/100) - excluded language found
```

### Invalid Distribution #3 (Multiple Issues)
```
Language Distribution:
  en: 94%  ✗ (exceeds max 92% + 1% tolerance)
  zh: 3%   ✗ (EXCLUDED language present)
  ja: 2%   ✗ (EXCLUDED language present)
  hi: 1%   ✓ (within max 8%)

Result: VIOLATION (25/100) - multiple critical issues
```

## Running Validation in CI/CD

```bash
#!/bin/bash
# ci_validate_language_policy.sh

# Generate validation reports
python tools/validate_coreset_outputs.py --stages 1B 3B 8B 70B --format both

# Extract compliance scores
echo "Checking language policy compliance..."

for stage in 1B 3B 8B 70B; do
    score=$(grep "Compliance Score:" output/validation_reports/${stage}_verification_report.txt | \
            grep -oE "[0-9]+/100" | cut -d/ -f1)
    
    if [ $score -lt 75 ]; then
        echo "FAILED: Stage $stage has compliance score $score/100 (threshold: 75)"
        exit 1
    else
        echo "PASSED: Stage $stage has compliance score $score/100"
    fi
done

echo "All stages passed language policy compliance checks!"
exit 0
```

## Troubleshooting

### Validator Shows Old Results

The validator reads from saved manifest files. To refresh:

```bash
# Regenerate coresets
python coreset_builder.py --config config/pipeline.yaml --curriculum config/curriculum.yaml

# Then validate
python tools/validate_coreset_outputs.py --stages 1B 3B 8B 70B --format both
```

### Language Policy Not Enforced During Generation

Check SelectionEngine logs and ensure:
1. Curriculum loads successfully: `python -c "from src.curriculum.loader import CurriculumLoader; CurriculumLoader.load_curriculum('config/curriculum.yaml')"`
2. Language policy section is present in curriculum.yaml
3. SelectionEngine._enforce_language_policy is called during selection

### Reports Show "0/100 Compliance"

Likely causes:
1. Manifest file has no language_distribution field
2. Indices file is empty
3. Curriculum not loaded

Debug:
```bash
# Check manifest structure
python -c "import json; print(json.dumps(json.load(open('output/coresets/1B/manifest.json')), indent=2))" | head -50

# Count indices
wc -l output/coresets/1B/selected_indices.jsonl
```

## Next Steps

1. Run validation: `python tools/validate_coreset_outputs.py --stages 1B --format report`
2. Review LANGUAGE POLICY COMPLIANCE METRICS section
3. Address any violations found
4. Re-run validation to confirm fixes

For more details, see [LANGUAGE_POLICY_FIX.md](../LANGUAGE_POLICY_FIX.md)

