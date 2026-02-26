# Curriculum Schema Update - Implementation Summary

## Overview
The coreset engine code has been updated to support the new curriculum schema (v0.4) while maintaining backward compatibility with the old schema (v0.0.1).

## Key Changes

### 1. **Curriculum Loader (`src/curriculum/loader.py`)**

#### New Dataclasses Added:
- `GlobalContract`: Encapsulates global contract, safety defaults, enforcement, and rejection reasons
- `DifficultySystem`: Contains definition method, tokenizer proxy config, difficulty centroids, and floors
- `GrowthSchedule`: Encapsulates stages and stage profiles with modality weights
- `Guardrails`: Wraps rolling window and global caps (CoT, agentic)
- `DomainGrouping`: Domain definitions and band-domain policies

#### Enhanced BandDefinition:
- Added `intent`: Band purpose/description
- Added `allowed_modalities`: List of allowed modalities (general_text, code, math, etc.)
- Added `reasoning_policy`: CoT and agentic policies structured as dict
- Added `constraints`: Tokenizer and other constraints
- Maintained backward compatibility with legacy fields

#### Dual-Schema Loading:
- `_load_old_schema()`: Parses old curriculum format (v0.0.1)
- `_load_new_schema()`: Parses new curriculum format (v0.4)
- Automatic schema detection based on presence of `global_contract` or `language_and_context` keys
- Both schemas map to common internal structures

#### New Helper Methods:
```python
def get_allowed_domains_for_band(band: DifficultyBand) -> List[str]:
    """Get allowed domains supporting both old and new schemas"""
```

### 2. **Selection Engine (`src/selection/engine.py`)**

#### Updated Domain Lookup:
- Changed from direct `band_def.allowed_domains` access to use new helper method
- Enables seamless support for both schema versions
- Supports domain policies from `band_domain_policy` in new schema

```python
# OLD (v0.0.1)
band_def = self.curriculum.bands.get(band)
allowed_domains = band_def.allowed_domains

# NEW (compatible with both)
allowed_domains = self.curriculum.get_allowed_domains_for_band(band)
```

### 3. **Curriculum Config (`config/curriculum.yaml`)**

#### Domain Mapping Update:
Updated `band_domain_policy` to match actual data domains in the dataset:
- `B0`: `["clean_web"]`
- `B1`: `["clean_web"]`
- `B2`: `["clean_web"]`
- `B3`: `["clean_web", "code"]`
- `B4`: `["code", "reasoning"]`
- `B5`: `["code", "reasoning", "math", "agentic", "indic"]`

This aligns with the sample data which contains `clean_web` and `code` domains.

## Schema Comparison

### New v0.4 Advantages:
✓ **Hierarchical organization**: Better structure with `global_contract`, `language_and_context`, `difficulty_system`
✓ **Modalities system**: Explicit support for different content types
✓ **Growth schedule**: Formal stage profiles with band/modality weights
✓ **Guardrails**: Centralized constraints and caps
✓ **Domain grouping**: Explicit domain definitions and band-domain policy
✓ **Global contract**: Clear safety guarantees and rejection reasons

### Migration Path:
1. Code automatically detects schema version
2. Both v0.0.1 and v0.4 curricula load and work identically
3. New features (growth schedule, stage profiles) available when using v0.4
4. No breaking changes for existing code

## Testing

✓ **Old Schema (v0.0.1)**: Successfully loads from `config/curriculum_old.yaml`
✓ **New Schema (v0.4)**: Successfully loads from `config/curriculum.yaml`
✓ **Selection Engine**: Works with both schemas
✓ **Integration Tests**: `TestIntegration::test_selection_using_real_sample` passes
✓ **Pipeline**: `coreset_builder.py` runs successfully with new schema

## Backward Compatibility

- **Zero breaking changes** for existing code using old schema
- **Automatic schema detection** means no code changes needed
- **Legacy fields preserved** in `BandDefinition` for old schema support
- **Helper methods** abstract schema differences from calling code

## Migration Guide

### For Users:
- No action required - both schemas work automatically
- Consider updating to v0.4 schema for new features (stage profiles, modalities)

### For New Features:
To use stage profiles and modality weights (v0.4 only):
```yaml
growth_schedule:
  stages:
    - name: "1B"
      profile: "base"
      params: 1000000000.0
  stage_profiles:
    base:
      band_weights: {...}
      modality_weights: {...}
```

## Files Modified

1. [src/curriculum/loader.py](src/curriculum/loader.py) - +250 lines of schema parsing logic
2. [src/selection/engine.py](src/selection/engine.py) - Updated domain lookup (3 lines changed)
3. [config/curriculum.yaml](config/curriculum.yaml) - Updated domain mappings

## Next Steps

1. **Data alignment**: Ensure actual dataset domains match curriculum definitions
2. **Stage tokens**: Verify stage targets in new schema align with training requirements
3. **Validation**: Run full pipeline on production data to verify behavior
4. **Documentation**: Update team documentation to reference new schema features

## Future Enhancements

- Consider adding schema validation with `pydantic` for type safety
- Add schema version migration utilities for automated v0.0.1 → v0.4 conversion
- Document best practices for new schema features
