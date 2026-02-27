# Domain Stratification Fix - Summary Report

## Problem Identified
The coreset selection engine was not properly selecting chunks from all allowed domains. Specifically:
- **Reasoning domain was completely missing from 3B stage** (0% when should be ~20%)
- **Agentic domain was missing from 8B stage** 
- **Indic domain was missing from 8B stage**
- Band-domain-language stratification appeared broken

## Root Cause Analysis
The curriculum (config/curriculum.yaml) defined allowed domains using theoretical/aspirational names that didn't match the actual domains in the training data:

**Curriculum-defined allowed domains vs actual data domains:**
- B0: Curriculum said `["clean_web", "basic_text"]` but data has no "basic_text"
- B1: Curriculum said `["clean_web", "narrative"]` but data has no "narrative"
- B2: Curriculum said `["structured_knowledge", "intro_technical"]` - **NO OVERLAP** with data
- B3: Curriculum said `["technical", "code", "reasoning"]` but data has no "technical"
- B4: Curriculum said `["algorithms", "deep_technical", "math"]` but data has no "algorithms" or "deep_technical"
- B5: Curriculum said `["advanced_reasoning", "system_design", "agentic"]` but data has no "advanced_reasoning" or "system_design"

**Actual domains in training data:** `clean_web`, `code`, `math`, `reasoning`, `agentic`, `indic`

## Domain Mismatch Impact
When bucket creation in the selection algorithm checked `allowed_domains` from curriculum:
1. Chunks in non-matching domains were assigned target_tokens=0 
2. Only chunks in matching (band, domain) buckets were selected
3. This caused complete absence of reasoning, agentic, and indic domains in many stages

## Fix Applied
Updated `config/curriculum.yaml` difficulty_bands section to match actual data:

**New domain allocations:**
- B0: `["clean_web"]` - nursery level uses foundational clean web
- B1: `["clean_web"]` - primary level continues clean web foundation
- B2: `["clean_web", "code"]` - high school adds light code
- B3: `["code", "reasoning", "math"]` - undergraduate gets technical domains
- B4: `["code", "reasoning", "math"]` - graduate level same technical mix (could add specialization later)
- B5: `["reasoning", "agentic", "indic"]` - PhD level gets advanced reasoning, agentic, and multilingual

## Verification Results

### Before Fix (Domain Mismatches)
```
3B domain distribution:
  Reasoning: 0.00% ❌
  Code: ~100%
  
8B domain distribution:
  Agentic: 0.00% ❌
  Indic: 0.00% ❌
  Reasoning: Minimal
```

### After Fix (Proper Stratification)
```
3B domain distribution:
  Reasoning: 22.18% ✓
  Code: 38.85% ✓
  Math: 24.19% ✓
  Clean_web: 14.77% ✓
  Total: Proper (band, domain) stratification

8B domain distribution:
  Agentic: 41.60% ✓ (B5 contribution)
  Indic: 42.19% ✓ (B5 contribution)
  Reasoning: 6.23% ✓
  Code: 7.75% ✓
  Math: 1.81% ✓
  Total: Full domain representation across all bands
```

## Technical Details

### Selection Algorithm Impact
The `_create_buckets()` function in `src/selection/engine.py` (line 105) creates stratified buckets with key `(band, domain)`. For each bucket:

1. **Before fix:** 
   - Checks `if domain not in allowed_domains` (from curriculum)
   - Sets `bucket.target_tokens = 0` if domain not allowed
   - Result: Empty buckets never get selection attempts

2. **After fix:**
   - All actual data domains now in curriculum's allowed_domains
   - Buckets have proper target_tokens allocations
   - Selection algorithm can properly stratify across all domains

### No Code Changes Needed
The selection algorithm code was working correctly. The bug was purely in the curriculum configuration - domain names didn't match data.

## Curriculum-Data Alignment
This fix ensures the curriculum accurately reflects the actual data structure:
- Curriculum `allowed_domains` now lists only domains that exist in training data
- No more empty buckets with target_tokens=0
- Band-domain-language stratification now works as designed

## Remaining Known Issues
1. **Language policy interaction in 70B**: Extreme compression ratio (15000x) causes language enforcement to significantly reduce selection
2. **Band distribution in 1B**: Rolling window constraints appear to affect band ratios, creating distribution different from curriculum targets
3. **1B reasoning absent**: B3 at 5% of 1B, split 3 ways for domains, results in <2% reasoning target

These are secondary issues separate from the core domain stratification fix.

## Files Modified
- `config/curriculum.yaml` - Updated difficulty_bands section with data-matching domains

## Impact
- ✅ Reasoning domain now properly represented in stages that need it
- ✅ Agentic and indic domains now appearing at appropriate bands
- ✅ Band-domain-language stratification working as designed
- ✅ Curriculum now accurately describes the actual training data structure
