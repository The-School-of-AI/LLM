# Design Decisions Log

This document captures the key design decisions made for the tokenizer selection and reindexing system.

**Date**: 2026-01-31
**Team**: Token Reindexing Lab

---

## Decision Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| ID Reordering Strategy | **Category Blocks** | Provides interpretability and clear head/torso/tail boundaries |
| Frequency Smoothing | **Log Transform** | Balances MoE safety with maintaining frequency signal |
| Special Token Count | **256 Reserved IDs** | Allows for extensive special token vocabulary with room for expansion |
| Dataset Sample Size | **5GB per dataset** | Balance between statistical significance and computational efficiency |
| MoE Routing Validation | **Enabled** | Critical for validating that reindexing doesn't introduce routing bias |

---

## Detailed Decisions

### 1. ID Reordering Strategy: Category Blocks

**Choice**: Category blocks with frequency sorting within each block

**ID Ranges**:
```
0-511:      Special Tokens (256 slots)
512-10,512:   High Frequency (head, ~10k tokens)
10,512-80,512:  Medium Frequency (torso, ~70k tokens)
80,512-128,000: Low Frequency (tail, ~47k tokens)
```

**Rationale**:
- Clear semantic boundaries for downstream use
- Easier to set embedding layer cutoffs (e.g., "use full embeddings for ID < 10,512")
- Provides "junk token" heuristic (ID > 120,000 likely low-quality)
- Still maintains frequency ordering within blocks for MoE safety

**Alternatives Considered**:
- Pure frequency: Too opaque, no clear boundaries
- Reserved special + pure frequency: Better, but less interpretable

**Implementation**: See `config.yaml` → `reindexing` → `category_blocks`

---

### 2. Frequency Smoothing: Log Transform

**Choice**: Log smoothing with temperature = 0.1

**Formula**: `smoothed_freq = log(1 + raw_freq) × 0.1`

**Rationale**:
- Breaks perfect rank-frequency correlation (prevents exact MoE routing based on frequency)
- Preserves relative ordering of high-frequency tokens (most important)
- Compresses tail token frequencies (less important for differentiation)
- Temperature 0.1 provides moderate smoothing (can be increased if correlation still too high)

**Validation Target**: Spearman correlation between new IDs and raw frequencies should be < 0.95

**Alternatives Considered**:
- No smoothing: Too risky for MoE models (perfect correlation)
- Square root: Less aggressive, may not break correlation sufficiently
- Power-law: More tunable, but log is simpler and well-understood

**Implementation**: See `config.yaml` → `reindexing` → `moe_smoothing`

---

### 3. Special Token Count: 256 Reserved IDs

**Choice**: Reserve IDs 0-511 for special tokens (256 slots)

**Allocation**:
- 0-127: Core special tokens (defined in config)
- 128-255: Reserved for immediate expansion
- 256-511: Reserved for future needs

**Currently Defined**: ~64 special tokens

**Categories**:
- Document structure: `<|begin_of_text|>`, `<|end_of_text|>`, `<|chunk_sep|>`
- Chat roles: `<|system|>`, `<|user|>`, `<|assistant|>`
- Code blocks: `<|code_begin|>`, `<|code_end|>`, language tags
- JSON/Tools: `<|json_begin|>`, `<|tool_call|>`, `<|tool_result|>`
- Metadata: Source tags (Wikipedia, GitHub, etc.)

**Rationale**:
- 64 tokens insufficient for modern LLM use cases (tool calling, multi-modal, etc.)
- 256 provides comfortable room for expansion without wasting too much vocab space
- Clean boundary at 512 (power of 2) for implementation efficiency

**Alternatives Considered**:
- 128 tokens: Too restrictive, would need frequent expansion
- 1024 tokens: Wasteful, reduces regular vocab capacity

**Implementation**: See `config.yaml` → `special_tokens` and `reindexing` → `category_blocks` → `special_tokens`

---

### 4. Dataset Sample Size: 5GB per Dataset

**Choice**: 5GB for both IndicCorpV2 and Dolma (10GB total)

**Rationale**:
- Statistically significant for frequency estimation (billions of tokens)
- Computationally feasible (1-2 hours total processing time)
- Reduces risk of dataset-specific artifacts dominating frequency distribution
- Allows room to add more datasets without excessive computation

**Extensibility**:
- Config supports adding arbitrary HuggingFace datasets
- Can easily increase to 10GB or 50GB if needed
- Can add domain-specific datasets (e.g., scientific papers, legal text)

**Sample Size Justification**:
- 5GB text ≈ 1-2B tokens (depending on tokenizer)
- Zipf's law: frequent tokens stabilize quickly, rare tokens always noisy
- Focus on head/torso stability, tail tokens are inherently noisy

**Implementation**: See `config.yaml` → `frequency_analysis` → `datasets`

---

### 5. MoE Routing Validation: Enabled

**Choice**: Track vocabulary skew and validate ID-frequency correlation

**Metrics Tracked**:
1. **Spearman Correlation** (new_id, raw_frequency)
   - Target: < 0.95
   - Measures linear correlation strength
   - Fails if smoothing insufficient

2. **Vocabulary Skew** (Gini coefficient)
   - Expected: 0.7-0.9 (natural language follows Zipf's law)
   - For context, not pass/fail

3. **Shannon Entropy** (frequency distribution)
   - Higher = more uniform
   - For context, not pass/fail

**Rationale**:
- MoE models are sensitive to routing bias
- Frequency-based routing is suboptimal (should be semantic)
- Validation ensures our reindexing doesn't accidentally create routing artifacts
- Provides quantitative measure of smoothing effectiveness

**Action on Failure**:
- If correlation > 0.95: increase smoothing temperature
- If correlation > 0.98: switch to sqrt or power-law smoothing
- Document correlation value for downstream teams (they may want to adjust model routing)

**Implementation**:
- See `config.yaml` → `reindexing` → `validation` → `moe_routing`
- See `src/moe_validation.py`

---

## Future Decision Points

### When to Re-evaluate

1. **Add new dataset**: Re-run frequency analysis with updated data
2. **Target different domain**: May need different category block sizes
3. **MoE validation fails**: Adjust smoothing temperature or method
4. **Discover new special token needs**: Add to config, stay within 256 limit

### Potential Adjustments

| Scenario | Adjustment |
|----------|------------|
| High-frequency block too small | Expand to 20k tokens (512-20,512) |
| Special tokens insufficient | Use 128-511 range, still have 384 slots |
| MoE correlation too high | Increase smoothing temperature to 0.2-0.5 |
| Dataset distribution shifts | Re-run frequency analysis quarterly |

---

## Validation Checklist

Before finalizing a reindexed tokenizer:

- [ ] All 10 candidate tokenizers evaluated
- [ ] Top tokenizer selected based on scorecard
- [ ] Frequency analysis run on both Indic and Code datasets
- [ ] Frequency stats merged correctly
- [ ] Reindexing completed with chosen strategy
- [ ] All validation tests passed (vocab, strings, IDs, special tokens, encode/decode)
- [ ] MoE routing validation passed (correlation < 0.95)
- [ ] Documentation generated (metadata.json, ID_SCHEME.md)
- [ ] Backup of original tokenizer created
- [ ] Git commit with decision rationale

---

## References

- **Configuration**: `config.yaml`
- **Implementation**: `src/id_reindexer.py`, `src/moe_validation.py`
- **Documentation**: `docs/ID_SCHEME.md`, `docs/USAGE.md`
- **Evaluation Results**: `results/evaluation_results.json` (to be generated)

---

## Approval

**Decisions Approved By**: User
**Date**: 2026-01-31
**Implementation Status**: Complete
**Next Steps**: Run evaluation pipeline on candidate tokenizers

---

## Change Log

| Date | Change | Reason |
|------|--------|--------|
| 2026-01-31 | Initial decisions | Project setup |
| - | - | - |

---

**Document Version**: 1.0
**Maintainer**: Token Reindexing Lab Team
