# Token ID Allocation Scheme

## Overview

This document describes the frequency-aware token ID allocation scheme used in our tokenizer reindexing system. The ID scheme is designed to be:

1. **Dataset-aware**: Token IDs encode soft inverse frequency from target corpus
2. **MoE-safe**: Log-smoothed frequencies minimize routing skew
3. **Interpretable**: Category blocks provide clear semantics
4. **Extensible**: Reserved ranges for future special tokens

## Core Philosophy

**Token IDs are a data control surface, not a model feature.**

By ordering token IDs based on corpus frequency (with smoothing), we achieve:
- Better memory locality for common tokens
- Reduced MoE routing artifacts (no exact rank = exact frequency correlation)
- Clear boundaries between boilerplate, content, and junk tokens
- Debugging aid (ID ranges indicate token importance)

## ID Allocation Strategy: Category Blocks (Default)

### Block Structure

```
┌─────────────────────────────────────────────────────────┐
│ ID Range         │ Category         │ Description       │
├─────────────────────────────────────────────────────────┤
│ 0 - 255          │ Special Tokens   │ Control tokens    │
│ 256 - 10,000     │ High Frequency   │ Head (top 10%)    │
│ 10,000 - 80,000  │ Medium Frequency │ Torso (middle 50%)│
│ 80,000 - 128,000 │ Low Frequency    │ Tail (bottom 40%) │
└─────────────────────────────────────────────────────────┘
```

### Category Definitions

#### 1. Special Tokens (0-255)

**Purpose**: Reserved for control and metadata tokens

**Allocation**:
- 0-127: Defined special tokens (document structure, chat roles, code blocks, etc.)
- 128-255: Reserved for future expansion

**Characteristics**:
- Fixed IDs (do not change across reindexing)
- Manually assigned (see `config.yaml`)
- High priority (always preserved)

**Examples**:
```
ID 0:  <|begin_of_text|>
ID 1:  <|end_of_text|>
ID 10: <|system|>
ID 11: <|user|>
ID 12: <|assistant|>
ID 20: <|code_begin|>
ID 40: <|json_begin|>
```

See `config.yaml` for complete special token definitions.

---

#### 2. High Frequency (256-10,000)

**Purpose**: Most common tokens in target corpus (top 10%)

**Allocation**: Sorted by log-smoothed frequency (descending)

**Characteristics**:
- Common words: "the", "and", "is", "to", "of"
- Frequent symbols: ",", ".", "(", ")", "{", "}"
- Common subwords: "ing", "ed", "er", "tion"
- High-frequency Devanagari syllables (Indic priority)
- Common code tokens: "def", "class", "return", "import"

**Why this range?**
- ~9,700 slots for frequent tokens
- Represents core vocabulary for most documents
- Likely to be cached in embedding layer
- Critical for efficient tokenization

**Usage Heuristic**: If a token has ID < 10,000, it's a common token that appears frequently in the corpus.

---

#### 3. Medium Frequency (10,000-80,000)

**Purpose**: Regular vocabulary (middle 50% by frequency)

**Allocation**: Sorted by log-smoothed frequency (descending)

**Characteristics**:
- Domain-specific terms
- Less common words and subwords
- Technical vocabulary
- Rare Indic tokens (non-Devanagari)
- Specialized code patterns

**Why this range?**
- ~70,000 slots for regular vocabulary
- Represents the bulk of tokenizer capacity
- Balances coverage vs. efficiency

**Usage Heuristic**: If a token has 10,000 ≤ ID < 80,000, it's regular vocabulary—not super common, but not junk.

---

#### 4. Low Frequency (80,000-128,000)

**Purpose**: Rare tokens and potential junk (bottom 40%)

**Allocation**: Sorted by log-smoothed frequency (descending, but all low)

**Characteristics**:
- Very rare words
- Typos, artifacts, noise
- Ultra-long subwords
- Unusual unicode combinations
- Byte-fallback sequences

**Why this range?**
- ~48,000 slots for tail tokens
- Likely includes filtering artifacts
- May indicate tokenization failures

**Usage Heuristic**: If a token has ID ≥ 80,000, it's rare. If ID ≥ 120,000, it's likely junk or boilerplate.

**Downstream Decision Point**:
- Embedding layer: Consider dropping tail tokens or using compressed representations
- Training: Monitor if high-ID tokens receive gradients (may indicate data quality issues)

---

## Frequency Smoothing: MoE Safety

### Problem

Exact frequency ordering creates perfect correlation: `rank(token) ↔ frequency(token)`

For MoE models, this can cause **routing skew**: routers learn to route based on token frequency rather than semantic content.

### Solution: Log Smoothing

Before ID assignment, we apply log smoothing to raw frequencies:

```
smoothed_freq(token) = log(1 + raw_freq(token)) × temperature
```

Default temperature: 0.1

**Effect**:
- High-frequency tokens: still get low IDs, but not perfectly ordered
- Low-frequency tokens: compressed into smaller ID range
- Breaks exact rank-frequency correlation

**Validation**: Compute Spearman correlation between token ID and raw frequency. Target: ρ < 0.95 (not perfect correlation).

---

## Percentile Bands

For finer-grained analysis and debugging, we provide percentile bands:

| Percentile | Frequency Threshold | Interpretation             |
|------------|---------------------|----------------------------|
| p99        | Top 1%              | Ultra-common tokens        |
| p95        | Top 5%              | Very common tokens         |
| p90        | Top 10%             | Common tokens (head)       |
| p75        | Top 25%             | Above-average frequency    |
| p50        | Top 50%             | Median frequency           |

**Usage**: Downstream teams can use percentile bands to:
- Set embedding layer cutoffs
- Filter low-quality tokens
- Analyze tokenization efficiency

---

## Alternative Strategy: Pure Frequency

**Not default, but available in config.**

```
┌─────────────────────────────────────────────┐
│ ID 0:     Most frequent token               │
│ ID 1:     2nd most frequent                 │
│ ...                                         │
│ ID N-1:   Least frequent token              │
└─────────────────────────────────────────────┘
```

**When to use**:
- Maximum MoE safety (with heavy smoothing)
- Debugging frequency statistics
- Research experiments

**Caveats**:
- No reserved special token range
- Less interpretable (no clear head/torso/tail boundaries)
- Requires careful handling of special tokens

---

## Downstream Usage Guidelines

### For Embedding Layers

```python
# Example: Separate embedding tiers based on ID ranges
head_embeddings = nn.Embedding(10_000, dim=768)  # High frequency, full capacity
torso_embeddings = nn.Embedding(70_000, dim=384)  # Medium frequency, compressed
tail_embeddings = nn.Embedding(48_000, dim=128)   # Low frequency, highly compressed
```

### For Data Filtering

```python
# Example: Flag documents with excessive junk tokens
def check_quality(token_ids):
    junk_threshold = 120_000
    junk_ratio = sum(1 for id in token_ids if id >= junk_threshold) / len(token_ids)
    return junk_ratio < 0.1  # Reject if >10% junk
```

### For Training Monitoring

```python
# Example: Monitor gradient flow by ID range
def analyze_gradients(embedding_grads, token_ids):
    head_grads = [g for id, g in zip(token_ids, embedding_grads) if id < 10_000]
    tail_grads = [g for id, g in zip(token_ids, embedding_grads) if id >= 80_000]

    print(f"Head grad mean: {np.mean(head_grads)}")
    print(f"Tail grad mean: {np.mean(tail_grads)}")
    # Expect head >> tail; if not, may indicate data quality issues
```

---

## Validation Checklist

After reindexing, validate:

- [ ] All special tokens have correct IDs (0-255 range)
- [ ] No ID overlaps or gaps in category blocks
- [ ] High-frequency tokens are in head range (256-10,000)
- [ ] Frequency smoothing applied (check metadata)
- [ ] Encoding/decoding equivalence preserved
- [ ] Token strings unchanged (only IDs changed)
- [ ] Spearman correlation (ID, frequency) < 0.95

Use `validation_suite.py` to automate these checks.

---

## References

- `config.yaml`: Full configuration of ID ranges and special tokens
- `id_reindexer.py`: Implementation of reindexing algorithm
- `validation_suite.py`: Automated validation tests
- `frequency_analyzer.py`: Frequency computation from datasets

---

## FAQ

**Q: Why category blocks instead of pure frequency?**

A: Category blocks provide interpretability and clear boundaries. Downstream teams can make decisions based on ID ranges without needing frequency statistics.

**Q: Can I change the block boundaries?**

A: Yes, edit `config.yaml` → `reindexing` → `category_blocks`. Ensure no overlaps and that total capacity fits within 128k tokens.

**Q: What if my tokenizer has <128k tokens?**

A: Unused ID slots remain unallocated. The reindexer adapts to actual vocab size.

**Q: How do I add more special tokens later?**

A: Add to `config.yaml` → `special_tokens`, using IDs in the 128-255 range. Re-run reindexing to update tokenizer files.

**Q: Does reindexing affect model weights?**

A: **Yes!** If you retrain with reindexed tokenizer, embedding and output layers must be reinitialized or permuted to match new IDs. This is a data-side intervention for new training runs, not a drop-in replacement for existing models.

---

**Document Version**: 1.0
**Last Updated**: 2026-01-31
**Maintainer**: Token Reindexing Lab Team
