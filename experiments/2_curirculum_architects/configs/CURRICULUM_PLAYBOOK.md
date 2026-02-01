
# Curriculum Architects: Practical Workflow (No-Proxy-Model Budget)

This provides a reusable implementation (`curriculum_tools.py`) that works across datasets
(Dolma/FineWeb/Sangraha/etc.) without training proxy models.

---

## What is in already (keep)

1) **Signals-first pipeline**
- Early pruning for short documents is good for scale.
- Single entry `extract_signals()` encourages consistency.

2) **Map/Reduce distribution pattern**
- `map_band_tokens()` + `reduce_base_distribution()` is the correct shape for trillion tokens.

3) **Cached syllable counting**
- `@lru_cache` is a good micro-optimization.

4) **Log-parameter capacity scaling**
- Your `model_capacity()` is correct and stable.

---

## What should be improved (and why)

### A) Avoid “modality => band” hard rules
Notebook rules like:
- `has_code => B3`
- `has_agentic => B5`

are useful for quick bootstraps, but they conflate:
- **difficulty** (how conceptually hard)
with
- **modality** (code / math / traces)

**Fix:** keep modality as tags and use them as *floors/caps*,
while computing difficulty via a continuous score and quantiles.

### B) Sentence splitting
`re.split(r"[.!?]+", text)` breaks on:
- abbreviations, decimals, code, URLs

**Fix:** use a more conservative split heuristic (still cheap).

### C) Rare ratio
Counting word-frequency==1 is noisy across languages.
Keep it as a weak feature now, but plan to replace with:
- subword length stats
- or a per-language common-word list

### D) Base distribution completeness
Notebook’s reducer returns only bands that appear.
This makes later optimization brittle.

**Fix:** always return all `B0..B5` keys (missing => 0).

### E) Capacity–difficulty alignment improvement requested
Notebook uses `exp(-|d - c|)` alignment.
That’s a good baseline, but you asked for:
1) **Anchor to target median/quantile**
2) **KL divergence regularization to base distribution**

**Fix:** implement a small optimizer that matches target quantiles (median, p75, etc.)
while staying close to `base_distribution` via KL.

---

## How to use `curriculum_tools.py` in your pipeline

### 1) Calibration pass: compute quantile edges
- sample ~100k docs from all sources
- compute difficulty scores
- compute score quantiles (15/30/50/70/85)

### 2) Production pass: map docs -> band + tokens
- apply edges to assign B0..B5
- reduce into base distribution per dataset or per global corpus

### 3) Stage weights: compute curriculum proportions
- compute target distribution from model capacity
- solve for stage band weights:
  - match target median/quantiles
  - minimize KL(w || base)
  - obey floors/caps

---

## Recommended policy defaults (starting point)

### Floors (prevent forgetting)
- 1B: B0 >= 0.45, B1 >= 0.25
- 3B: B0 >= 0.15
- 8B: B1 >= 0.10
- 70B: B0 >= 0.10

### Caps (prevent instability early)
- 1B: B4 <= 0.01, B5 = 0.00
- 3B: B5 <= 0.01
- 8B: B5 <= 0.03

---

## What’s still intentionally out of scope
- “Best possible” thresholds without validation
- Language-specific readability models
- Advanced quality scoring (KenLM / fastText) – belongs to Data Quality or Coreset team
