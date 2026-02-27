# Band Probabilities & Conservative Assignment Policy

## Overview

Each record is annotated with **probabilistic difficulty bands** (`band_p_B0` … `band_p_B5`) that describe the model’s *belief* about where the content lies on the difficulty spectrum.

In addition, a single **`final_band`** is assigned for curriculum execution and gating.

This system intentionally separates:
- **Description** (what the content might be)
- **Execution safety** (where the content is allowed to train)

---

## Probabilistic Band Columns (Descriptive)

For every record, we compute:

```
band_p_B0
band_p_B1
band_p_B2
band_p_B3
band_p_B4
band_p_B5
```

Constraints:
- Each value ∈ [0, 1]
- All probabilities sum to 1
- Probabilities may overlap across multiple bands
- These columns are **descriptive signals**, not hard decisions

---

## Conservative Final Band Assignment

Curriculum configuration:

```yaml
downgrade_on_uncertainty: true
upgrade_never_allowed: true
```

Policy interpretation:

> If a record plausibly belongs to multiple bands, we **always choose the lowest safe band**.

This prevents training on content beyond the model’s current capacity.

---

## Final Band Selection Rule

Let `EPS` be a small credibility threshold.

Default:
```
EPS = 0.10
```

Final band is assigned as:

```python
EPS = 0.10

for band in [B0, B1, B2, B3, B4, B5]:
    if band_p[band] >= EPS:
        final_band = band
        break
```

Properties:
- The **lowest credible band** is always selected
- Upgrading is never allowed
- Ambiguous samples are handled conservatively
- Sharp, confident samples are not unnecessarily penalized

---

## Example Scenarios

### Ambiguous Sample

```
B2: 0.25
B3: 0.45
B4: 0.30
```

Result:
```
final_band = B2
```

---

### High-Confidence Sample

```
B3: 0.92
B4: 0.08
```

Result:
```
final_band = B3
```

---

## How Probabilities Are Used After Downgrade

Even when `final_band` is conservatively lowered, the band probabilities are **not discarded**.

They are used for:
- Curriculum smoothing:
  ```
  E[band] = Σ p(Bi) * i
  ```
- MoE routing analysis and entropy diagnostics
- Anti-spike weighting in early stages
- Coreset stratification
- Identifying regions for synthetic data injection

Only **execution eligibility** is conservative — the difficulty signal remains intact.

---

## Design Rationale

- Ensures **training safety** under uncertainty
- Prevents curriculum cliffs and premature exposure
- Preserves rich difficulty information at zero extra cost
- Aligns naturally with MoE routing dynamics
- Compatible with limited compute and large-scale data

---

## Mental Model

**`final_band` answers:** “Where is this safe to train?”  
**`band_p_*` answers:** “What might this actually be?”

With `downgrade_on_uncertainty: true`, the system is **conservative in action, not blind in understanding**.
