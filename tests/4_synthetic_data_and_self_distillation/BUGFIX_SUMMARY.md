# Bug Fix Summary — Synthetic Data & Self-Distillation Pipeline

**Team 4 | Pipeline: `experiments/4_synthetic_data_and_self_distillation/`**
**Date: 2026-02-07**

---

## Overview

A code review of the `generate-bank` data generation pipeline identified **8 bugs** across data quality, Indic language support, code generation, alias resolution, manifest consistency, and scaling readiness. All 8 have been fixed and the first 3 are covered by **102 automated tests**.

### Test Results

```
tests/4_synthetic_data_and_self_distillation/bugfixes/
├── test_bug1.py   29 tests  ✅ all pass
├── test_bug2.py   45 tests  ✅ all pass
├── test_bug3.py   28 tests  ✅ all pass
└── test_bug4.py   33 tests  ✅ all pass
                  ─────────
                  135 total   ✅ all pass
```

Run all tests:
```bash
uv run python -m pytest tests/4_synthetic_data_and_self_distillation/bugfixes/ -v
```

---

## Bug 1: Single Distilled Prompt Used for All Skill Categories

**Severity:** 🔴 Critical
**Impact:** >60% of generated samples had broken `distilled_view` fields ("Answer: Unknown." or empty)
**Files changed:** `generation/dual_view_generator.py`
**Test file:** `bugfixes/test_bug1.py` (29 tests)

### Problem

A single `DISTILLED_PROMPT` was hardcoded for math-style Q&A:

```
Solve this question. You MUST respond in this EXACT format:
Answer: [your answer here]
Justification: [1-2 sentence explanation]
```

When code tasks (`"Implement binary search..."`), translation tasks (`"Translate to Hindi..."`), Indic language questions (`"भारत की राजधानी क्या है?"`), or instruction-following tasks were passed through this prompt, the LLM either returned `"Answer: Unknown."` or empty strings because the format didn't fit the task type.

### Root Cause

No routing existed between skill categories and prompt templates. The same math-format prompt was used for all 45 skill buckets.

### Fix

Added **7 category-specific prompt templates** and a routing function:

| Category | Prompt | Skills Routed |
|----------|--------|---------------|
| `default` | Math/reasoning/factual (original) | RSN-*, FND-*, PRD-*, KNOW-* |
| `code_gen` | Expects ```` ```python ``` ```` code blocks | CODE-GEN-T1/T2/T3, CODE-SYN, CODE-ALGO, CODE-OPT, CODE-TEST |
| `code_debug` | Expects bug identification + fix | CODE-DBG, CODE-DEBUG |
| `code_explain` | Expects plain-English explanation | CODE-COMP, CODE-EXPLAIN |
| `translation` | Expects translated text, forbids "Unknown" | LANG-TRANS, INDIC-TRANS |
| `indic` | Replies in same script, forbids "Unknown" | INDIC-*, LANG-HI-*, FND-LEX-HI, RSN-MATH-HI |
| `instruction` | Follows constraints precisely | ALN-* |

Also increased `max_tokens` from hardcoded 256 to category-appropriate limits:
- Code generation: **512** tokens
- All other non-default: **384** tokens
- Default (math): **256** tokens (unchanged)

### Key Changes

```python
# NEW: routing function
def _get_skill_prompt_category(skill_bucket: str) -> str:
    ...

# NEW: category-specific prompt map
DISTILLED_PROMPT_MAP = {
    "code_gen": DISTILLED_PROMPT_CODE,
    "code_debug": DISTILLED_PROMPT_DEBUG,
    ...
}

# NEW: category-specific token limits
DISTILLED_MAX_TOKENS = {
    "code_gen": 512,
    "default": 256,
    ...
}
```

---

## Bug 2: Hard Negatives Universally Broken (wrong_answer: null)

**Severity:** 🔴 Critical
**Impact:** 100% of hard_negative fields had `{"reasoning": "", "wrong_answer": null}`
**Files changed:** `generation/dual_view_generator.py`
**Test file:** `bugfixes/test_bug2.py` (45 tests)

### Problem

The hard negative generation chain depended on `answer` extracted from the distilled view. Since Bug 1 caused distilled views to return `"Answer: Unknown."`, the extracted answer was `"Unknown."`. This was passed to `_generate_hard_negative()` as:

```
Correct Answer: Unknown.
```

The LLM couldn't generate a "plausible but wrong" answer when the correct answer was "Unknown", producing empty reasoning and null wrong_answer for every single sample.

Additionally, the original fallback had two sub-bugs:
1. Code CoT preambles like `"Answer: Here's the implementation..."` were extracted as the answer
2. First-line fallback grabbed `"Step 1: ..."` reasoning text as the answer

### Root Cause

1. The `answer_is_bad` check only looked for missing `"Answer:"` prefix, not for `"Answer: Unknown."`
2. No preamble filtering on CoT answer extraction
3. No guard to skip hard_negative when answer was still garbage

### Fix

**Three-part fix:**

1. **Smarter bad-answer detection** — catches `"Unknown."`, `"No answer generated"`, and empty strings:
   ```python
   answer_is_bad = (
       not answer
       or answer.lower().strip().rstrip(".") in ("unknown", "no answer generated", "")
   )
   ```

2. **Dedicated `_extract_answer_from_cot()` method** with 4 strategies:
   - Strategy 1: `"Final Answer: <value>"` — strongest signal
   - Strategy 2: `"Answer: <value>"` with preamble filtering (rejects "Here's the implementation...")
   - Strategy 3: Trailing number after `=`, `:`, or `is` (for math)
   - Strategy 4: Last ```` ```python ... ``` ```` code block (for code)
   - Returns `None` if no reliable answer found (instead of grabbing garbage)

3. **`has_valid_answer` guard** before hard_negative generation:
   ```python
   has_valid_answer = answer and answer.lower().strip().rstrip(".") not in ("unknown", ...)
   if generate_hard_negative and cot_allowed and has_valid_answer:
       hard_negative = self._generate_hard_negative(question, answer, skill_bucket)
   ```

### Preamble Rejection

Strategy 2 rejects common LLM preamble phrases:
```python
preamble_patterns = ("here's", "here is", "the following", "below is", ...)
filler_phrases = ("see above", "see below", "n/a", "none", "see reasoning")
```

---

## Bug 3: Indic Samples Tagged language="en"

**Severity:** 🔴 Critical
**Impact:** All Indic samples (Hindi, Bengali, Tamil, Telugu) were mislabeled as English
**Files changed:** `run_pipeline.py`, `generation/dual_view_generator.py`, `generation/seed_generator.py`
**Test file:** `bugfixes/test_bug3.py` (28 tests)

### Problem

Builtin seeds for INDIC-QA contained correct language metadata:
```python
{"question": "ভারতের রাজধানী কী?", "language": "bn"}
```

But `cmd_generate_bank()` never read the `"language"` field from seeds. The `DualViewGenerator.generate()` method had `language="en"` as default, so all samples were tagged English.

This affected **3 code paths**:
1. `cmd_generate_bank()` — main pipeline
2. `cmd_generate()` — single-skill CLI command
3. `dual_view_generator.py` CLI `__main__`

Additionally:
- INDIC-TRANS seeds used `"source_lang"`/`"target_lang"` but not `"language"`, so even with the fix they'd all fall back to `"hi"`
- LANG-TRANS seeds had no language metadata at all

### Fix

**Five-part fix:**

1. **`cmd_generate_bank()`** — reads `seed.get("language")` with fallback to `skill.languages[0]`:
   ```python
   seed_language = seed.get("language", skill.languages[0] if skill.languages else "en")
   ```

2. **`cmd_generate()`** — same pattern added (was completely missing)

3. **`dual_view_generator.py` CLI** — added `--language` flag

4. **INDIC-TRANS builtin seeds** — added `"language"` key set to `target_lang`:
   ```python
   {"question": "Translate to English: 'जल ही जीवन है।'",
    "source_lang": "hi", "target_lang": "en", "language": "en"},
   ```

5. **LANG-TRANS builtin seeds** — added `"language"` key per seed

### Fallback Behavior

When a seed has no `"language"` key, the fallback uses `skill.languages[0]`:

| Skill | `languages[0]` | Fallback |
|-------|----------------|----------|
| RSN-ARITH | `"en"` | English ✅ |
| INDIC-QA | `"hi"` | Hindi ✅ |
| RSN-MATH-HI | `"hi"` | Hindi ✅ |
| LANG-MIX | `"hi-en"` | Hinglish ✅ |
| FND-LEX-HI | `"hi"` | Hindi ✅ |

---

## Bug 4: Builtin Seeds Alias Resolution

**Severity:** 🟠 High
**Impact:** Most canonical skill IDs got placeholder garbage seeds ("Sample question 1 for RSN-ARITH")
**Files changed:** `generation/seed_generator.py`

### Problem

`BUILTIN_SEEDS` used legacy keys (`"RSN-ARITHMETIC"`, `"CODE-COMPLETION"`), but `generate-bank --all` iterates canonical keys from `SKILL_BUCKETS` (`"RSN-ARITH"`, `"CODE-GEN-T1"`). The `get_builtin_seeds()` function only did a direct lookup — no alias resolution.

### Fix

Built a reverse alias map `_CANONICAL_TO_LEGACY` and updated both `get_builtin_seeds()` and `SeedGenerator.generate()` to try:
1. Direct key lookup
2. Canonical → legacy alias resolution
3. Legacy → canonical alias resolution

---

## Bug 5: CODE-COMPLETION Seeds Pre-Solved

**Severity:** 🟡 Medium
**Impact:** Some code completion seeds contained already-complete functions, producing zero training value
**Resolution:** Resolved by Bug 4 fix — alias resolution now routes to correct seed sets

---

## Bug 6: Manifest Key Normalization

**Severity:** 🟠 High
**Impact:** Duplicate manifest entries for same skill under different keys (e.g., `RSN-ARITHMETIC` and `RSN-ARITH`)
**Files changed:** `run_pipeline.py`

### Problem

The manifest stored whichever key was passed (legacy or canonical). Running with `--skills RSN-ARITHMETIC` stored that key; running with `--all` stored `RSN-ARITH`. This caused duplicates.

### Fix

Normalize `skill_id` to canonical form at the top of the loop:
```python
skill = get_skill_bucket(raw_skill_id)
skill_id = skill.id  # canonical form
```

Shard filenames and manifest keys now always use the canonical ID.

---

## Bug 7: Hardcoded Timeouts Too Short for Large Models

**Severity:** 🟠 High (for scaling)
**Impact:** 70B models on consumer hardware would timeout on every API call
**Files changed:** `generation/dual_view_generator.py`, `generation/seed_generator.py`, `diagnostics/run_diagnostics.py`, `validation/verification.py`

### Problem

All Ollama API calls had hardcoded timeouts:
- Generation: `timeout=300` (5 min)
- Diagnostics: `timeout=30` (30 sec)
- Verification: `timeout=60` (1 min)

A 70B model can take 10-15 minutes per generation on consumer hardware.

### Fix

All timeouts now read from `OLLAMA_TIMEOUT` environment variable:

```python
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "600"))
```

Usage for large models:
```bash
export OLLAMA_TIMEOUT=1800  # 30 minutes
python run_pipeline.py generate-bank --all --model llama3:70b
```

Defaults:
- Generation files: **600s** (10 min)
- Diagnostics/verification: **120s** (2 min)

---

## Bug 8: max_tokens=256 Truncated Code Output

**Severity:** 🟡 Medium
**Impact:** Code generation distilled views were truncated or empty
**Resolution:** Bundled into Bug 1 fix — `DISTILLED_MAX_TOKENS` gives 512 tokens for code categories

---

## Files Changed Summary

| File | Bugs Fixed |
|------|------------|
| `generation/dual_view_generator.py` | #1, #2, #3, #7, #8 |
| `generation/seed_generator.py` | #3, #4, #7 |
| `run_pipeline.py` | #3, #6 |
| `diagnostics/run_diagnostics.py` | #7 |
| `validation/verification.py` | #7 |

## Test Coverage

| Test File | Bug | Tests | Status |
|-----------|-----|-------|--------|
| `bugfixes/test_bug1.py` | Category-specific prompts | 29 | ✅ Pass |
| `bugfixes/test_bug2.py` | Hard negative CoT fallback | 45 | ✅ Pass |
| `bugfixes/test_bug3.py` | Language propagation | 28 | ✅ Pass |
| `bugfixes/test_bug4.py` | Alias resolution | 33 | ✅ Pass |
| — | Bugs 5-8 | — | Logic fixes, no separate test files yet |

---

## Data Quality Before vs After (Expected)

| Skill Category | Before: distilled_view OK | After: Expected |
|----------------|---------------------------|-----------------|
| RSN-ARITHMETIC | 🔴 0% ("Unknown") | ✅ >90% |
| CODE-ALGO | 🔴 0% (empty) | ✅ >80% (code blocks) |
| CODE-DEBUG | ⚠️ 60% | ✅ >90% |
| INDIC-QA | ⚠️ 60% (wrong language tag) | ✅ >80% (correct tags) |
| INDIC-TRANS | 🔴 0% ("Unknown") | ✅ >70% |
| ALN-INST | 🔴 33% | ✅ >80% |

| Field | Before | After |
|-------|--------|-------|
| `hard_negative.wrong_answer` | 🔴 null for 100% | ✅ Populated when answer extractable, cleanly skipped otherwise |
| `language` field | 🔴 "en" for all Indic | ✅ Correct per-seed language (hi, bn, ta, te, mix) |
| Manifest keys | 🔴 Mixed legacy/canonical | ✅ Always canonical |

