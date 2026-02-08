# Bug Fix Summary — Synthetic Data & Self-Distillation Pipeline

**Team 4 | Pipeline: `experiments/4_synthetic_data_and_self_distillation/`**
**Date: 2026-02-07**

---

## Overview

A code review of the `generate-bank` data generation pipeline identified **8 bugs** across data quality, Indic language support, code generation, alias resolution, manifest consistency, and scaling readiness. All 8 have been fixed and **216 automated tests** verify the fixes across 8 test files.

### Test Results

```
tests/4_synthetic_data_and_self_distillation/bugfixes/
├── test_bug1.py   29 tests  ✅ all pass
├── test_bug2.py   45 tests  ✅ all pass
├── test_bug3.py   28 tests  ✅ all pass
├── test_bug4.py   33 tests  ✅ all pass
├── test_bug5.py   12 tests  ✅ all pass
├── test_bug6.py   24 tests  ✅ all pass
├── test_bug7.py   24 tests  ✅ all pass
└── test_bug8.py   21 tests  ✅ all pass
                  ─────────
                  216 total   ✅ all pass
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
**Impact:** Generated CODE-COMPLETION.jsonl contained 5 samples with COMPLETE functions — LLM responded "already implemented" → zero training value
**Resolution:** Resolved by Bug 4 (alias resolution) and Bug 6 (manifest normalization)
**Files changed:** None directly — resolved via Bug 4 + Bug 6 fixes
**Test file:** `bugfixes/test_bug5.py` (12 tests)

### Problem

The generated `CODE-COMPLETION.jsonl` contained 5 samples where the "question" had **fully complete function implementations** (is_palindrome, binary_search, bubble_sort, are_anagrams, generate_permutations). The LLM responded "The function is already correctly implemented" — zero training value.

### Root Cause Analysis

The initial assumption was that `BUILTIN_SEEDS["CODE-COMPLETION"]` had pre-solved seeds. **This was wrong.** Thorough verification showed:

1. **BUILTIN_SEEDS["CODE-COMPLETION"]** (8 seeds) — **properly incomplete** ✅
   All 8 seeds end mid-line: bare `return`, `return s ==`, `if n > max_val:`, `while left < right:`, etc.

2. **The JSONL data** — came from **LLM-generated seeds**, NOT builtin seeds. `SeedGenerator.generate("CODE-COMPLETION")` used the `SEED_PROMPTS["CODE-COMPLETION"]` prompt which asked for "Partial function definitions to complete" — but the LLM ignored that instruction and generated complete functions as the "question".

### How Bugs 4+6 Resolve It

| Workflow | Before | After |
|---|---|---|
| `generate-bank --all --builtin-seeds` | Used CODE-COMPLETION key → 8 incomplete seeds (fine) | Uses CODE-GEN-T1 direct → 5 "write from scratch" seeds ✅ |
| `generate-bank --all` (LLM seeds) | Used CODE-COMPLETION prompt → LLM gave complete functions | Uses CODE-GEN-T1 prompt → "Generate simple code generation problems" ✅ |
| `generate-bank --skills CODE-COMPLETION` | Used legacy key directly | Bug 6 normalizes to CODE-GEN-T1 → same as above ✅ |

### Key Verification Results

```
CODE-GEN-T1 (canonical) builtin seeds:
  ✅ "Write a Python function to check if a number is even."
  ✅ "Write a Python function to find the maximum of two numbers."

CODE-COMPLETION (legacy) builtin seeds — properly incomplete:
  ✅ "Complete this function:\ndef factorial(n):...    return"     (bare return)
  ✅ "Complete this function:\ndef find_max(numbers):...if n > max_val:"  (no body)
```

### Note

- The 8 CODE-COMPLETION builtin seeds are **orphaned** (only reachable via legacy key) — by design, since CODE-GEN-T1 is a different task type.
- The existing `CODE-COMPLETION.jsonl` in the bank still has broken data and needs **regeneration**.
- The `SEED_PROMPTS["CODE-COMPLETION"]` entry is now dead code (unreachable from canonical paths) but harmless.

---

## Bug 6: Manifest Key Normalization

**Severity:** 🟠 High
**Impact:** Duplicate manifest entries for same skill under different keys; `rebuild-manifest` skipped 15/22 shards; `inject` failed cross-format lookups
**Files changed:** `run_pipeline.py`, `integration/synth_adapter.py`
**Test file:** `bugfixes/test_bug6.py` (24 tests)

### Problem

The manifest stored whatever key was passed — legacy or canonical. Running with `--skills RSN-ARITHMETIC` stored `"RSN-ARITHMETIC"`, while `--all` stored `"RSN-ARITH"`. This caused:
1. **Duplicate entries** for the same skill under different keys
2. **`inject --skills RSN-ARITH`** failing when manifest had `"RSN-ARITHMETIC"`
3. **`rebuild-manifest`** skipping 15 of 22 shard files (legacy filenames like `RSN-ARITHMETIC.jsonl`)

### Root Cause

Five code paths interact with the manifest, but only one was originally fixed:

| Code Path | Original Status |
|---|---|
| `cmd_generate_bank()` | ✅ Fixed (normalized `skill_id = skill.id`) |
| `cmd_status()` | ✅ Read-only, handled aliases in display |
| `cmd_rebuild_manifest()` | 🔴 **Checked `skill_id not in SKILL_BUCKETS` — rejected legacy filenames** |
| `cmd_inject()` | 🔴 **Direct `manifest["skills"]` lookup — no cross-format resolution** |
| `synth_adapter._update_manifest()` | 🔴 **Wrote legacy keys like `"KNOW-FACTUAL_synth"`** |

Verification showed `rebuild-manifest` would reject 15 of 22 shard files:
```
❌ SKIP CODE-COMPLETION — not in SKILL_BUCKETS
❌ SKIP RSN-ARITHMETIC — not in SKILL_BUCKETS
❌ SKIP RSN-ALGEBRA — not in SKILL_BUCKETS
... (15 total skipped)
```

And `inject` would miss skills when formats didn't match:
```
inject --skills RSN-ARITH       -> MISS in manifest (manifest has RSN-ARITHMETIC)
inject --skills CODE-GEN-T1     -> MISS in manifest (manifest has CODE-COMPLETION)
```

### Fix

**Four-part fix across 3 remaining code paths:**

1. **`cmd_generate_bank()`** — already fixed (normalized `skill_id = skill.id`)

2. **`cmd_rebuild_manifest()`** — uses `get_skill_bucket()` to resolve legacy shard filenames:
   ```python
   # OLD: skill_id = shard_path.stem; if skill_id not in SKILL_BUCKETS: SKIP
   # NEW: try to resolve via alias system
   try:
       skill = get_skill_bucket(raw_name)  # handles aliases
       skill_id = skill.id  # canonical form
   except ValueError:
       print(f"  [SKIP] {raw_name} - not a known skill or alias")
       continue
   ```

3. **`cmd_inject()`** — tries 3 lookup strategies:
   ```python
   # Step 1: Direct lookup (raw key)
   # Step 2: Resolve raw as alias → try canonical in manifest
   # Step 3: Reverse lookup — check if any legacy alias of canonical is in manifest
   ```

4. **`synth_adapter._update_manifest()`** — resolves to canonical before writing:
   ```python
   # OLD: entry_key = f"{skill}_synth"  (skill from EXERCISE_TO_SKILL, e.g. "KNOW-FACTUAL")
   # NEW: canonical_skill = get_skill_bucket(skill).id  → "FND-FACT_synth"
   ```

### Verification Results

After fix, all cross-format lookups work:
```
inject --skills RSN-ARITH        -> FOUND (via reverse legacy lookup) ✅
inject --skills RSN-ARITHMETIC   -> FOUND (direct) ✅
inject --skills CODE-GEN-T1      -> FOUND (via reverse lookup to CODE-COMPLETION) ✅
inject --skills CODE-COMPLETION  -> FOUND (via canonical resolution to CODE-GEN-T1) ✅
```

And `rebuild-manifest` now accepts all legacy shard files:
```
✅ RSN-ARITHMETIC.jsonl → RSN-ARITH (via alias)
✅ CODE-COMPLETION.jsonl → CODE-GEN-T1 (via alias)
✅ KNOW-FACTUAL.jsonl → FND-FACT (via alias)
... (all 22 non-synth shards resolve)
```

---

## Bug 7: Hardcoded Timeouts Too Short for Large Models

**Severity:** 🟠 High (for scaling)
**Impact:** 70B models on consumer hardware would timeout on every API call
**Files changed:** `generation/dual_view_generator.py`, `generation/seed_generator.py`, `diagnostics/run_diagnostics.py`, `validation/verification.py`
**Test file:** `bugfixes/test_bug7.py` (24 tests)

### Problem

All Ollama API calls had hardcoded timeouts that were too short for large models:

| File | Old Timeout | Endpoint |
|------|-------------|----------|
| `generation/dual_view_generator.py` | `timeout=300` (5 min) | `/api/chat` |
| `generation/seed_generator.py` | `timeout=300` (5 min) | `/api/chat` |
| `diagnostics/run_diagnostics.py` | `timeout=30` (30 sec) | `/api/generate` |
| `validation/verification.py` | `timeout=60` (1 min) | `/api/generate` |

A 70B model on consumer hardware can take 10-15 minutes per generation. Every API call would timeout.

### Root Cause

Timeouts were hardcoded integer literals inside `urllib.request.urlopen()` calls. No mechanism existed to configure them for different model sizes.

### Fix

**Four-part fix:**

1. **All 4 files** now read `OLLAMA_TIMEOUT` from environment variable with appropriate defaults:

   | File | Default | Reasoning |
   |------|---------|-----------|
   | `dual_view_generator.py` | 600s (10 min) | Generation calls are the longest |
   | `seed_generator.py` | 600s (10 min) | Seed generation can produce long responses |
   | `run_diagnostics.py` | 120s (2 min) | Diagnostic prompts are short |
   | `verification.py` | 120s (2 min) | Verification prompts are short |

2. **Safe env var parsing** — all 4 use `try/except (ValueError, TypeError)` to handle empty or non-numeric `OLLAMA_TIMEOUT`:
   ```python
   try:
       OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "600"))
   except (ValueError, TypeError):
       OLLAMA_TIMEOUT = 600  # safe fallback if env var is empty or non-numeric
   ```

3. **Health check unchanged** — `check_ollama()` in `run_diagnostics.py` keeps `timeout=5` (hardcoded, intentional — it's a quick connectivity test, not a model call).

4. **Clean imports** — moved `import os` in `verification.py` from inline (line 27) to the standard import block at the top of the file.

### Usage for Large Models

```bash
# 30 minutes for 70B models
export OLLAMA_TIMEOUT=1800
python run_pipeline.py generate-bank --all --model llama3:70b

# Custom Ollama host
export OLLAMA_HOST=http://192.168.1.100:11434
python run_pipeline.py diagnose --model llama3:70b
```

### Edge Cases Verified

| Scenario | `OLLAMA_TIMEOUT` env var | Result |
|---|---|---|
| Normal | `"1800"` | 1800 ✅ |
| Unset | not set | Uses default (600 or 120) ✅ |
| Empty string | `""` | Falls back to default ✅ (was crash before fix) |
| Non-numeric | `"abc"` | Falls back to default ✅ (was crash before fix) |
| Float string | `"1.5"` | Falls back to default ✅ (was crash before fix) |

### Calls Audited

| File | `urlopen` calls | All use `OLLAMA_TIMEOUT`? |
|---|---|---|
| `dual_view_generator.py` | 1 | ✅ |
| `seed_generator.py` | 1 | ✅ |
| `run_diagnostics.py` | 2 (generate + health check) | ✅ generate uses OLLAMA_TIMEOUT, health check keeps 5s |
| `verification.py` | 1 | ✅ |
| `proxy_validation.py` | 0 (calls diagnostics indirectly) | N/A ✅ |

---

## Bug 8: max_tokens=256 Truncated Code Output

**Severity:** 🟡 Medium
**Impact:** Code generation distilled views were truncated mid-function or returned empty for all 9 code skills
**Files changed:** `generation/dual_view_generator.py` (bundled into Bug 1 fix)
**Test file:** `bugfixes/test_bug8.py` (21 tests)

### Problem

`_generate_distilled()` used `max_tokens=256` for ALL skill categories. Code generation tasks (implementing functions, algorithms) need 512+ tokens for a complete code block + justification. At 256 tokens, code responses were truncated mid-function, producing broken or empty `distilled_view` fields for all 9 code skills.

### Root Cause

The `max_tokens` parameter was a hardcoded integer literal:
```python
# OLD
response = ollama_chat(..., max_tokens=256, ...)
```

No mechanism existed to vary it by task type.

### Fix

Added `DISTILLED_MAX_TOKENS` dict (bundled into Bug 1 implementation):

```python
DISTILLED_MAX_TOKENS = {
    "code_gen":    512,   # Function/algorithm implementations
    "code_debug":  384,   # Bug identification + fix
    "code_explain": 384,  # Code explanation
    "translation": 384,   # Translated text
    "indic":       384,   # Devanagari/other Indic scripts
    "instruction": 384,   # Instruction-following output
    "default":     256,   # Math/reasoning (short answers, unchanged)
}
```

`_generate_distilled()` now reads the limit dynamically:
```python
category = _get_skill_prompt_category(skill_bucket)
max_tokens = DISTILLED_MAX_TOKENS[category]
```

### Edge Case Found During Review

`LANG-MIX` (Hinglish code-mixing) was routing to `"default"` (256 tokens) instead of `"indic"` (384 tokens). Fixed by adding `"LANG-MIX"` to the indic category in `_get_skill_prompt_category()`.

### Token Allocation Audit (all 45 skills)

| Category | Skills | Token Limit | Old Limit |
|----------|--------|-------------|-----------|
| `code_gen` | CODE-GEN-T1/T2/T3, CODE-ALGO, CODE-SYN, CODE-OPT, CODE-TEST | **512** | 256 |
| `code_debug` | CODE-DBG | **384** | 256 |
| `code_explain` | CODE-COMP | **384** | 256 |
| `translation` | LANG-TRANS, INDIC-TRANS | **384** | 256 |
| `indic` | INDIC-QA/NLI/SENT/NER, LANG-HI-*, FND-LEX-HI, RSN-MATH-HI, LANG-MIX | **384** | 256 |
| `instruction` | ALN-INST/STRUCT/HALL/SAFE/HELP | **384** | 256 |
| `default` | RSN-*, FND-* (non-HI), PRD-*, LANG-GRAMMAR | **256** | 256 |

### Why 256 is Fine for Default Skills

Math/reasoning distilled answers are short:
- `"Answer: 42\nJustification: 3x + 7 = 22, so 3x = 15, x = 5."` (~20 tokens)
- `"Answer: Paris\nJustification: Paris is the capital of France."` (~15 tokens)

The 256 limit was never the bottleneck for these categories.

### Other `max_tokens` Values (Not Changed)

| Call | Value | Location | Rationale |
|------|-------|----------|-----------|
| CoT generation | 1024-2048 | Band spec in `bands.py` | Already large enough ✅ |
| Hard negative | 512 | `_generate_hard_negative()` | Flawed reasoning + wrong answer, not full code ✅ |
| Error correction | 768 | `_generate_error_correction()` | Error ID + explanation + fix ✅ |

---

## Files Changed Summary

| File | Bugs Fixed |
|------|------------|
| `generation/dual_view_generator.py` | #1, #2, #3, #7, #8 |
| `generation/seed_generator.py` | #3, #4, #7 |
| `run_pipeline.py` | #3, #6 (generate-bank, generate, rebuild-manifest, inject) |
| `integration/synth_adapter.py` | #6 (_synth manifest key normalization) |
| `diagnostics/run_diagnostics.py` | #7 |
| `validation/verification.py` | #7 |

## Test Coverage

| Test File | Bug | Tests | Status |
|-----------|-----|-------|--------|
| `bugfixes/test_bug1.py` | Category-specific prompts | 29 | ✅ Pass |
| `bugfixes/test_bug2.py` | Hard negative CoT fallback | 45 | ✅ Pass |
| `bugfixes/test_bug3.py` | Language propagation | 28 | ✅ Pass |
| `bugfixes/test_bug4.py` | Alias resolution | 33 | ✅ Pass |
| `bugfixes/test_bug5.py` | CODE-COMPLETION pre-solved | 12 | ✅ Pass |
| `bugfixes/test_bug6.py` | Manifest key normalization | 24 | ✅ Pass |
| `bugfixes/test_bug7.py` | Configurable timeouts | 24 | ✅ Pass |
| `bugfixes/test_bug8.py` | Token limits for code/indic | 21 | ✅ Pass |

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

