# First Principles Redesign

## What We Actually Need
- Curriculum bands B0-B5 for 3,900 GB of text
- Fast, cheap processing
- Good enough approximation (not perfect classification)
- Runs on very limited credits on AWS for all pending datasets (~$100-150 total)


## Current Approach Problems
❌ 20+ regex per row = CPU hell
❌ Pattern matching for every signal
❌ Trying to be "accurate" when approximate is fine
❌ Processing 100% of data

## New Approach

### 1. STATISTICAL PROXIES (No Regex)
Instead of detecting code with regex, use:
- `character_diversity`: High diversity = code/math (lots of {}, [], (), etc.)
- `punctuation_ratio`: Code/math heavy in punctuation
- `avg_word_length`: Code has longer "words" (variable names)
- `line_length_variance`: Code has high variance, prose is uniform
- `uppercase_ratio`: Code has camelCase, constants

### 2. WORD-BASED HEURISTICS (Fast Lookups)
Instead of complex patterns:
- **Code keywords**: Count("def", "function", "class", "import", "return")
- **Math terms**: Count("theorem", "proof", "equation", "integral")
- **Reasoning words**: Count("therefore", "thus", "because", "implies")

### 3. SAMPLING STRATEGY
Don't process every character:
- Process enough signal to get a good estimate
- Skip mega-documents (>100K chars) - usually garbage
- Sample 10% for expensive metrics, extrapolate

### 4. SIMPLE DIFFICULTY SCORE
```
difficulty = (
    0.3 * vocabulary_diversity +
    0.2 * avg_sentence_length +
    0.2 * rare_word_ratio +
    0.15 * structure_score +
    0.15 * specialty_score
)
```

All computable with simple Spark ops.

## Expected Performance
- Current: 20+ regex × 50 operations per row
- New: 5-10 simple operations per row
- **Speed gain: 10-20x faster**
- **Cost: $50-100 instead of $500**

## Things to know: 
- Almost 98% will be moved to stage 3 by design, only very junk/garbage will be rejected out.
- We have data from different sources - ncert/books can have complete text book in one record, cc_head can have small text snippets.. Its important when we add thershold/sample at exact number, those can be very misleading. 
- Also we can get some information from metadata of ncert and or other sources


## Expectations:
- Build fast metrics calculator that can run on very limited credits on AWS (pure PySpark, no regex)
- Clear documentation of the new approach and rationale
- Clear documentation of the new metrics and how they approximate the desired signals
- Clear definition of curriculum bands B0-B5 based on the new metrics
- Guidelines on how to deploy and run the new processing pipeline on AWS with the given budget constraints

## Band Definitions
### B0 — Nursery

*Language fundamentals*

* grammar, syntax, high-frequency constructions
* simple declarative text
* no reasoning traces
* no chain-of-thought
* no agentic artifacts

---

### B1 — Primary

*Fluent everyday language*

* common knowledge
* clean narrative and exposition
* still no explicit reasoning traces
* trivial or illustrative code only

---

### B2 — High School

*Structured knowledge*

* richer topics, explanations, historical or technical exposition
* implicit reasoning allowed (but no explicit chains)
* introductory technical text

---

### B3 — Undergraduate

*Reasoning begins*

* multi-step explanations
* meaningful technical content
* non-trivial code (functions, APIs, documentation)
* **limited, curated reasoning structure** allowed
* chain-of-thought only if high-signal and explicitly gated

---

### B4 — Graduate

*Explicit reasoning*

* math, algorithms, proofs, deep technical text
* controlled chain-of-thought exposure
* harder code and planning-style explanations
* strict quality gating to avoid “high-difficulty garbage”

---

### B5 — PhD

*Maximum trusted complexity*

* hardest reasoning, planning, and abstraction
* advanced code and system-level thinking
* limited agentic traces (tool use, planning logs)
* chain-of-thought **never dominant**, always capped


Let's build it.
