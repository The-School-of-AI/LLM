
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

