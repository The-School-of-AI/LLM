# Team 4: Skill Definitions

**70B MoE Language Model — Synthetic Data & Self-Distillation**

---

## Training Context

### Model Stages

| Stage | Parameters | Tokens | Primary Focus | Phase |
|-------|------------|--------|---------------|:-----:|
| **Stage 1** | 1B | 20B | Language fundamentals, grammar, basic patterns | `[PRE]` |
| **Stage 2** | 3B | 40B | Knowledge acquisition, structured text understanding | `[PRE]` |
| **Stage 3** | 8B | 100B | Reasoning emergence, code foundations | `[PRE]` |
| **Stage 4 (MoE)** | 16B active / 70B total | 240B | Advanced reasoning, specialization, alignment | `[PRE]` `[SFT]` `[RLHF]` |

### Difficulty Bands

| Band | Level | Description | CoT Policy |
|:----:|-------|-------------|------------|
| **B0** | Nursery | Grammar, syntax, high-frequency constructions | ❌ No CoT |
| **B1** | Primary | Fluent everyday language, common knowledge | ❌ No CoT |
| **B2** | High School | Structured knowledge, implicit reasoning | ❌ No CoT |
| **B3** | Undergraduate | Multi-step explanations, meaningful technical content | ⚠️ CoT rare, capped, short |
| **B4** | Graduate | Math, algorithms, proofs, controlled CoT | ✅ CoT allowed, capped |
| **B5** | PhD | Maximum complexity, advanced reasoning, limited agentic | ✅ CoT allowed, never dominant |

### Phase Tags

| Tag | Meaning | Description |
|-----|---------|-------------|
| `[PRE]` | Pretraining | Core capability built during pretraining |
| `[SFT]` | Supervised Fine-Tuning | Capability refined during SFT |
| `[RLHF]` | Reinforcement Learning | Capability aligned during RLHF |
| `[ALL]` | All Phases | Relevant across entire training lifecycle |

### Summary Statistics

| Metric | Value |
|--------|-------|
| **Total Skills** | 8 |
| **Total Tests** | 39 |
| **Critical Skills** | 2 (Skill 5: Math, Skill 7: Code Gen) |
| **High Priority Skills** | 2 (Skill 4: Logic, Skill 6: Code Understanding) |
| **Test Type** | Continuation / likelihood probes (base model) |
| **Synthetic Bands** | B3–B5 only |
| **Scoring** | P(expected_token \| prompt) ≥ threshold → Pass |

---

## Master Skill Table

| # | Skill | Tests | Priority | Phase | Bands | Verifier | Key Benchmarks |
|:-:|-------|:-----:|:--------:|:-----:|:-----:|----------|---------------|
| 1 | [Multilingual LM Competence](#skill-1-multilingual-lm-competence) | 6 | MEDIUM | `[PRE]` | B0–B2 | closed_set | [HellaSwag](https://rowanzellers.com/hellaswag/) |
| 2 | [Long-Context Retention](#skill-2-long-context-retention) | 3 | LOW | `[PRE]` | B3–B5 | retrieval / exact_match | [LongBench](https://github.com/THUDM/LongBench) |
| 3 | [General Knowledge & Academic](#skill-3-general-knowledge--academic) | 4 | MEDIUM | `[PRE]` | B1–B3 | micro_kb | [TriviaQA](https://nlp.cs.washington.edu/triviaqa/), [NQ](https://ai.google.com/research/NaturalQuestions) |
| 4 | [Logical & Deductive Reasoning](#skill-4-logical--deductive-reasoning) | 6 | HIGH | `[PRE]` `[SFT]` | B3–B5 | logic_solver / SAT | [BigBench](https://github.com/google/BIG-bench), [FOLIO](https://github.com/Yale-LILY/FOLIO) |
| 5 | [Mathematical & Quantitative Reasoning](#skill-5-mathematical--quantitative-reasoning) | 8 | **CRITICAL** | `[PRE]` `[SFT]` | B2–B5 | arithmetic + SymPy | [GSM8K](https://github.com/openai/grade-school-math), [MATH](https://github.com/hendrycks/math) |
| 6 | [Code Understanding](#skill-6-code-understanding) | 4 | HIGH | `[PRE]` `[SFT]` | B3–B4 | exec + classify | [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) |
| 7 | [Code Generation & Synthesis](#skill-7-code-generation--synthesis) | 6 | **CRITICAL** | `[PRE]` `[SFT]` | B3–B5 | exec + unit tests | [HumanEval](https://github.com/openai/human-eval), [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) |
| 8 | [Structured Output](#skill-8-structured-output) | 2 | LOW | `[PRE]` | B3–B5 | json_parse / table_parse | Internal |

---

## Skill Definitions

---

### Skill 1: Multilingual LM Competence

English + Hindi/Devanagari language foundation — vocabulary, grammar, idioms, code-mixing.

| Attribute | Value |
|-----------|-------|
| **Tests** | TEST-029 – TEST-034 |
| **Priority** | MEDIUM |
| **Phase** | `[PRE]` |
| **Bands** | B0–B2 |
| **Verifier** | closed_set |

**Tests**

| Test | Band | Prompt | Expected | Threshold |
|------|:----:|--------|----------|:---------:|
| TEST-029 | B3 | "The opposite of hot is" | " cold" | P > 0.45 |
| TEST-030 | B3 | "Happy means the same as" | " joyful" / " glad" | P > 0.35 |
| TEST-031 | B3 | "It's raining cats and" | " dogs" | P > 0.50 |
| TEST-032 | B3 | "गर्म का विलोम है" | " ठंडा" | P > 0.40 |
| TEST-033 | B3 | "मैं school जा रहा हूं means I am going to" | " school" | P > 0.35 |
| TEST-034 | B3 | "तीन और पांच का योग है" | " आठ" / " 8" | P > 0.30 |

**Acceptance Criteria**

| Metric | Target |
|--------|:------:|
| Grammar error rate | < 2% |
| Syntactic validity | > 98% |
| Hindi delta vs English | ≤ 5% |
| Test pass rate | ≥ 4/6 |

**Failure Modes:** Subject-verb disagreement, tense mixing, Hindi gender agreement errors, Devanagari numeral confusion (६↔९).

---

### Skill 2: Long-Context Retention

Fact retention and retrieval across context windows. Synthetic data uses keyed facts at 10%/50%/90% positions with exact-match verification.

| Attribute | Value |
|-----------|-------|
| **Tests** | TEST-035 – TEST-037 |
| **Priority** | LOW |
| **Phase** | `[PRE]` |
| **Bands** | B3–B5 |
| **Verifier** | retrieval / exact_match |

**Tests**

| Test | Band | Prompt | Expected | Threshold |
|------|:----:|--------|----------|:---------:|
| TEST-035 | B3 | "Alice lives in Seattle. Bob lives in Portland. Carol lives in Vancouver. Alice lives in" | " Seattle" | P > 0.40 |
| TEST-036 | B4 | "Item A costs 10. Item B costs 15. Item C costs 20. Item B costs:" | " 15" | P > 0.35 |
| TEST-037 | B4 | "The meeting is at 3pm. The meeting starts in 2 hours. Current time is" | " 1pm" / " 1:00" | P > 0.25 |

**Acceptance Criteria**

| Metric | Target |
|--------|:------:|
| Retrieval accuracy @ 32K | ≥ 85% |
| Position bias | < 10% |
| No catastrophic decay after 24K | true |
| Test pass rate | ≥ 2/3 |

**Failure Modes:** Lost-in-the-middle, recency/primacy bias, context truncation.

---

### Skill 3: General Knowledge & Academic

Factual knowledge recall. Synthetic data uses **closed-world micro-KB** format (not open-world trivia) to reduce contamination.

| Attribute | Value |
|-----------|-------|
| **Tests** | TEST-025 – TEST-028 |
| **Priority** | MEDIUM |
| **Phase** | `[PRE]` |
| **Bands** | B1–B3 |
| **Verifier** | micro_kb |

**Tests**

| Test | Band | Prompt | Expected | Threshold |
|------|:----:|--------|----------|:---------:|
| TEST-025 | B3 | "Water freezes at" | " 0°C" / " 32°F" | P > 0.40 |
| TEST-026 | B3 | "The capital of France is" | " Paris" | P > 0.50 |
| TEST-027 | B3 | "World War II ended in" | " 1945" | P > 0.40 |
| TEST-028 | B3 | "The human body has" | " 206 bones" / " 206" | P > 0.25 |

**Acceptance Criteria**

| Metric | Target |
|--------|:------:|
| Closed-book QA accuracy | ≥ 70% |
| Entity recognition F1 | ≥ 85% |
| Test pass rate | ≥ 3/4 |

**Failure Modes:** Entity confusion (John Adams ↔ John Quincy Adams), temporal anachronism, quantitative hallucination.

---

### Skill 4: Logical & Deductive Reasoning

Syllogisms, contradiction detection, transitive/temporal/spatial inference. Synthetic data verified via logic solvers and SAT evaluators.

| Attribute | Value |
|-----------|-------|
| **Tests** | TEST-019 – TEST-024 |
| **Priority** | HIGH |
| **Phase** | `[PRE]` `[SFT]` |
| **Bands** | B3–B5 |
| **Verifier** | logic_solver / SAT / truth-table |

**Tests**

| Test | Band | Prompt | Expected | Threshold |
|------|:----:|--------|----------|:---------:|
| TEST-019 | B3 | "All cats are mammals. Felix is a cat. Therefore Felix is a" | " mammal" | P > 0.40 |
| TEST-020 | B4 | "The box is empty. The box contains 5 apples. These statements are" | " contradictory" / " inconsistent" | P > 0.25 |
| TEST-021 | B4 | "A > B. B > C. Therefore A" | " > C" | P > 0.30 |
| TEST-022 | B4 | "John is tall. John is not" | " short" | P > 0.25 |
| TEST-023 | B4 | "Event A happened Monday. Event B was 3 days later. Event B happened on" | " Thursday" | P > 0.30 |
| TEST-024 | B4 | "The key is in the drawer. The drawer is in the desk. The key is in the" | " desk" | P > 0.35 |

**Acceptance Criteria**

| Metric | Target |
|--------|:------:|
| Contradiction rate | < 5% |
| Syllogism accuracy | ≥ 85% |
| Valid inference rate | ≥ 90% |
| Test pass rate | ≥ 4/6 |

**Failure Modes:** Affirming the consequent, denying the antecedent, missed contradiction, transitivity/spatial chain break.

---

### Skill 5: Mathematical & Quantitative Reasoning

Arithmetic, algebra, word problems, rate/percentage calculations. **Deterministic verification mandatory** (arithmetic + SymPy). Heaviest test coverage — this is the primary benchmark-alignment skill.

| Attribute | Value |
|-----------|-------|
| **Tests** | TEST-001 – TEST-008 |
| **Priority** | **CRITICAL** |
| **Phase** | `[PRE]` `[SFT]` |
| **Bands** | B2–B5 |
| **Verifier** | arithmetic + SymPy |

**Tests**

| Test | Band | Prompt | Expected | Threshold |
|------|:----:|--------|----------|:---------:|
| TEST-001 | B3 | "A bakery made 240 cookies. They sold 95 in the morning and 63 in the afternoon. Cookies remaining:" | " 82" | P > 0.30 |
| TEST-002 | B4 | "Solve: 3x + 7 = 22. Therefore x =" | " 5" | P > 0.30 |
| TEST-003 | B3 | "Sarah earns 18 per hour. She worked 7 hours Monday, 5 hours Wednesday, 8 hours Friday. Total earnings:" | " 360" | P > 0.25 |
| TEST-004 | B3 | "Simplify 45/60 to lowest terms:" | " 3/4" | P > 0.30 |
| TEST-005 | B3 | "A car travels 65 mph for 3 hours. Distance covered:" | " 195" | P > 0.30 |
| TEST-006 | B3 | "Price: 45.50 Taxed at 8% Total cost:" | " 49.14" | P > 0.25 |
| TEST-007 | B4 | "Alice is 12. Bob is three times as old. Bob's age:" | " 36" | P > 0.30 |
| TEST-008 | B4 | "Alice: 50. Bob: 30 more than Alice. Carol: 20 less than Bob. Carol has" | " 60" | P > 0.25 |

> TEST-007 uses patched wording ("three times as old" not "3 times older") per Generation Plan §2.1.

**Acceptance Criteria**

| Metric | Target |
|--------|:------:|
| Single-step accuracy | ≥ 95% |
| Multi-step accuracy | ≥ 85% |
| GSM8K accuracy | ≥ 75% |
| No systematic carrying errors | true |
| Test pass rate | ≥ 6/8 |

**Failure Modes:** Carrying errors, PEMDAS violations, decimal placement, percentage miscalculation, wrong-question solving, unit confusion.

---

### Skill 6: Code Understanding

Bug detection, type errors, logic inversion, off-by-one — **detection and classification only**. Fix generation is out of scope (requires SFT).

| Attribute | Value |
|-----------|-------|
| **Tests** | TEST-015 – TEST-018 |
| **Priority** | HIGH |
| **Phase** | `[PRE]` `[SFT]` |
| **Bands** | B3–B4 |
| **Verifier** | exec + classify |

**Tests**

| Test | Band | Prompt (abbreviated) | Expected | Threshold |
|------|:----:|---------------------|----------|:---------:|
| TEST-015 | B4 | `def divide(a, b): return a / b # This fails when b =` | " 0" | P > 0.30 |
| TEST-016 | B4 | `x = "5"; y = 3; z = x + y # This causes a` | " TypeError" | P > 0.25 |
| TEST-017 | B4 | `def is_even(n): return n % 2 == 1 # This returns True when n is` | " odd" | P > 0.30 |
| TEST-018 | B4 | `for i in range(10): print(i) # This prints 0 through` | " 9" | P > 0.35 |

**Acceptance Criteria**

| Metric | Target |
|--------|:------:|
| Code explanation accuracy | ≥ 85% |
| Execution trace accuracy | ≥ 85% |
| Test pass rate | ≥ 3/4 |

**Failure Modes:** Missing edge cases, loop miscounting, side effect blindness, recursion confusion.

---

### Skill 7: Code Generation & Synthesis

Loop tracing, function completion, conditionals, list/string operations. **Verification via code execution + golden outputs is mandatory.**

| Attribute | Value |
|-----------|-------|
| **Tests** | TEST-009 – TEST-014 |
| **Priority** | **CRITICAL** |
| **Phase** | `[PRE]` `[SFT]` |
| **Bands** | B3–B5 |
| **Verifier** | exec + unit tests / golden outputs |

**Tests**

| Test | Band | Prompt (abbreviated) | Expected | Threshold |
|------|:----:|---------------------|----------|:---------:|
| TEST-009 | B3 | `total = 0; for i in range(1,6): total += i; print(total)` | " 15" | P > 0.35 |
| TEST-010 | B3 | `def square(x): return x *` | " x" | P > 0.40 |
| TEST-011 | B4 | `def max_two(a,b): if a>b: return a; else: return` | " b" | P > 0.40 |
| TEST-012 | B4 | `nums = [1,2,3,4,5]; result = sum(nums) # result =` | " 15" | P > 0.35 |
| TEST-013 | B3 | `text = "hello"; result = text.upper() # result =` | " \"HELLO\"" | P > 0.30 |
| TEST-014 | B3 | `items = [10,20,30,40]; value = items[2] # value =` | " 30" | P > 0.40 |

> TEST-009 uses patched version with `total = 0` initialization per Generation Plan §2.1.

**Acceptance Criteria**

| Metric | Target |
|--------|:------:|
| HumanEval Pass@1 | ≥ 70% |
| MBPP Pass@1 | ≥ 65% |
| Runtime exception rate | < 5% |
| Test pass rate | ≥ 4/6 |

**Failure Modes:** Off-by-one, missing base case, null handling, infinite loops, timeout from inefficient algorithms.

---

### Skill 8: Structured Output

JSON format completion and tabular data extraction — **pretraining-safe** (continuation probes only, no instruction-following).

| Attribute | Value |
|-----------|-------|
| **Tests** | TEST-038 – TEST-039 |
| **Priority** | LOW |
| **Phase** | `[PRE]` |
| **Bands** | B3–B5 |
| **Verifier** | json_parse / table_parse |

**Tests**

| Test | Band | Prompt | Expected | Threshold |
|------|:----:|--------|----------|:---------:|
| TEST-038 | B4 | `{"name":"Alice","age":25,"city":"Paris"` | `}` | P > 0.30 |
| TEST-039 | B4 | `Name Age\nAlice 25\nBob 30\nBob_age =` | " 30" | P > 0.40 |

> Both tests use patched versions per Generation Plan §2.1 (deterministic expectation for JSON, continuation format for table).

**Acceptance Criteria**

| Metric | Target |
|--------|:------:|
| JSON parse success | ≥ 95% |
| Table extraction accuracy | ≥ 90% |
| Test pass rate | ≥ 1/2 |

**Failure Modes:** Trailing comma, unescaped characters, wrong cell extraction.

---

## Validation Strategy

| Phase | Action | Success Criteria |
|-------|--------|-----------------|
| **Baseline** | Run 39 tests on early B3 checkpoint | Record pass rates; identify weak skills (< 50%) |
| **Injection** | Generate synthetic data for weak skills; inject at 5–10% cap | Respect curriculum caps and rolling-window limits |
| **Post-Injection** | Re-run 39 tests on post-injection checkpoint | **+15%** in targeted skills; **< -5%** regression in others |
| **Correlation** | Compare diagnostic gains to benchmark scores | Improvements align with GSM8K / HumanEval gains |

---

## Document Info

| Field | Value |
|-------|-------|
| **Version** | 3.0 |
| **Team** | Synthetic Data & Self-Distillation (Team 4) |
| **Suite Version** | `diag_suite_v1.1` |
| **Skills** | 8 |
| **Tests** | 39 |
