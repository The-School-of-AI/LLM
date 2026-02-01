## Diagnostic Test Design

Setup phase deliverable for #117. References skill buckets from #116.

---

## Objective

Design 25-50 lightweight diagnostic tests to detect model weaknesses on base (non-chat) models before training starts. Tests act as early-warning sensors to guide synthetic data generation and avoid mid-training failures.

---

## Scope & Constraints

Testing buckets 1-8 from #116 only. Buckets 9-14 require instruction-tuning, MoE infrastructure, or alignment - not testable on base models.

Note: #116 has additional buckets in the full taxonomy, but #117 diagnostics are intentionally limited to buckets 1–8 only (base-model testable).

Tests must use continuation/likelihood probes. No instruction-following or chat format.

---

## Relation to Full Skill-Bucket Taxonomy (#116)

This diagnostic suite is intentionally scoped to **Buckets 1–8** only (Foundation + Reasoning + Code), because it must run on **base (non-instruction-tuned) models** using continuation/likelihood probes.

Buckets 9–14 (MoE + Alignment) and the expanded bucket list from the larger taxonomy document are **out of scope** for this phase.

---

## Test Distribution

39 tests across 8 testable skill buckets:

| Skill Bucket                             | Tests | Priority | Rationale                                |
| ---------------------------------------- | ----- | -------- | ---------------------------------------- |
| 5. Mathematical & Quantitative Reasoning | 8     | CRITICAL | GSM8K/MATH alignment, core weakness area |
| 7. Code Generation & Synthesis           | 6     | CRITICAL | HumanEval/MBPP competence check          |
| 6. Code Understanding                    | 4     | HIGH     | Bug detection, syntax comprehension      |
| 4. Logical & Deductive Reasoning         | 6     | HIGH     | Contradiction detection, syllogisms      |
| 3. General Knowledge & Academic          | 4     | MEDIUM   | MMLU-style factual checks                |
| 1. Multilingual LM Competence            | 6     | MEDIUM   | English + Hindi/Devanagari foundation    |
| 2. Long-Context Retention                | 3     | LOW      | Basic context tracking                   |
| 8. Structured Output                     | 2     | LOW      | JSON/format validation                   |

Heavy emphasis on buckets 5 and 7 since GSM8K/HumanEval are primary benchmark targets.

---

## Test Structure

Each test includes:

- Test ID (TEST-XXX)
- Skill bucket mapping to #116
- Type: continuation, likelihood, or consistency
- Prompt: base model context (no instructions)
- Expected: correct completion token(s)
- Failure threshold: P(correct) threshold
- Difficulty band: B3, B4, or B5
- Purpose: capability being probed

---

## How Scoring Works (Likelihood Probe)

For each test, the harness computes **P(expected_token | prompt)** from model logits and compares it to a threshold.

- **Pass:** P(expected_token) >= threshold
- **Fail:** P(expected_token) < threshold

For tests with multiple valid completions, the expected set is treated as an **OR**:
P(correct) = max(P(ans1), P(ans2), ...)

---

## Sample Tests

### Bucket 5: Mathematical Reasoning

**TEST-001: Multi-step arithmetic**

- Type: Continuation
- Band: B3
- Prompt: "A bakery made 240 cookies. They sold 95 in the morning and 63 in the afternoon. Cookies remaining:"
- Expected: " 82" (P > 0.30)
- Failure: P < 0.30 OR calculation error
- Purpose: Sequential arithmetic operations

**TEST-002: Linear equation**

- Type: Continuation
- Band: B4
- Prompt: "Solve: 3x + 7 = 22. Therefore x ="
- Expected: " 5" (P > 0.30)
- Failure: P < 0.30 OR algebraic error
- Purpose: Basic equation solving

**TEST-003: Word problem**

- Type: Continuation
- Band: B3
- Prompt: "Sarah earns 18 per hour. She worked 7 hours Monday, 5 hours Wednesday, 8 hours Friday. Total earnings :"
- Expected: "360" (P > 0.25)
- Failure: P < 0.25 OR calculation error
- Purpose: Word problem to operation translation

**TEST-004: Fraction simplification**

- Type: Continuation
- Band: B3
- Prompt: "Simplify 45/60 to lowest terms:"
- Expected: " 3/4" (P > 0.30)
- Failure: P < 0.30 OR simplification error
- Purpose: Fraction manipulation

**TEST-005: Rate problem**

- Type: Continuation
- Band: B3
- Prompt: "A car travels 65 mph for 3 hours. Distance covered:"
- Expected: " 195" or " 195 miles" (P > 0.30)
- Failure: P < 0.30 OR rate calculation error
- Purpose: Rate × time = distance

**TEST-006: Percentage**

- Type: Continuation
- Band: B3
- Prompt: "Price: 45.50 Taxed at 8% Total cost:"
- Expected: " 49.14" or "49.1" (P > 0.25)
- Failure: P < 0.25 OR percentage error
- Purpose: Percentage calculation

**TEST-007: Age problem**

- Type: Continuation
- Band: B4
- Prompt: "Alice is 12. Bob is 3 times older. Bob's age:"
- Expected: " 36" (P > 0.30)
- Failure: P < 0.30 OR multiplication in context error
- Purpose: Translating linguistic relation to operation

**TEST-008: Two-hop inference**

- Type: Continuation
- Band: B4
- Prompt: "Alice: 50. Bob: 30 more than Alice. Carol: 20 less than Bob. Carol has "
- Expected: " 60" (P > 0.25)
- Failure: P < 0.25 OR comparative chain error
- Purpose: Multi-hop comparative relations

---

### Bucket 7: Code Generation

**TEST-009: Loop trace**

- Type: Continuation
- Band: B3
- Prompt: (code block below)
  ```python
  for i in range(1, 6):
      total += i
  print(total)
  ```
- Expected: " 15" (P > 0.35)
- Failure: P < 0.35 OR trace error
- Purpose: Loop execution understanding

**TEST-010: Function completion**

- Type: Continuation
- Band: B3
- Prompt: (code block below)
  ```python
  def square(x):
      return x *
  ```
- Expected: " x" (P > 0.40)
- Failure: P < 0.40 OR operation error
- Purpose: Function logic completion

**TEST-011: Conditional logic**

- Type: Continuation
- Band: B4
- Prompt: (code block below)
  ```python
  def max_two(a, b):
      if a > b:
          return a
      else:
          return
  ```
- Expected: " b" (P > 0.40)
- Failure: P < 0.40 OR logic error
- Purpose: If-else comprehension

**TEST-012: List operation**

- Type: Continuation
- Band: B4
- Prompt: (code block below)
  ```python
  nums = [1, 2, 3, 4, 5]
  result = sum(nums)  # result = 15
  ```
- Expected: " 15" (P > 0.35)
- Failure: P < 0.35 OR aggregation error
- Purpose: Built-in function understanding

**TEST-013: String operation**

- Type: Continuation
- Band: B3
- Prompt: (code block below)
  ```python
  text = "hello"
  result = text.upper()  # result = "HELLO"
  ```
- Expected: " \"HELLO\"" or " 'HELLO'" (P > 0.30)
- Failure: P < 0.30 OR string method error
- Purpose: String method comprehension

**TEST-014: List indexing**

- Type: Continuation
- Band: B3
- Prompt: (code block below)
  ```python
  items = [10, 20, 30, 40]
  value = items[2]  # value = 30
  ```
- Expected: " 30" (P > 0.40)
- Failure: P < 0.40 OR indexing error
- Purpose: Array indexing understanding

---

### Bucket 6: Code Understanding

**TEST-015: Edge case detection**

- Type: Continuation
- Band: B4
- Prompt: (code block below)

  ```python
  def divide(a, b):
      return a / b # This fails when b = 0
  ```
- Expected: " 0" (P > 0.30)
- Failure: P < 0.30 OR edge case missed
- Purpose: Error case identification

**TEST-016: Type error**

- Type: Continuation
- Band: B4
- Prompt: (code block below)
  ```python
  x = "5"
  y = 3
  z = x + y  # This causes a TypeError
  ```
- Expected: " TypeError" or " error" (P > 0.25)
- Failure: P < 0.25 OR type system misunderstanding
- Purpose: Type mismatch detection

**TEST-017: Logic inversion**

- Type: Continuation
- Band: B4
- Prompt: (code block below)
  ```python
  def is_even(n):
      return n % 2 == 1  # This returns True when n is odd
  ```
- Expected: " odd" (P > 0.30)
- Failure: P < 0.30 OR logic error
- Purpose: Inverted logic detection

**TEST-018: Off-by-one**

- Type: Continuation
- Band: B4
- Prompt: (code block below)

  ```python
  for i in range(10):
      print(i)  # This prints 0 through 9
  ```
- Expected: " 9" (P > 0.35)
- Failure: P < 0.35 OR range misunderstanding
- Purpose: Range boundary understanding

---

### Bucket 4: Logical Reasoning

**TEST-019: Syllogism**

- Type: Continuation
- Band: B3
- Prompt: "All cats are mammals. Felix is a cat. Therefore Felix is a"
- Expected: " mammal" (P > 0.40)
- Failure: P < 0.40 OR logical error
- Purpose: Syllogistic reasoning

**TEST-020: Contradiction detection**

- Type: Continuation
- Band: B4
- Prompt: "The box is empty. The box contains 5 apples. These statements are"
- Expected: " contradictory" or " inconsistent" (P > 0.25)
- Failure: P < 0.25 OR missed contradiction
- Purpose: Logical conflict detection

**TEST-021: Transitive relation**

- Type: Continuation
- Band: B4
- Prompt: "A > B. B > C. Therefore A"
- Expected: " > C" or " is greater than C" (P > 0.30)
- Failure: P < 0.30 OR transitivity error
- Purpose: Transitive property reasoning

**TEST-022: Negation**

- Type: Continuation
- Band: B4
- Prompt: "John is tall. John is not"
- Expected: " short" (P > 0.25)
- Failure: P < 0.25 OR antonym error
- Purpose: Logical negation understanding

**TEST-023: Temporal ordering**

- Type: Continuation
- Band: B4
- Prompt: "Event A happened Monday. Event B was 3 days later. Event B happened on"
- Expected: " Thursday" (P > 0.30)
- Failure: P < 0.30 OR calendar arithmetic error
- Purpose: Multi-step temporal inference

**TEST-024: Spatial reasoning**

- Type: Continuation
- Band: B4
- Prompt: "The key is in the drawer. The drawer is in the desk. The key is in the"
- Expected: " desk" (P > 0.35)
- Failure: P < 0.35 OR inference chain lost
- Purpose: Chaining spatial relations

---

### Bucket 3: General Knowledge

**TEST-025: Basic science**

- Type: Continuation
- Band: B3
- Prompt: "Water freezes at"
- Expected: " 0°C" or " 32°F" (P > 0.40)
- Failure: P < 0.40 OR factual error
- Purpose: Basic scientific fact recall

**TEST-026: Geography**

- Type: Continuation
- Band: B3
- Prompt: "The capital of France is"
- Expected: " Paris" (P > 0.50)
- Failure: P < 0.50 OR factual error
- Purpose: Basic geographic knowledge

**TEST-027: History**

- Type: Continuation
- Band: B3
- Prompt: "World War II ended in"
- Expected: " 1945" (P > 0.40)
- Failure: P < 0.40 OR historical error
- Purpose: Basic historical fact

**TEST-028: Biology**

- Type: Continuation
- Band: B3
- Prompt: "The human body has"
- Expected: " 206 bones" or " 206" (P > 0.25)
- Failure: P < 0.25 OR biological fact error
- Purpose: Biological knowledge

---

### Bucket 1: Multilingual LM

**TEST-029: English vocabulary**

- Type: Continuation
- Band: B3
- Prompt: "The opposite of hot is"
- Expected: " cold" (P > 0.45)
- Failure: P < 0.45 OR vocabulary error
- Purpose: Basic lexical knowledge

**TEST-030: Synonym**

- Type: Continuation
- Band: B3
- Prompt: "Happy means the same as"
- Expected: " joyful" or " glad" (P > 0.35)
- Failure: P < 0.35 OR synonym error
- Purpose: Lexical relations

**TEST-031: Common phrase**

- Type: Continuation
- Band: B3
- Prompt: "It's raining cats and"
- Expected: " dogs" (P > 0.50)
- Failure: P < 0.50 OR idiom error
- Purpose: Idiomatic expression knowledge

**TEST-032: Hindi vocabulary (Devanagari)**

- Type: Continuation
- Band: B3
- Prompt: "गर्म का विलोम है" (opposite of hot is)
- Expected: " ठंडा" (cold) (P > 0.40)
- Failure: P < 0.40 OR vocabulary error
- Purpose: Basic Hindi lexical knowledge

**TEST-033: Hindi-English code-mix**

- Type: Continuation
- Band: B3
- Prompt: "मैं school जा रहा हूं means I am going to"
- Expected: " school" (P > 0.35)
- Failure: P < 0.35 OR code-mix comprehension error
- Purpose: Code-mixed text understanding

**TEST-034: Devanagari numeral**

- Type: Continuation
- Band: B3
- Prompt: "तीन और पांच का योग है" (sum of 3 and 5 is)
- Expected: " आठ" (eight) or " 8" (P > 0.30)
- Failure: P < 0.30 OR arithmetic in Hindi error
- Purpose: Hindi numerical reasoning

---

### Bucket 2: Long Context

**TEST-035: Information retention**

- Type: Continuation
- Band: B3
- Prompt: "Alice lives in Seattle. Bob lives in Portland. Carol lives in Vancouver. Alice lives in"
- Expected: " Seattle" (P > 0.40)
- Failure: P < 0.40 OR retention error
- Purpose: Short-term fact retention

**TEST-036: Multi-fact tracking**

- Type: Continuation
- Band: B4
- Prompt: "Item A costs 10. Item B costs 15. Item C costs 20. Item B costs:
- Expected: " 15" or "15" (P > 0.35)
- Failure: P < 0.35 OR tracking error
- Purpose: Multiple fact tracking

**TEST-037: Context consistency**

- Type: Continuation
- Band: B4
- Prompt: "The meeting is at 3pm. The meeting starts in 2 hours. Current time is"
- Expected: " 1pm" or " 1:00" (P > 0.25)
- Failure: P < 0.25 OR inference from context error
- Purpose: Maintaining temporal consistency

---

### Bucket 8: Structured Output

**TEST-038: JSON completion**

- Type: Continuation
- Band: B4
- Prompt: '{"name": "Alice", "age": 25, "city": "'
- Expected: Valid city name followed by '"}' (P > 0.30)
- Failure: P < 0.30 OR invalid JSON structure
- Purpose: JSON format understanding

**TEST-039: Table reading**

- Type: Continuation
- Band: B4
- Prompt: (table below)

| Name | Age |
| :---: | :-: |
| Alice | 25 |
|  Bob  | 30 |

> What Bob's age?

- Expected: " 30" (P > 0.40)
- Failure: P < 0.40 OR table reading error
- Purpose: Structured data extraction

---

## Evaluation Harness

### Architecture

- `tests/4_synthetic_data_and_self_distillation/diagnostic_tests.json` - 39 test definitions
- `tests/4_synthetic_data_and_self_distillation/run_diagnostic_tests.py` - Evaluation harness
- `tests/4_synthetic_data_and_self_distillation/results/` - Test outputs

### Functionality

1. Load tests from JSON config
2. For each test:
   - Query model API for P(expected_token | prompt)
   - Compare probability vs threshold
   - Record pass/fail, actual probability, model output
3. Aggregate results by skill bucket
4. Generate report

### Metrics

Per Test:

- Pass/Fail based on threshold
- P(expected_token)
- Actual model output

Per Skill Bucket:

- Pass rate percentage
- Average probability across tests
- Common failure patterns

Overall:

- Total pass rate
- Weakest buckets identified
- Readiness for synthetic data injection

---

## Validation Strategy

Phase 1: Baseline Evaluation

- Run 39 tests on early B3 checkpoint
- Record baseline pass rates per bucket
- Identify weak buckets (< 50% pass rate)
- Prioritize buckets 5 and 7 for synthetic data

Phase 2: Synthetic Data Injection

- Generate targeted synthetic data for identified weak buckets
- Inject into training mix at 5-10% cap per stage
- Respect curriculum structure and difficulty bands

Phase 3: Post-Injection Validation

- Re-run same 39 tests on post-injection checkpoint
- Measure delta per bucket
- Check for regressions in non-targeted buckets

Success Criteria:

- +15% pass rate in targeted buckets
- < -5% regression in non-targeted buckets
- Improvements correlate with GSM8K/HumanEval gains

---

## Dependencies

- Team 10: Early B3 checkpoint for baseline run
- #116: Skill bucket taxonomy (complete)

---

## Limitations / Out of Scope

- Not an instruction-following or chat capability evaluation
- MoE-specific behavior (routing/efficiency/think-token) is not measurable via these probes
- Alignment/safety/hallucination resistance diagnostics are excluded in this phase
- Threshold values are initial and will be calibrated after the first baseline run

---

## Open Questions

- Probability extraction API: HuggingFace `generate(..., output_scores=True)` vs direct logit access (depends on infra)
- Threshold calibration after baseline: confirm the 0.25–0.50 ranges
- Multilingual coverage: whether to add 2–3 Hindi variants for Bucket 1 later

---

Moving #117 to In Progress.
