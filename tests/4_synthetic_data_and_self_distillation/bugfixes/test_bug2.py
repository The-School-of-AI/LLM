"""
Test Bug Fix #2: Hard Negative Chain — CoT Answer Fallback
============================================================

BUG (before fix):
    When _generate_distilled() returned "Answer: Unknown." or an empty string,
    the _extract_answer() function would return "Unknown." as the answer.
    This "Unknown." was then passed directly to _generate_hard_negative() as the
    correct_answer parameter, producing the prompt:
        "Correct Answer: Unknown."
    The LLM could not generate a plausible wrong answer when the correct answer
    was "Unknown", resulting in hard_negative fields with:
        {"reasoning": "", "wrong_answer": null, ...}
    for 100% of generated samples.

    Additionally, the original first-line fallback grabbed "Step 1: ..." reasoning
    text as the answer, and code CoT preambles like "Here's the implementation..."
    were extracted as the answer instead of the actual code.

FIX (after fix):
    1. Added answer_is_bad check that catches "Unknown.", "No answer generated", and
       empty strings — not just missing "Answer:" prefix.
    2. Replaced inline regex fallback with _extract_answer_from_cot() method that
       uses 4 strategies in order of reliability:
         Strategy 1: "Final Answer: <value>" — strongest signal
         Strategy 2: "Answer: <value>" with preamble filtering
         Strategy 3: Trailing number after "=", ":", or "is" (math)
         Strategy 4: Last ```python ... ``` code block (code)
    3. Added has_valid_answer guard before _generate_hard_negative() — if no
       reliable answer can be extracted, hard_negative is cleanly skipped (None)
       rather than generated with garbage input.

FILES CHANGED:
    experiments/4_synthetic_data_and_self_distillation/generation/dual_view_generator.py
"""

import sys
from pathlib import Path

# ── Setup path so we can import from experiments/ ──────────────────────
EXPERIMENT_DIR = Path(__file__).resolve().parents[3] / "experiments" / "4_synthetic_data_and_self_distillation"
sys.path.insert(0, str(EXPERIMENT_DIR))

from generation.dual_view_generator import DualViewGenerator


# Instantiate once — we only test static/extraction methods, no Ollama calls needed
_gen = DualViewGenerator.__new__(DualViewGenerator)


# ======================================================================
# TEST 1: Strategy 1 — "Final Answer:" extraction
# ======================================================================

class TestStrategy1FinalAnswer:
    """_extract_answer_from_cot should prefer 'Final Answer:' when present."""

    def test_final_answer_simple_number(self):
        """Extract numeric final answer from math CoT."""
        cot = (
            "Step 1: Calculate 240 / 60 = 4 hours.\n"
            "Step 2: Calculate 180 / 45 = 4 hours.\n"
            "Step 3: Total = 4 + 4 = 8 hours.\n"
            "Final Answer: 8 hours"
        )
        result = _gen._extract_answer_from_cot(cot)
        assert result == "8 hours", f"Got: {result}"

    def test_final_answer_with_currency(self):
        """Extract currency value from Final Answer."""
        cot = "Step 1: ...\nStep 2: ...\nFinal Answer: $197.82"
        result = _gen._extract_answer_from_cot(cot)
        assert result == "$197.82", f"Got: {result}"

    def test_final_answer_hindi(self):
        """Extract Hindi answer from Final Answer."""
        cot = "Step 1: Identify...\nFinal Answer: नई दिल्ली"
        result = _gen._extract_answer_from_cot(cot)
        assert result == "नई दिल्ली", f"Got: {result}"

    def test_final_answer_algebraic(self):
        """Extract algebraic expression from Final Answer."""
        cot = "Step 1: ...\nFinal Answer: x = 4, y = 2"
        result = _gen._extract_answer_from_cot(cot)
        assert result == "x = 4, y = 2", f"Got: {result}"

    def test_final_answer_ignores_filler(self):
        """Should reject filler phrases like 'see above'."""
        cot = "Step 1: ...\nFinal Answer: see above"
        result = _gen._extract_answer_from_cot(cot)
        # Should NOT return "see above"; should fall through to other strategies
        assert result != "see above", f"Should not return filler phrase, got: {result}"


# ======================================================================
# TEST 2: Strategy 2 — "Answer:" with preamble filtering
# ======================================================================

class TestStrategy2AnswerWithPreambleFilter:
    """'Answer:' extraction should reject common LLM preamble phrases."""

    def test_answer_marker_simple(self):
        """Accept a clean 'Answer: 42' line."""
        cot = "Some reasoning...\nAnswer: 42\nMore text"
        result = _gen._extract_answer_from_cot(cot)
        assert result == "42", f"Got: {result}"

    def test_rejects_heres_the_implementation(self):
        """Reject preamble 'Here's the implementation...'."""
        cot = (
            "Answer: Here's the implementation of a function:\n"
            "```python\n"
            "def foo(): return 1\n"
            "```"
        )
        result = _gen._extract_answer_from_cot(cot)
        # Should NOT be the preamble text; should be the code block (Strategy 4)
        assert result is not None, "Should extract code block via Strategy 4"
        assert "Here's the implementation" not in result, f"Preamble leaked: {result}"
        assert "def foo" in result, f"Expected code block, got: {result}"

    def test_rejects_here_is(self):
        """Reject preamble 'Here is the solution...'."""
        cot = "Answer: Here is the solution to the problem.\nThe answer is 5."
        result = _gen._extract_answer_from_cot(cot)
        # Preamble rejected; should fall through to Strategy 3 (trailing number)
        assert result != "Here is the solution to the problem."

    def test_rejects_below_is(self):
        """Reject preamble 'Below is...'."""
        cot = "Answer: Below is the corrected version.\ndef foo(): pass"
        result = _gen._extract_answer_from_cot(cot)
        assert result is None or "Below is" not in result

    def test_accepts_actual_text_answer(self):
        """Accept a real text answer like 'Paris'."""
        cot = "The capital of France...\nAnswer: Paris"
        result = _gen._extract_answer_from_cot(cot)
        assert result == "Paris", f"Got: {result}"

    def test_accepts_code_one_liner(self):
        """Accept a short code answer like 'return n % 2 == 0'."""
        cot = "The bug is...\nAnswer: return n % 2 == 0"
        result = _gen._extract_answer_from_cot(cot)
        assert result == "return n % 2 == 0", f"Got: {result}"


# ======================================================================
# TEST 3: Strategy 3 — Trailing number extraction (math)
# ======================================================================

class TestStrategy3TrailingNumber:
    """Extract last number after '=', ':', or 'is' when no Answer marker exists."""

    def test_trailing_equals(self):
        """'= 42' at end of line."""
        cot = "Step 1: Add.\nStep 2: Multiply.\nThe total = 42"
        result = _gen._extract_answer_from_cot(cot)
        assert result == "42", f"Got: {result}"

    def test_trailing_is(self):
        """'is 15' at end of line."""
        cot = "Step 1: Tom has 12.\nStep 2: Give 5, get 8.\nTom now has is 15"
        # "is 15" at end of a line
        result = _gen._extract_answer_from_cot(cot)
        assert result == "15", f"Got: {result}"

    def test_trailing_colon(self):
        """': 99' at end of line."""
        cot = "Step 1: ...\nResult: 99"
        result = _gen._extract_answer_from_cot(cot)
        assert result == "99", f"Got: {result}"

    def test_negative_number(self):
        """Negative numbers should be extracted."""
        cot = "Step 1: ...\nThe value is -7"
        result = _gen._extract_answer_from_cot(cot)
        assert result == "-7", f"Got: {result}"

    def test_decimal_number(self):
        """Decimal numbers should be extracted."""
        cot = "Step 1: ...\nTotal cost = 49.14"
        result = _gen._extract_answer_from_cot(cot)
        assert result == "49.14", f"Got: {result}"

    def test_fraction(self):
        """Fractions like 3/4 should be extracted."""
        cot = "Step 1: ...\nSimplified = 3/4"
        result = _gen._extract_answer_from_cot(cot)
        assert result == "3/4", f"Got: {result}"


# ======================================================================
# TEST 4: Strategy 4 — Code block extraction
# ======================================================================

class TestStrategy4CodeBlock:
    """Extract the last ```python ... ``` block when other strategies fail."""

    def test_single_code_block(self):
        """Extract a single code block."""
        cot = (
            "Answer: Here's the implementation:\n"
            "```python\n"
            "def square(x):\n"
            "    return x * x\n"
            "```"
        )
        result = _gen._extract_answer_from_cot(cot)
        assert result is not None
        assert "def square" in result
        assert "return x * x" in result

    def test_multiple_code_blocks_returns_last(self):
        """When multiple code blocks exist, return the last (most complete) one."""
        cot = (
            "First attempt:\n"
            "```python\n"
            "def foo(): pass\n"
            "```\n"
            "Corrected:\n"
            "```python\n"
            "def foo():\n"
            "    return 42\n"
            "```"
        )
        result = _gen._extract_answer_from_cot(cot)
        assert result is not None
        assert "return 42" in result
        # Should NOT contain the first (incomplete) block's content only
        assert "pass" not in result

    def test_code_block_without_python_tag(self):
        """Code blocks without 'python' language tag should also be extracted."""
        cot = (
            "Answer: Here's the solution:\n"
            "```\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "```"
        )
        result = _gen._extract_answer_from_cot(cot)
        assert result is not None
        assert "def add" in result


# ======================================================================
# TEST 5: Returns None when no reliable answer exists
# ======================================================================

class TestReturnsNone:
    """_extract_answer_from_cot should return None when nothing extractable."""

    def test_empty_string(self):
        assert _gen._extract_answer_from_cot("") is None

    def test_none_input(self):
        assert _gen._extract_answer_from_cot(None) is None

    def test_whitespace_only(self):
        assert _gen._extract_answer_from_cot("   \n\n  ") is None

    def test_pure_reasoning_no_answer(self):
        """CoT with only reasoning steps and no answer marker or trailing number."""
        cot = (
            "Step 1: Consider the premises.\n"
            "Step 2: Apply modus ponens.\n"
            "Step 3: The conclusion follows logically."
        )
        result = _gen._extract_answer_from_cot(cot)
        # Should return None, NOT "Step 1: ..." (the old bug)
        assert result is None, f"Expected None, got: {result}"

    def test_does_not_grab_step_as_answer(self):
        """Old bug: first line 'Step 1: ...' was used as answer. Must NOT happen."""
        cot = "Step 1: Break down the problem into parts.\nStep 2: Analyze each part."
        result = _gen._extract_answer_from_cot(cot)
        assert result is None or "Step 1" not in str(result), \
            f"Must not grab reasoning step as answer, got: {result}"


# ======================================================================
# TEST 6: answer_is_bad guard — verify bad answers are caught
# ======================================================================

class TestAnswerIsBadGuard:
    """Verify the answer_is_bad logic correctly identifies garbage answers."""

    def _is_bad(self, answer: str) -> bool:
        """Replicate the answer_is_bad check from generate()."""
        return (
            not answer
            or answer.lower().strip().rstrip(".") in ("unknown", "no answer generated", "")
        )

    def test_unknown_is_bad(self):
        assert self._is_bad("Unknown.")

    def test_unknown_no_dot_is_bad(self):
        assert self._is_bad("Unknown")

    def test_unknown_mixed_case_is_bad(self):
        assert self._is_bad("UNKNOWN.")

    def test_no_answer_generated_is_bad(self):
        assert self._is_bad("No answer generated.")

    def test_empty_string_is_bad(self):
        assert self._is_bad("")

    def test_none_is_bad(self):
        assert self._is_bad(None)

    def test_real_number_is_good(self):
        assert not self._is_bad("42")

    def test_real_text_is_good(self):
        assert not self._is_bad("Paris")

    def test_code_is_good(self):
        assert not self._is_bad("def foo(): return 1")

    def test_hindi_is_good(self):
        assert not self._is_bad("नई दिल्ली")


# ======================================================================
# TEST 7: has_valid_answer guard — verify hard_negative is skipped for bad answers
# ======================================================================

class TestHasValidAnswerGuard:
    """Verify the has_valid_answer check that gates _generate_hard_negative()."""

    def _has_valid(self, answer: str) -> bool:
        """Replicate the has_valid_answer check from generate()."""
        return bool(
            answer
            and answer.lower().strip().rstrip(".") not in ("unknown", "no answer generated", "")
        )

    def test_unknown_blocked(self):
        """'Unknown.' should block hard negative generation."""
        assert not self._has_valid("Unknown.")

    def test_empty_blocked(self):
        assert not self._has_valid("")

    def test_none_blocked(self):
        assert not self._has_valid(None)

    def test_real_answer_allowed(self):
        assert self._has_valid("8 hours")

    def test_code_answer_allowed(self):
        assert self._has_valid("```python\ndef foo(): pass\n```")


# ======================================================================
# TEST 8: End-to-end CoT extraction with real-world examples from data
# ======================================================================

class TestRealWorldCoTExamples:
    """Test against actual CoT text seen in the generated JSONL files."""

    def test_rsn_arithmetic_cot(self):
        """From RSN-ARITHMETIC.jsonl — flour recipe problem."""
        cot = (
            "Step 1: Determine the amount of flour needed per serving by dividing "
            "the total flour by the number of servings: 1.5 cups ÷ 6 servings = 0.25 cups per serving.\n"
            "Step 2: Multiply the per-serving amount by the desired number of servings: "
            "0.25 cups/serving × 4 servings = 1 cup.\n"
            "Final Answer: 1 cup of flour is needed for 4 servings."
        )
        result = _gen._extract_answer_from_cot(cot)
        assert result is not None
        assert "1 cup" in result, f"Got: {result}"

    def test_code_algo_cot_with_preamble(self):
        """From CODE-ALGO.jsonl — binary search with 'Answer: Here's the implementation'."""
        cot = (
            "Answer: Here's the implementation of a function to find the floor of the square root:\n"
            "\n"
            "```python\n"
            "def floor_sqrt(n):\n"
            "    if n == 0 or n == 1:\n"
            "        return n\n"
            "    low, high = 0, n\n"
            "    while low <= high:\n"
            "        mid = (low + high) // 2\n"
            "        if mid * mid == n:\n"
            "            return mid\n"
            "        elif mid * mid < n:\n"
            "            low = mid + 1\n"
            "        else:\n"
            "            high = mid - 1\n"
            "    return high\n"
            "```"
        )
        result = _gen._extract_answer_from_cot(cot)
        assert result is not None
        # Should be the code block, NOT the preamble
        assert "Here's the implementation" not in result
        assert "def floor_sqrt" in result

    def test_indic_qa_cot_telugu(self):
        """From INDIC-QA.jsonl — Telugu question about first PM."""
        cot = (
            "Step 1: The question asks for the first Prime Minister of India.\n"
            "Step 2: After India's independence in 1947, Jawaharlal Nehru was chosen.\n"
            "Final Answer: జవహర్ లాల్ నేహురు (Jawaharlal Nehru)"
        )
        result = _gen._extract_answer_from_cot(cot)
        assert result is not None
        assert "Nehru" in result, f"Got: {result}"

    def test_indic_trans_cot_empty(self):
        """From INDIC-TRANS.jsonl — some translations had empty CoT."""
        cot = ""
        result = _gen._extract_answer_from_cot(cot)
        assert result is None

    def test_rsn_algebra_system_of_equations(self):
        """From RSN-ALGEBRA.jsonl — system of equations answer."""
        cot = (
            "Answer: The solution to the system is x = 4 and y = 2.\n"
            "Step 1: Add the two equations to eliminate y:\n"
            "  (2x + y) + (x - y) = 10 + 2 → 3x = 12\n"
            "Step 2: Solve for x: x = 12/3 = 4\n"
            "Step 3: Substitute x = 4: 4 - y = 2 → y = 2\n"
            "Final Answer: x = 4, y = 2"
        )
        result = _gen._extract_answer_from_cot(cot)
        assert result is not None
        # Should prefer "Final Answer" (Strategy 1) over "Answer:" (Strategy 2)
        assert "x = 4" in result and "y = 2" in result, f"Got: {result}"


# ======================================================================
# Run with: python -m pytest tests/4_synthetic_data_and_self_distillation/bugfixes/test_bug2.py -v
# ======================================================================

