"""
diagnostic_tests.py — 50 Lightweight Diagnostic Tests

These are NOT benchmarks. They are internal probes to:
  1. Detect early signs of weakness
  2. Validate synthetic data effectiveness
  3. Run FAST (< 1 second per test)

Each test:
  • Works on base LMs (no instruction-following required)
  • Has clear pass/fail criteria
  • Maps to skill bucket(s)
  • Specifies band level
"""

from dataclasses import dataclass, field
from typing import Callable, Literal
import re


@dataclass
class DiagnosticTest:
    """A single diagnostic test."""
    id: str
    name: str
    purpose: str
    skill_buckets: list[str]  # Which skills this probes
    band: str                  # B0-B5
    
    # Test specification
    prompt: str
    expected: str | list[str]  # Expected output (string or list of acceptable)
    failure_signals: list[str]  # What indicates failure
    
    # Test type
    test_type: Literal[
        "completion",      # Complete the text
        "next_token",      # Predict next token likelihood
        "consistency",     # Check consistency across prompts
        "error_detection", # Detect error in input
        "classification",  # Classify into categories
    ] = "completion"
    
    # Evaluation
    eval_method: Literal[
        "exact_match",     # Exact string match
        "contains",        # Contains expected substring
        "numeric",         # Numeric comparison
        "regex",           # Regex match
        "not_contains",    # Should NOT contain
        "code_valid",      # Valid syntax
    ] = "contains"
    
    # Metadata
    timeout_ms: int = 1000  # Max time in milliseconds
    max_tokens: int = 50    # Max tokens to generate


# ================================================================
# DIAGNOSTIC TEST DEFINITIONS (50 tests)
# ================================================================

DIAGNOSTIC_TESTS: list[DiagnosticTest] = [
    
    # ─────────────────────────────────────────────────────────────
    # ARITHMETIC REASONING (8 tests)
    # ─────────────────────────────────────────────────────────────
    
    DiagnosticTest(
        id="ARITH-001",
        name="Simple Addition",
        purpose="Basic arithmetic capability",
        skill_buckets=["RSN-ARITHMETIC"],
        band="B2",
        prompt="What is 23 + 47? The answer is",
        expected=["70"],
        failure_signals=["wrong number", "text explanation instead of number"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    DiagnosticTest(
        id="ARITH-002",
        name="Multi-step Arithmetic",
        purpose="State tracking across operations",
        skill_buckets=["RSN-ARITHMETIC"],
        band="B3",
        prompt="Calculate step by step: (15 + 25) × 2 = 40 × 2 =",
        expected=["80"],
        failure_signals=["loses intermediate result", "wrong final answer"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    DiagnosticTest(
        id="ARITH-003",
        name="Word Problem Decomposition",
        purpose="Extract numbers and operations from text",
        skill_buckets=["RSN-ARITHMETIC"],
        band="B3",
        prompt="Tom has 12 apples. He gives 5 to Jane and buys 8 more. Tom now has",
        expected=["15"],
        failure_signals=["wrong count", "ignores one operation"],
        eval_method="contains",
        max_tokens=20,
    ),
    
    DiagnosticTest(
        id="ARITH-004",
        name="Percentage Calculation",
        purpose="Percentage reasoning",
        skill_buckets=["RSN-ARITHMETIC"],
        band="B3",
        prompt="What is 25% of 80? Answer:",
        expected=["20"],
        failure_signals=["wrong percentage", "decimal error"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    DiagnosticTest(
        id="ARITH-005",
        name="Negative Numbers",
        purpose="Handle negative number arithmetic",
        skill_buckets=["RSN-ARITHMETIC"],
        band="B3",
        prompt="What is -7 + 12? The result is",
        expected=["5"],
        failure_signals=["sign error", "wrong magnitude"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    DiagnosticTest(
        id="ARITH-006",
        name="Division with Remainder",
        purpose="Division reasoning",
        skill_buckets=["RSN-ARITHMETIC"],
        band="B3",
        prompt="17 divided by 5 is 3 with remainder",
        expected=["2"],
        failure_signals=["wrong quotient", "wrong remainder"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    DiagnosticTest(
        id="ARITH-007",
        name="Order of Operations",
        purpose="PEMDAS/BODMAS adherence",
        skill_buckets=["RSN-ARITHMETIC"],
        band="B3",
        prompt="Evaluate: 3 + 4 × 2 =",
        expected=["11"],
        failure_signals=["14 (left-to-right error)", "wrong precedence"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    DiagnosticTest(
        id="ARITH-008",
        name="Fraction Basics",
        purpose="Simple fraction operations",
        skill_buckets=["RSN-ARITHMETIC"],
        band="B3",
        prompt="1/2 + 1/4 =",
        expected=["3/4", "0.75"],
        failure_signals=["wrong denominator", "addition error"],
        eval_method="contains",
        max_tokens=15,
    ),
    
    # ─────────────────────────────────────────────────────────────
    # ALGEBRAIC REASONING (5 tests)
    # ─────────────────────────────────────────────────────────────
    
    DiagnosticTest(
        id="ALG-001",
        name="Simple Equation",
        purpose="Solve for x",
        skill_buckets=["RSN-ALGEBRA"],
        band="B3",
        prompt="If x + 5 = 12, then x =",
        expected=["7"],
        failure_signals=["wrong value", "shows working but wrong answer"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    DiagnosticTest(
        id="ALG-002",
        name="Two-step Equation",
        purpose="Multi-step algebraic manipulation",
        skill_buckets=["RSN-ALGEBRA"],
        band="B4",
        prompt="Solve: 2x + 6 = 14. Therefore x =",
        expected=["4"],
        failure_signals=["arithmetic error", "wrong operation order"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    DiagnosticTest(
        id="ALG-003",
        name="Variable Substitution",
        purpose="Track variable values",
        skill_buckets=["RSN-ALGEBRA"],
        band="B4",
        prompt="If a = 3 and b = 2, then 2a + 3b =",
        expected=["12"],
        failure_signals=["substitution error", "arithmetic error"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    DiagnosticTest(
        id="ALG-004",
        name="Inequality Direction",
        purpose="Maintain inequality direction",
        skill_buckets=["RSN-ALGEBRA"],
        band="B4",
        prompt="If x > 5 and we add 3 to both sides, then x + 3 >",
        expected=["8"],
        failure_signals=["flipped inequality", "wrong number"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    DiagnosticTest(
        id="ALG-005",
        name="Pattern Continuation",
        purpose="Identify algebraic patterns",
        skill_buckets=["RSN-ALGEBRA", "RSN-ANALOGICAL"],
        band="B4",
        prompt="Sequence: 2, 6, 18, 54, ... The next number is",
        expected=["162"],
        failure_signals=["wrong pattern (×3)", "arithmetic error"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    # ─────────────────────────────────────────────────────────────
    # LOGICAL REASONING (8 tests)
    # ─────────────────────────────────────────────────────────────
    
    DiagnosticTest(
        id="LOGIC-001",
        name="Transitive Relation",
        purpose="Basic transitivity",
        skill_buckets=["RSN-LOGIC"],
        band="B3",
        prompt="John is older than Mary. Mary is older than David. Who is oldest?",
        expected=["John"],
        failure_signals=["David", "Mary", "cannot determine"],
        eval_method="contains",
        max_tokens=20,
    ),
    
    DiagnosticTest(
        id="LOGIC-002",
        name="Modus Ponens",
        purpose="Basic logical inference",
        skill_buckets=["RSN-LOGIC"],
        band="B3",
        prompt="All cats are mammals. Felix is a cat. Therefore, Felix is a",
        expected=["mammal"],
        failure_signals=["cat", "animal (too vague)", "unclear"],
        eval_method="contains",
        max_tokens=15,
    ),
    
    DiagnosticTest(
        id="LOGIC-003",
        name="Negation Handling",
        purpose="Correctly process negations",
        skill_buckets=["RSN-LOGIC"],
        band="B3",
        prompt="If it is not raining, the ground is not wet. It is not raining. Is the ground wet?",
        expected=["no", "not wet"],
        failure_signals=["yes", "wet", "cannot tell"],
        eval_method="contains",
        max_tokens=20,
    ),
    
    DiagnosticTest(
        id="LOGIC-004",
        name="Contradiction Detection",
        purpose="Identify logical contradictions",
        skill_buckets=["RSN-CONTRADICTION"],
        band="B3",
        prompt="Statement A: All birds can fly. Statement B: Penguins are birds that cannot fly. These statements are",
        expected=["contradict", "inconsistent", "conflict"],
        failure_signals=["consistent", "compatible", "both true"],
        eval_method="contains",
        max_tokens=20,
    ),
    
    DiagnosticTest(
        id="LOGIC-005",
        name="Syllogism Validity",
        purpose="Validate syllogistic arguments",
        skill_buckets=["RSN-LOGIC"],
        band="B4",
        prompt="All A are B. All B are C. Therefore, all A are C. This argument is",
        expected=["valid", "correct", "true"],
        failure_signals=["invalid", "false", "fallacy"],
        eval_method="contains",
        max_tokens=15,
    ),
    
    DiagnosticTest(
        id="LOGIC-006",
        name="Contrapositive",
        purpose="Apply contrapositive reasoning",
        skill_buckets=["RSN-LOGIC"],
        band="B4",
        prompt="If it rains, the ground gets wet. The ground is not wet. Therefore, it",
        expected=["not rain", "didn't rain", "is not raining"],
        failure_signals=["rained", "is raining", "unknown"],
        eval_method="contains",
        max_tokens=20,
    ),
    
    DiagnosticTest(
        id="LOGIC-007",
        name="Quantifier Scope",
        purpose="Handle universal vs existential",
        skill_buckets=["RSN-LOGIC"],
        band="B4",
        prompt="Some students passed the exam. Does this mean all students passed?",
        expected=["no", "not necessarily"],
        failure_signals=["yes", "all passed"],
        eval_method="contains",
        max_tokens=20,
    ),
    
    DiagnosticTest(
        id="LOGIC-008",
        name="Multi-hop Reasoning",
        purpose="Chain multiple inferences",
        skill_buckets=["RSN-LOGIC"],
        band="B4",
        prompt="A is left of B. B is left of C. C is left of D. What is the order from left to right?",
        expected=["A, B, C, D", "ABCD", "A B C D"],
        failure_signals=["wrong order", "reversed"],
        eval_method="contains",
        max_tokens=20,
    ),
    
    # ─────────────────────────────────────────────────────────────
    # CODE COMPLETION (10 tests)
    # ─────────────────────────────────────────────────────────────
    
    DiagnosticTest(
        id="CODE-001",
        name="Fibonacci Base Case",
        purpose="Complete recursive base case",
        skill_buckets=["CODE-COMPLETION"],
        band="B3",
        prompt="def fibonacci(n):\n    if n <= 1:\n        return",
        expected=["n", "1", "n if n >= 0 else"],
        failure_signals=["fibonacci(", "explanation in English", "syntax error"],
        eval_method="contains",
        test_type="completion",
        max_tokens=20,
    ),
    
    DiagnosticTest(
        id="CODE-002",
        name="List Append",
        purpose="Basic list operation",
        skill_buckets=["CODE-COMPLETION"],
        band="B2",
        prompt="numbers = [1, 2, 3]\nnumbers.append(4)\nprint(numbers)  # Output:",
        expected=["[1, 2, 3, 4]"],
        failure_signals=["[1, 2, 3]", "[4, 1, 2, 3]", "error"],
        eval_method="contains",
        max_tokens=20,
    ),
    
    DiagnosticTest(
        id="CODE-003",
        name="For Loop Range",
        purpose="Understand range behavior",
        skill_buckets=["CODE-COMPLETION"],
        band="B2",
        prompt="for i in range(3):\n    print(i)\n# Output:",
        expected=["0", "1", "2"],
        failure_signals=["3", "1 2 3"],
        eval_method="contains",
        max_tokens=20,
    ),
    
    DiagnosticTest(
        id="CODE-004",
        name="Variable Scope",
        purpose="Track variable scope",
        skill_buckets=["CODE-COMPLETION"],
        band="B3",
        prompt="x = 10\ndef foo():\n    x = 5\n    return x\nresult = foo()\nprint(x)  # Output:",
        expected=["10"],
        failure_signals=["5", "error", "None"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    DiagnosticTest(
        id="CODE-005",
        name="Dictionary Access",
        purpose="Dict key access",
        skill_buckets=["CODE-COMPLETION"],
        band="B2",
        prompt='data = {"name": "Alice", "age": 30}\nprint(data["name"])  # Output:',
        expected=["Alice"],
        failure_signals=["name", "30", "KeyError"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    DiagnosticTest(
        id="CODE-006",
        name="String Slicing",
        purpose="Slice notation understanding",
        skill_buckets=["CODE-COMPLETION"],
        band="B3",
        prompt='s = "Hello"\nprint(s[1:4])  # Output:',
        expected=["ell"],
        failure_signals=["Hel", "ello", "Hell"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    DiagnosticTest(
        id="CODE-007",
        name="Function Return",
        purpose="Understand return values",
        skill_buckets=["CODE-COMPLETION"],
        band="B3",
        prompt="def add(a, b):\n    return a + b\n\nresult = add(3, 4)\nprint(result)  # Output:",
        expected=["7"],
        failure_signals=["None", "a + b", "error"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    DiagnosticTest(
        id="CODE-008",
        name="List Comprehension",
        purpose="Comprehension syntax",
        skill_buckets=["CODE-COMPLETION"],
        band="B4",
        prompt="squares = [x**2 for x in range(4)]\nprint(squares)  # Output:",
        expected=["[0, 1, 4, 9]"],
        failure_signals=["[1, 4, 9, 16]", "[1, 4, 9]"],
        eval_method="contains",
        max_tokens=20,
    ),
    
    DiagnosticTest(
        id="CODE-009",
        name="Error Type",
        purpose="Identify error type",
        skill_buckets=["CODE-DEBUG"],
        band="B3",
        prompt='x = "5" + 3  # This raises a',
        expected=["TypeError"],
        failure_signals=["ValueError", "SyntaxError", "no error"],
        eval_method="contains",
        max_tokens=15,
    ),
    
    DiagnosticTest(
        id="CODE-010",
        name="Off-by-One Detection",
        purpose="Spot boundary errors",
        skill_buckets=["CODE-DEBUG"],
        band="B4",
        prompt="# Bug: prints numbers 1 to 9 instead of 1 to 10\nfor i in range(1, 10):\n    print(i)\n# Fix: change range(1, 10) to range(1,",
        expected=["11"],
        failure_signals=["10", "9"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    # ─────────────────────────────────────────────────────────────
    # LANGUAGE & COHERENCE (7 tests)
    # ─────────────────────────────────────────────────────────────
    
    DiagnosticTest(
        id="LANG-001",
        name="Subject-Verb Agreement",
        purpose="Basic grammar",
        skill_buckets=["LANG-GRAMMAR"],
        band="B0",
        prompt="The dogs in the park _ running. (is/are)",
        expected=["are"],
        failure_signals=["is"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    DiagnosticTest(
        id="LANG-002",
        name="Pronoun Reference",
        purpose="Track referents",
        skill_buckets=["LANG-COHERENCE"],
        band="B1",
        prompt="Sarah gave the book to Tom. He thanked her. Who said thank you?",
        expected=["Tom"],
        failure_signals=["Sarah", "she", "unclear"],
        eval_method="contains",
        max_tokens=15,
    ),
    
    DiagnosticTest(
        id="LANG-003",
        name="Sentence Continuation",
        purpose="Coherent continuation",
        skill_buckets=["LANG-COHERENCE"],
        band="B1",
        prompt="The weather was beautiful, so we decided to",
        expected=["go outside", "walk", "picnic", "beach", "park"],
        failure_signals=["stay inside", "sleep", "work"],
        eval_method="contains",
        max_tokens=20,
    ),
    
    DiagnosticTest(
        id="LANG-004",
        name="Tense Consistency",
        purpose="Maintain tense",
        skill_buckets=["LANG-GRAMMAR"],
        band="B1",
        prompt="Yesterday, I went to the store and _ some groceries. (buy/bought)",
        expected=["bought"],
        failure_signals=["buy"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    DiagnosticTest(
        id="LANG-005",
        name="Hindi Basic",
        purpose="Hindi comprehension",
        skill_buckets=["LANG-HINDI"],
        band="B1",
        prompt="मेरा नाम राहुल है। मेरा नाम क्या है?",
        expected=["राहुल", "Rahul"],
        failure_signals=["wrong name", "don't know"],
        eval_method="contains",
        max_tokens=15,
    ),
    
    DiagnosticTest(
        id="LANG-006",
        name="Paragraph Topic",
        purpose="Topic maintenance",
        skill_buckets=["LANG-COHERENCE"],
        band="B2",
        prompt="The Amazon rainforest is the largest tropical forest. It contains millions of species. The forest",
        expected=["tree", "animal", "plant", "biodiversity", "ecosystem"],
        failure_signals=["ocean", "desert", "completely unrelated"],
        eval_method="contains",
        max_tokens=30,
    ),
    
    DiagnosticTest(
        id="LANG-007",
        name="Conjunction Usage",
        purpose="Logical connectors",
        skill_buckets=["LANG-GRAMMAR"],
        band="B1",
        prompt="I wanted to go swimming, _ it was too cold. (and/but/so)",
        expected=["but"],
        failure_signals=["and", "so"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    # ─────────────────────────────────────────────────────────────
    # KNOWLEDGE & COMMONSENSE (7 tests)
    # ─────────────────────────────────────────────────────────────
    
    DiagnosticTest(
        id="KNOW-001",
        name="Basic Fact",
        purpose="Factual recall",
        skill_buckets=["KNOW-FACTUAL"],
        band="B1",
        prompt="The capital of France is",
        expected=["Paris"],
        failure_signals=["London", "Berlin", "Rome"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    DiagnosticTest(
        id="KNOW-002",
        name="Scientific Fact",
        purpose="Basic science",
        skill_buckets=["KNOW-SCIENCE"],
        band="B2",
        prompt="Water freezes at _ degrees Celsius.",
        expected=["0", "zero"],
        failure_signals=["100", "32", "-10"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    DiagnosticTest(
        id="KNOW-003",
        name="Commonsense Physics",
        purpose="Physical intuition",
        skill_buckets=["KNOW-COMMONSENSE"],
        band="B1",
        prompt="If you drop a ball, it will",
        expected=["fall", "drop", "ground", "down"],
        failure_signals=["float", "rise", "stay"],
        eval_method="contains",
        max_tokens=15,
    ),
    
    DiagnosticTest(
        id="KNOW-004",
        name="Temporal Order",
        purpose="Historical ordering",
        skill_buckets=["KNOW-FACTUAL"],
        band="B2",
        prompt="World War I happened before or after World War II?",
        expected=["before"],
        failure_signals=["after", "same time"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    DiagnosticTest(
        id="KNOW-005",
        name="Object Function",
        purpose="Object affordances",
        skill_buckets=["KNOW-COMMONSENSE"],
        band="B1",
        prompt="A hammer is used to",
        expected=["hit", "nail", "pound", "build"],
        failure_signals=["cut", "cook", "write"],
        eval_method="contains",
        max_tokens=15,
    ),
    
    DiagnosticTest(
        id="KNOW-006",
        name="Causal Chain",
        purpose="Cause-effect reasoning",
        skill_buckets=["RSN-CAUSAL"],
        band="B2",
        prompt="If you don't water a plant, it will",
        expected=["die", "wilt", "dry"],
        failure_signals=["grow", "bloom", "thrive"],
        eval_method="contains",
        max_tokens=15,
    ),
    
    DiagnosticTest(
        id="KNOW-007",
        name="Social Situation",
        purpose="Social commonsense",
        skill_buckets=["KNOW-COMMONSENSE"],
        band="B2",
        prompt="If someone gives you a gift, you should say",
        expected=["thank", "thanks"],
        failure_signals=["nothing", "no", "give it back"],
        eval_method="contains",
        max_tokens=15,
    ),
    
    # ─────────────────────────────────────────────────────────────
    # ADVANCED REASONING (5 tests)
    # ─────────────────────────────────────────────────────────────
    
    DiagnosticTest(
        id="ADV-001",
        name="Counterfactual",
        purpose="Counterfactual reasoning",
        skill_buckets=["RSN-CAUSAL"],
        band="B4",
        prompt="If the sun didn't exist, Earth would be",
        expected=["cold", "dark", "frozen", "lifeless"],
        failure_signals=["warm", "bright", "normal"],
        eval_method="contains",
        max_tokens=20,
    ),
    
    DiagnosticTest(
        id="ADV-002",
        name="Analogy Completion",
        purpose="A:B::C:D patterns",
        skill_buckets=["RSN-ANALOGICAL"],
        band="B4",
        prompt="hot is to cold as up is to",
        expected=["down"],
        failure_signals=["left", "right", "warm"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    DiagnosticTest(
        id="ADV-003",
        name="Constraint Satisfaction",
        purpose="Multiple constraints",
        skill_buckets=["RSN-LOGIC"],
        band="B4",
        prompt="I need a number that is: greater than 5, less than 10, and even. The number is",
        expected=["6", "8"],
        failure_signals=["7", "9", "10", "4"],
        eval_method="contains",
        max_tokens=10,
    ),
    
    DiagnosticTest(
        id="ADV-004",
        name="Algorithm Trace",
        purpose="Trace algorithm execution",
        skill_buckets=["CODE-ALGO"],
        band="B5",
        prompt="Bubble sort [3, 1, 2] after one full pass: [1, 2, 3] or [1, 3, 2]?",
        expected=["[1, 3, 2]", "1, 3, 2"],
        failure_signals=["[1, 2, 3]", "1, 2, 3", "[3, 1, 2]"],
        eval_method="contains",
        max_tokens=20,
    ),
    
    DiagnosticTest(
        id="ADV-005",
        name="Recursive Thinking",
        purpose="Understand recursion",
        skill_buckets=["CODE-ALGO"],
        band="B5",
        prompt="factorial(3) = 3 × factorial(2) = 3 × 2 × factorial(1) = 3 × 2 × 1 =",
        expected=["6"],
        failure_signals=["3", "2", "1"],
        eval_method="contains",
        max_tokens=10,
    ),
]


# ================================================================
# TEST RUNNER UTILITIES
# ================================================================

def get_tests_for_skill(skill_id: str) -> list[DiagnosticTest]:
    """Get all tests for a skill bucket."""
    return [t for t in DIAGNOSTIC_TESTS if skill_id in t.skill_buckets]


def get_tests_for_band(band: str) -> list[DiagnosticTest]:
    """Get all tests for a band."""
    return [t for t in DIAGNOSTIC_TESTS if t.band == band]


def get_test_by_id(test_id: str) -> DiagnosticTest | None:
    """Get a test by its ID."""
    for t in DIAGNOSTIC_TESTS:
        if t.id == test_id:
            return t
    return None


def evaluate_test(test: DiagnosticTest, model_output: str) -> dict:
    """Evaluate model output against a test.
    
    Returns:
        {
            "passed": bool,
            "expected": str | list,
            "actual": str,
            "failure_detected": str | None,
        }
    """
    output = model_output.strip().lower()
    expected = test.expected if isinstance(test.expected, list) else [test.expected]
    expected_lower = [e.lower() for e in expected]
    
    passed = False
    failure_detected = None
    
    if test.eval_method == "exact_match":
        passed = output in expected_lower
        
    elif test.eval_method == "contains":
        passed = any(e in output for e in expected_lower)
        
    elif test.eval_method == "numeric":
        try:
            output_num = float(re.search(r'-?\d+\.?\d*', output).group())
            passed = any(abs(output_num - float(e)) < 0.01 for e in expected)
        except (AttributeError, ValueError):
            passed = False
            
    elif test.eval_method == "regex":
        passed = any(re.search(e, output, re.IGNORECASE) for e in expected)
        
    elif test.eval_method == "not_contains":
        passed = not any(e in output for e in expected_lower)
        
    elif test.eval_method == "code_valid":
        try:
            compile(output, "<string>", "exec")
            passed = True
        except SyntaxError:
            passed = False
    
    # Check for failure signals
    if not passed:
        for signal in test.failure_signals:
            if signal.lower() in output:
                failure_detected = signal
                break
    
    return {
        "passed": passed,
        "expected": test.expected,
        "actual": model_output[:100],
        "failure_detected": failure_detected,
    }


# ================================================================
# SUMMARY
# ================================================================

def get_test_summary() -> dict:
    """Get summary statistics about tests."""
    by_skill = {}
    by_band = {}
    
    for t in DIAGNOSTIC_TESTS:
        for skill in t.skill_buckets:
            by_skill[skill] = by_skill.get(skill, 0) + 1
        by_band[t.band] = by_band.get(t.band, 0) + 1
    
    return {
        "total_tests": len(DIAGNOSTIC_TESTS),
        "by_skill": by_skill,
        "by_band": by_band,
    }
