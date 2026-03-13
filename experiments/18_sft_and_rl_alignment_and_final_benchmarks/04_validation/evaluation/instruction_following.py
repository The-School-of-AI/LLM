"""
Instruction-Following Scorer
==============================
Evaluates how well a model response adheres to the explicit instructions
in an evaluation prompt's rubric and constraint specification.

Scoring strategy (automatic heuristics + manual override support):
  1. must_have  checks  — presence of expected elements (keywords, structure)
  2. must_not_have checks — absence of forbidden elements
  3. constraint checks   — format, count, length, word count, etc.

Each check is binary (pass=1 / fail=0). The overall IF score is the
fraction of checks that passed, in [0.0, 1.0].

A response is labelled "instruction_followed" if its score >= threshold
(default 0.75 — configurable via config.yaml).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str = ""


@dataclass
class IFScore:
    score: float                        # 0.0 – 1.0
    followed: bool                      # True if score >= threshold
    checks_passed: int = 0
    checks_total: int = 0
    check_details: list[CheckResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "followed": self.followed,
            "checks_passed": self.checks_passed,
            "checks_total": self.checks_total,
            "check_details": [
                {"name": c.name, "passed": c.passed, "details": c.details}
                for c in self.check_details
            ],
        }


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class InstructionFollowingScorer:
    """
    Scores model outputs for instruction adherence.

    Parameters
    ----------
    config : dict
        Scoring configuration (from config.yaml → scoring section).
        Keys used:
          - threshold (float, default 0.75): minimum score to count as "followed"
          - case_sensitive (bool, default False): for keyword checks
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.threshold: float = float(cfg.get("threshold", 0.75))
        self.case_sensitive: bool = bool(cfg.get("case_sensitive", False))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        output: str,
        rubric: dict,
        constraints: dict,
    ) -> dict:
        """
        Score `output` against `rubric` and `constraints`.

        Returns
        -------
        dict  — serialisable IFScore dict.
        """
        checks: list[CheckResult] = []

        # 1. Rubric: must_have
        for item in rubric.get("must_have", []):
            checks.append(self._check_must_have(output, item))

        # 2. Rubric: must_not_have
        for item in rubric.get("must_not_have", []):
            checks.append(self._check_must_not_have(output, item))

        # 3. Structural constraints
        checks.extend(self._check_constraints(output, constraints))

        total = len(checks)
        passed = sum(1 for c in checks if c.passed)
        score_val = (passed / total) if total > 0 else 1.0
        followed = score_val >= self.threshold

        result = IFScore(
            score=round(score_val, 4),
            followed=followed,
            checks_passed=passed,
            checks_total=total,
            check_details=checks,
        )
        return result.to_dict()

    # ------------------------------------------------------------------
    # Rubric checks
    # ------------------------------------------------------------------

    def _normalise(self, text: str) -> str:
        return text if self.case_sensitive else text.lower()

    def _check_must_have(self, output: str, item: str) -> CheckResult:
        """Check that `item` pattern is present in output."""
        passed = self._pattern_present(output, item)
        return CheckResult(
            name=f"must_have: {item[:60]}",
            passed=passed,
            details="PASS" if passed else f"MISSING: '{item[:60]}' not found in output",
        )

    def _check_must_not_have(self, output: str, item: str) -> CheckResult:
        """Check that `item` pattern is absent from output."""
        present = self._pattern_present(output, item)
        passed = not present
        return CheckResult(
            name=f"must_not_have: {item[:60]}",
            passed=passed,
            details="PASS" if passed else f"FOUND FORBIDDEN: '{item[:60]}' present in output",
        )

    def _pattern_present(self, output: str, pattern: str) -> bool:
        """
        Check whether `pattern` is present in `output`.
        Supports a few shorthand patterns:
          - "numbered list"           → checks for "1." lines
          - "markdown table"          → checks for pipe-separated header+separator
          - exact strings otherwise
        """
        norm_out = self._normalise(output)
        norm_pat = self._normalise(pattern)

        # shorthand dispatchers
        if norm_pat == "numbered list":
            return bool(re.search(r"^\s*\d+[.)]", output, re.MULTILINE))
        if norm_pat == "bullet points instead of numbers":
            return bool(re.search(r"^\s*[-*•]", output, re.MULTILINE)) and not bool(
                re.search(r"^\s*\d+[.)]", output, re.MULTILINE)
            )
        if norm_pat == "valid json":
            try:
                import json
                json.loads(output)
                return True
            except Exception:
                return False
        if norm_pat == "markdown table" or "pipe" in norm_pat:
            return "|" in output and "---" in output

        # generic substring search
        return norm_pat in norm_out

    # ------------------------------------------------------------------
    # Constraint checks
    # ------------------------------------------------------------------

    def _check_constraints(self, output: str, constraints: dict) -> list[CheckResult]:
        checks: list[CheckResult] = []

        for key, value in constraints.items():
            check = self._dispatch_constraint(output, key, value)
            if check is not None:
                checks.append(check)

        return checks

    def _dispatch_constraint(
        self, output: str, key: str, value: Any
    ) -> CheckResult | None:
        """Route a constraint key to the appropriate check method."""
        handlers = {
            "sentence_count": self._check_sentence_count,
            "count": self._check_item_count,
            "paragraph_count": self._check_paragraph_count,
            "word_count_exact": self._check_word_count_exact,
            "max_sentences": self._check_max_sentences,
            "required_keywords": self._check_required_keywords,
            "forbidden_words": self._check_forbidden_words,
            "format": self._check_format,
            "output_format": self._check_output_format,
            "valid_json": self._check_valid_json,
        }

        handler = handlers.get(key)
        if handler is not None:
            return handler(output, value)
        return None  # unknown constraint — skip silently

    def _check_sentence_count(self, output: str, count: int) -> CheckResult:
        actual = self._count_sentences(output)
        passed = actual == count
        return CheckResult(
            name=f"sentence_count == {count}",
            passed=passed,
            details=f"Expected {count} sentences, found ~{actual}",
        )

    def _check_max_sentences(self, output: str, max_count: int) -> CheckResult:
        actual = self._count_sentences(output)
        passed = actual <= max_count
        return CheckResult(
            name=f"sentence_count <= {max_count}",
            passed=passed,
            details=f"Expected ≤ {max_count} sentences, found ~{actual}",
        )

    def _check_item_count(self, output: str, count: int) -> CheckResult:
        # Detect numbered-list items
        items = re.findall(r"^\s*\d+[.)]", output, re.MULTILINE)
        bullets = re.findall(r"^\s*[-*•]", output, re.MULTILINE)
        actual = max(len(items), len(bullets))
        passed = actual == count
        return CheckResult(
            name=f"item_count == {count}",
            passed=passed,
            details=f"Expected {count} items, found ~{actual}",
        )

    def _check_paragraph_count(self, output: str, count: int) -> CheckResult:
        paras = [p.strip() for p in re.split(r"\n\s*\n", output.strip()) if p.strip()]
        actual = len(paras)
        passed = actual == count
        return CheckResult(
            name=f"paragraph_count == {count}",
            passed=passed,
            details=f"Expected {count} paragraphs, found {actual}",
        )

    def _check_word_count_exact(self, output: str, target: int) -> CheckResult:
        words = len(output.split())
        tolerance = max(3, int(target * 0.05))  # ±5% or ±3 words
        passed = abs(words - target) <= tolerance
        return CheckResult(
            name=f"word_count ≈ {target}",
            passed=passed,
            details=f"Expected ~{target} words (±{tolerance}), found {words}",
        )

    def _check_required_keywords(self, output: str, keywords: list[str]) -> CheckResult:
        norm_out = self._normalise(output)
        missing = [kw for kw in keywords if self._normalise(kw) not in norm_out]
        passed = len(missing) == 0
        return CheckResult(
            name=f"required_keywords: {keywords}",
            passed=passed,
            details="All keywords found" if passed else f"Missing: {missing}",
        )

    def _check_forbidden_words(self, output: str, words: list[str]) -> CheckResult:
        norm_out = self._normalise(output)
        found = [w for w in words if self._normalise(w) in norm_out]
        passed = len(found) == 0
        return CheckResult(
            name=f"forbidden_words absent",
            passed=passed,
            details="No forbidden words found" if passed else f"Found forbidden: {found}",
        )

    def _check_format(self, output: str, fmt: str) -> CheckResult:
        fmt_lower = fmt.lower()
        if fmt_lower in ("numbered_list", "numbered list"):
            passed = bool(re.search(r"^\s*\d+[.)]", output, re.MULTILINE))
        elif fmt_lower in ("markdown_table", "markdown table"):
            passed = "|" in output and "---" in output
        elif fmt_lower in ("markdown_h2", "markdown h2"):
            passed = bool(re.search(r"^##\s", output, re.MULTILINE))
        elif fmt_lower == "json":
            try:
                import json
                json.loads(output)
                passed = True
            except Exception:
                passed = False
        else:
            passed = True  # unknown format — pass by default
        return CheckResult(
            name=f"format: {fmt}",
            passed=passed,
            details=f"Format '{fmt}' check {'PASS' if passed else 'FAIL'}",
        )

    def _check_output_format(self, output: str, fmt: str) -> CheckResult:
        return self._check_format(output, fmt)

    def _check_valid_json(self, output: str, required: bool) -> CheckResult:
        if not required:
            return CheckResult(name="valid_json (skipped)", passed=True)
        try:
            import json
            # Strip markdown code fences if present
            cleaned = re.sub(r"```(?:json)?\s*", "", output).replace("```", "").strip()
            json.loads(cleaned)
            passed = True
            detail = "Valid JSON found"
        except Exception as e:
            passed = False
            detail = f"Invalid JSON: {e}"
        return CheckResult(name="valid_json", passed=passed, details=detail)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_sentences(text: str) -> int:
        """Rough sentence count using punctuation heuristics."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return len([s for s in sentences if len(s) > 3])


# ---------------------------------------------------------------------------
# Stand-alone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scorer = InstructionFollowingScorer()

    sample_output = """
1. Exercise improves your cardiovascular health significantly.
2. Regular physical activity strengthens muscles and bones.
3. Exercise has proven mental health benefits including reduced anxiety.
4. It helps maintain a healthy body weight over time.
5. Better sleep quality is a well-documented benefit of exercise.
"""

    sample_rubric = {
        "must_have": ["numbered list", "exactly 5 items"],
        "must_not_have": ["bullet points instead of numbers"],
    }
    sample_constraints = {
        "format": "numbered_list",
        "count": 5,
    }

    result = scorer.score(sample_output, sample_rubric, sample_constraints)
    import json
    print(json.dumps(result, indent=2))
