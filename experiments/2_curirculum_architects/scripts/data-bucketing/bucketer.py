import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class BucketingResult:
    band: str
    rationale: List[str]
    flags: List[str] = field(default_factory=list)


class DataBucketer:
    def __init__(self):
        # Reasoning markers - indicators of explicit logic
        self.reasoning_markers = [
            r"\btherefore\b",
            r"\bbecause\b",
            r"\bthus\b",
            r"\bhence\b",
            r"\bimplies\b",
            r"\bconsequently\b",
            r"\bit follows that\b",
            r"\bgiven that\b",
            r"\bassume\b",
            r"\blet\s+[a-zA-Z]\s+be\b",
        ]

        # Abstraction markers - indicators of high-level academic/formal text
        self.abstraction_markers = [
            r"\bhypothesis\b",
            r"\btheorem\b",
            r"\bcorollary\b",
            r"\blemma\b",
            r"\bproposition\b",
            r"\bparadigm\b",
            r"\bepistemology\b",
            r"\bmethodology\b",
            r"\bsynthesis\b",
            r"\bdialectic\b",
        ]

        # Code indicators (stricter)
        self.code_markers = [
            r"```",
            r"def\s+[a-zA-Z_]\w*\(",
            r"class\s+[A-Z]\w*[:\(\{]",
            r"function\s+\w+\(",
            r"var\s+\w+\s*=",
            r"const\s+\w+\s*=",
            r"import\s+[\w\.]+",
            r"return\s+[\w\.]+",
        ]

    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyzes structural properties of the text."""
        # Split by double newline, but filter for "substantial" paragraphs (more than just a list item)
        raw_paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]

        # Consider a paragraph substantial if it has > 5 words OR ends with sentence punctuation
        paragraphs = []
        for p in raw_paragraphs:
            words_in_p = len(p.split())
            if words_in_p > 5 or re.search(r'[.!?]["\']?$', p.strip()):
                paragraphs.append(p)

        # Remove citations like [1], [2] to avoid splitting sentences on them if they match,
        # but simplified sentence split is usually okay.
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        words = text.split()

        avg_sentence_len = len(words) / len(sentences) if sentences else 0
        avg_paragraph_len = len(words) / len(paragraphs) if paragraphs else 0

        return {
            "paragraph_count": len(paragraphs),
            "sentence_count": len(sentences),
            "word_count": len(words),
            "avg_sentence_len": avg_sentence_len,
            "avg_paragraph_len": avg_paragraph_len,
            "has_enumerated_list": bool(re.search(r"^\s*[\d\*]\.", text, re.MULTILINE)),
        }

    def _count_matches(self, text: str, patterns: List[str]) -> int:
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count

    def bucket_sample(
        self,
        text: str,
        language: str = "en",
        domain_tags: List[str] = None,
        source_id: str = None,
    ) -> BucketingResult:
        """
        Assigns a difficulty band (B0-B5) to a text sample based on heuristics.
        """
        if not text or not isinstance(text, str):
            return BucketingResult(
                band="B0", rationale=["Empty or invalid input"], flags=["invalid_input"]
            )

        metrics = self._analyze_structure(text)
        reasoning_count = self._count_matches(text, self.reasoning_markers)
        abstraction_count = self._count_matches(text, self.abstraction_markers)
        code_count = self._count_matches(text, self.code_markers)

        rationale = []
        flags = []

        # Calculate density metrics (per 100 words to normalize)
        word_count = max(metrics["word_count"], 1)  # Avoid div/0
        reasoning_density = (reasoning_count / word_count) * 100
        abstraction_density = (abstraction_count / word_count) * 100

        rationale.append(
            f"Words: {metrics['word_count']}, Paragraphs: {metrics['paragraph_count']}"
        )
        rationale.append(
            f"Reasoning density: {reasoning_density:.2f}, Abstraction density: {abstraction_density:.2f}"
        )

        # --- HEURISTIC DECISION TREE ---
        # Start from highest complexity and fall through

        # Flags
        if metrics["word_count"] > 2000 and reasoning_density < 0.5:
            flags.append("suspected_verbosity_without_depth")

        if metrics["word_count"] < 10:
            return BucketingResult(
                band="B0", rationale=["Too short"], flags=["short_content"]
            )

        # B5: PhD
        # Requirements: High abstraction, novel synthesis, very low redundancy.
        if (
            abstraction_density > 1.5
            and reasoning_density > 1.5
            and metrics["avg_sentence_len"] > 14
            and metrics["paragraph_count"] >= 3
        ):
            return BucketingResult(
                "B5",
                rationale + ["High abstraction & reasoning density, complex structure"],
                flags,
            )

        # B4: Graduate
        # Requirements: Formal reasoning, long dependency chains.
        if (
            reasoning_density > 1.0
            and metrics["avg_sentence_len"] > 12
            and metrics["paragraph_count"] >= 2
        ):
            return BucketingResult(
                "B4",
                rationale + ["Significant reasoning markers, long sentences"],
                flags,
            )

        # B3: Undergraduate
        # Requirements: Multi-step explanations, code/tutorials.
        # Stricter code requirement: Must be substantial to count alone, or mixed with reasoning
        # Stricter reasoning: Enumerated lists alone aren't enough without high reasoning density
        if (code_count > 2) or (
            reasoning_density > 0.8 and metrics["has_enumerated_list"]
        ):
            reason = (
                "Significant code presence"
                if code_count > 2
                else "Reasoning with structured steps"
            )
            return BucketingResult("B3", rationale + [reason], flags)

        # B2: High School
        # Requirements: Structured knowledge, implicit reasoning.
        # Added: Must have substantial paragraph length to avoid listicles
        if (
            metrics["paragraph_count"] > 1
            and metrics["avg_paragraph_len"] > 15
            and metrics["avg_sentence_len"] > 10
        ):
            return BucketingResult(
                "B2",
                rationale + ["Structured multi-paragraph text, moderate complexity"],
                flags,
            )

        # B1: Primary
        # Requirements: Fluent everyday language, common knowledge.
        # Fallback for structured but simpler text
        if metrics["word_count"] > 20:
            # Check if it might be B2 candidate by vocabulary/length but missed structure
            if metrics["avg_sentence_len"] > 15:
                return BucketingResult(
                    "B2", rationale + ["Dense sentence structure"], flags
                )
            return BucketingResult(
                "B1", rationale + ["Fluent text, simple structure"], flags
            )

        # B0: Nursery
        # Default fallback
        return BucketingResult(
            "B0", rationale + ["Simple/short text, low complexity signals"], flags
        )


if __name__ == "__main__":
    # Simple CLI test
    import sys

    bucketer = DataBucketer()

    sample_text = """
    The cat sat on the mat. It was a sunny day.
    """

    if len(sys.argv) > 1:
        # If file path provided
        try:
            with open(sys.argv[1], "r") as f:
                sample_text = f.read()
        except Exception:
            sample_text = sys.argv[1]

    result = bucketer.bucket_sample(sample_text)
    print(f"Band: {result.band}")
    print(f"Rationale: {result.rationale}")
    print(f"Flags: {result.flags}")
