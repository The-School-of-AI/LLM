"""
verification.py — Sample Verification Pipeline

Multi-stage verification for synthetic data quality:
1. Teacher correctness: Verify the answer is correct
2. Student re-solve: Have a smaller model solve and compare
3. Consistency check: Verify reasoning matches answer

Usage:
    from validation.verification import VerificationPipeline

    pipeline = VerificationPipeline(teacher_model="qwen3:8b", student_model="qwen3:4b")
    results = pipeline.verify_samples(samples)
"""

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent to path for imports
_VERIFY_SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VERIFY_SCRIPT_DIR))

from common.ollama_client import ollama_generate as _ollama_generate_base  # noqa: E402


@dataclass
class VerificationResult:
    """Result of verifying a single sample."""

    sample_id: str
    verified: bool
    verification_score: float  # 0.0 to 1.0

    teacher_check: bool = False
    student_check: bool = False
    consistency_check: bool = False

    teacher_answer: str = ""
    student_answer: str = ""

    failure_reason: Optional[str] = None


@dataclass
class VerificationReport:
    """Aggregated verification report."""

    timestamp: str
    teacher_model: str
    student_model: str
    total_samples: int
    verified_samples: int
    verification_rate: float
    avg_score: float

    by_skill: dict = field(default_factory=dict)
    by_band: dict = field(default_factory=dict)
    failed_samples: list = field(default_factory=list)


def extract_answer(text: str) -> str:
    """Extract the final answer from a response."""
    if not text:
        return ""

    # Try to find explicit answer markers
    patterns = [
        r"(?:the )?answer(?:\s+is)?[:\s]+([^\n.]+)",
        r"(?:result|solution)[:\s]+([^\n.]+)",
        r"=\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*$",  # Last number in text
    ]

    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1).strip()

    # Fallback: return last line
    lines = text.strip().split("\n")
    return lines[-1].strip() if lines else ""


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if not answer:
        return ""
    # Remove whitespace, lowercase
    answer = answer.lower().strip()
    # Remove common prefixes
    for prefix in ["the answer is", "answer:", "result:", "="]:
        if answer.startswith(prefix):
            answer = answer[len(prefix) :].strip()
    # Try to extract just the number if numeric
    match = re.search(r"(-?\d+(?:\.\d+)?)", answer)
    if match:
        return match.group(1)
    return answer


def answers_match(answer1: str, answer2: str, tolerance: float = 0.01) -> bool:
    """Check if two answers match."""
    norm1 = normalize_answer(answer1)
    norm2 = normalize_answer(answer2)

    # Exact string match
    if norm1 == norm2:
        return True

    # Numeric comparison with tolerance
    try:
        num1 = float(norm1)
        num2 = float(norm2)
        return abs(num1 - num2) <= tolerance * max(abs(num1), abs(num2), 1)
    except (ValueError, TypeError):
        pass

    # Substring match for text answers
    if len(norm1) > 3 and len(norm2) > 3:
        return norm1 in norm2 or norm2 in norm1

    return False


def ollama_generate(model: str, prompt: str, max_tokens: int = 200) -> str:
    """Generate text from Ollama with error handling."""
    try:
        return _ollama_generate_base(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.1,
            think=False,
        )
    except Exception as e:
        print(f"  [Ollama Error] {e}")
        return ""


class VerificationPipeline:
    """Multi-stage verification pipeline."""

    def __init__(
        self,
        teacher_model: str = "qwen3:8b",
        student_model: str = "qwen3:4b",
        require_all_checks: bool = False,
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.require_all_checks = require_all_checks

    def verify_sample(self, sample: dict) -> VerificationResult:
        """Verify a single sample."""
        sample_id = sample.get("id", "unknown")
        question = sample.get("question") or sample.get("instruction", "")
        expected_answer = sample.get("answer") or sample.get("distilled_view", "")

        result = VerificationResult(
            sample_id=sample_id,
            verified=False,
            verification_score=0.0,
        )

        # Extract expected answer
        expected = normalize_answer(extract_answer(expected_answer))
        if not expected:
            # Try to extract from distilled_view
            expected = normalize_answer(
                extract_answer(sample.get("distilled_view", ""))
            )

        if not question:
            result.failure_reason = "no_question"
            return result

        # 1. Teacher correctness check
        teacher_prompt = f"""Solve this problem step by step, then give the final answer.

Problem: {question}

Solve:"""

        teacher_response = ollama_generate(self.teacher_model, teacher_prompt, 300)
        result.teacher_answer = teacher_response
        teacher_answer = normalize_answer(extract_answer(teacher_response))

        if expected and teacher_answer:
            result.teacher_check = answers_match(teacher_answer, expected)
        elif teacher_response:
            # If we don't have expected answer, accept teacher response
            result.teacher_check = True

        # 2. Student re-solve check
        student_prompt = f"""Solve this problem. Give a short answer.

Problem: {question}

Answer:"""

        student_response = ollama_generate(self.student_model, student_prompt, 100)
        result.student_answer = student_response
        student_answer = normalize_answer(extract_answer(student_response))

        if expected and student_answer:
            result.student_check = answers_match(student_answer, expected)
        elif teacher_answer and student_answer:
            # Compare student to teacher if no expected
            result.student_check = answers_match(student_answer, teacher_answer)

        # 3. Consistency check (reasoning matches answer)
        think_view = sample.get("think_view", "")
        distilled_view = sample.get("distilled_view", "")

        if think_view and distilled_view:
            # Check if distilled answer appears in reasoning conclusion
            think_answer = normalize_answer(extract_answer(think_view))
            distilled_answer = normalize_answer(extract_answer(distilled_view))
            result.consistency_check = answers_match(think_answer, distilled_answer)
        else:
            result.consistency_check = True  # No CoT to check

        # Calculate verification score
        checks = [result.teacher_check, result.student_check, result.consistency_check]
        result.verification_score = sum(checks) / len(checks)

        if self.require_all_checks:
            result.verified = all(checks)
        else:
            # Verified if majority of checks pass
            result.verified = sum(checks) >= 2

        if not result.verified:
            failed = []
            if not result.teacher_check:
                failed.append("teacher")
            if not result.student_check:
                failed.append("student")
            if not result.consistency_check:
                failed.append("consistency")
            result.failure_reason = f"failed_checks:{','.join(failed)}"

        return result

    def verify_samples(
        self,
        samples: list[dict],
        progress_callback=None,
    ) -> VerificationReport:
        """Verify multiple samples.

        Args:
            samples: List of sample dicts
            progress_callback: Optional callback(current, total) for progress
        """
        results = []
        by_skill = {}
        by_band = {}
        failed = []

        total = len(samples)
        print(f"\n[Verification] Processing {total} samples...")
        print(f"  Teacher: {self.teacher_model}")
        print(f"  Student: {self.student_model}")

        for i, sample in enumerate(samples):
            if progress_callback:
                progress_callback(i + 1, total)
            elif (i + 1) % 5 == 0 or i == 0:
                print(f"  [{i+1}/{total}]", end="\r")

            result = self.verify_sample(sample)
            results.append(result)

            # Track by skill
            skill = sample.get("skill_bucket", "unknown")
            if skill not in by_skill:
                by_skill[skill] = {"total": 0, "verified": 0}
            by_skill[skill]["total"] += 1
            if result.verified:
                by_skill[skill]["verified"] += 1

            # Track by band
            band = sample.get("band", "unknown")
            if band not in by_band:
                by_band[band] = {"total": 0, "verified": 0}
            by_band[band]["total"] += 1
            if result.verified:
                by_band[band]["verified"] += 1

            # Track failures
            if not result.verified:
                failed.append(
                    {
                        "id": result.sample_id,
                        "score": result.verification_score,
                        "reason": result.failure_reason,
                        "skill": skill,
                        "band": band,
                    }
                )

        verified = [r for r in results if r.verified]
        scores = [r.verification_score for r in results]

        report = VerificationReport(
            timestamp=datetime.now().isoformat(),
            teacher_model=self.teacher_model,
            student_model=self.student_model,
            total_samples=total,
            verified_samples=len(verified),
            verification_rate=len(verified) / total if total else 0.0,
            avg_score=sum(scores) / len(scores) if scores else 0.0,
            by_skill={
                k: v["verified"] / v["total"] if v["total"] else 0
                for k, v in by_skill.items()
            },
            by_band={
                k: v["verified"] / v["total"] if v["total"] else 0
                for k, v in by_band.items()
            },
            failed_samples=failed[:20],  # Limit to 20 examples
        )

        print(
            f"\n[Done] {report.verified_samples}/{report.total_samples} verified "
            f"({report.verification_rate*100:.1f}%)"
        )

        return report


def run_verification(
    data_path: str,
    output_path: str = None,
    teacher_model: str = "qwen3:8b",
    student_model: str = "qwen3:4b",
    update_samples: bool = True,
) -> VerificationReport:
    """Run verification on a data file.

    Args:
        data_path: Path to JSONL file
        output_path: Path to save verified data (default: same file with _verified suffix)
        teacher_model: Model for teacher verification
        student_model: Model for student verification
        update_samples: If True, update samples with verified field
    """
    # Load samples
    with open(data_path, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    # Run verification
    pipeline = VerificationPipeline(
        teacher_model=teacher_model,
        student_model=student_model,
    )
    report = pipeline.verify_samples(samples)

    # Update samples with verification results
    if update_samples:
        results = {r.sample_id: r for r in [pipeline.verify_sample(s) for s in samples]}

        for sample in samples:
            sid = sample.get("id", "")
            if sid in results:
                sample["verified"] = results[sid].verified
                sample["verification_score"] = results[sid].verification_score

    # Save updated samples
    if output_path is None:
        base = Path(data_path)
        output_path = str(base.parent / f"{base.stem}_verified{base.suffix}")

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[Saved] {output_path}")

    # Save report
    report_path = Path(output_path).parent / "verification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": report.timestamp,
                "teacher_model": report.teacher_model,
                "student_model": report.student_model,
                "total_samples": report.total_samples,
                "verified_samples": report.verified_samples,
                "verification_rate": report.verification_rate,
                "avg_score": report.avg_score,
                "by_skill": report.by_skill,
                "by_band": report.by_band,
                "failed_samples": report.failed_samples,
            },
            f,
            indent=2,
        )
    print(f"[Report] {report_path}")

    return report


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        run_verification(sys.argv[1])
    else:
        print("Usage: python verification.py <data.jsonl>")
