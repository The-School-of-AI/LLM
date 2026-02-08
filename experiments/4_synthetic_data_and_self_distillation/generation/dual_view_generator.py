"""
dual_view_generator.py — Generate samples with <dist> and <think> views.

Every synthetic sample has TWO views:
  1. <dist> — Distilled: Final answer + short justification
  2. <think> — COT: Full chain-of-thought reasoning

COT is gated by band:
  B0-B2: No COT (only distilled)
  B3: Rare, short COT
  B4-B5: Allowed, capped COT

Also generates:
  • Hard negatives (deliberate mistakes)
  • Error correction pairs
"""

import json
import os
import random
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Add parent to path
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from common.bands import (  # noqa: E402
    DIST_END,
    DIST_START,
    THINK_END,
    THINK_START,
    get_band_spec,
)
from common.skills import FAILURE_MODES, get_skill_bucket  # noqa: E402

# ================================================================
# OLLAMA CLIENT
# ================================================================

OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def ollama_chat(
    model: str,
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """Chat completion via Ollama."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }

    url = f"{OLLAMA_BASE}/api/chat"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    return result.get("message", {}).get("content", "").strip()


# ================================================================
# GENERATION PROMPTS
# ================================================================

# Distilled view prompt (answer + short justification)
DISTILLED_PROMPT = """Solve this question. You MUST respond in this EXACT format:

Answer: [your answer here]
Justification: [1-2 sentence explanation]

Example:
Question: What is 2+2?
Answer: 4
Justification: Adding 2 and 2 gives 4.

Now solve:
Question: {question}"""

# COT view prompt (full reasoning) - uses numbered steps instead of <think> tags
COT_PROMPT = """Solve this step-by-step. You MUST use numbered steps.

Question: {question}

Format your response as:
Step 1: [first reasoning step]
Step 2: [next step]
...
Final Answer: [your answer]

Begin solving:"""

# Hard negative prompt (deliberate mistake)
HARD_NEGATIVE_PROMPT = """You are generating training data with INTENTIONAL MISTAKES for teaching error detection.

Question: {question}
Correct Answer: {correct_answer}

Generate a PLAUSIBLE BUT WRONG answer with reasoning that contains a specific error.

Error type to introduce: {error_type}

Respond in this format:
<think>
[Reasoning that contains the error - make it look convincing but wrong]
</think>

Wrong Answer: [incorrect answer that follows from the flawed reasoning]
Error Location: [which step has the error]"""

# Error correction prompt
ERROR_CORRECTION_PROMPT = """You are generating error correction training data.

Question: {question}

Incorrect Solution:
{incorrect_solution}

Your task:
1. Identify the error
2. Explain why it's wrong
3. Provide the correct solution

Respond in this format:
Error Identified: [what went wrong]
Why It's Wrong: [explanation]
Correct Solution:
<think>
[step-by-step correct reasoning]
</think>
Answer: [correct answer]"""


# ================================================================
# DUAL VIEW SAMPLE
# ================================================================


@dataclass
class DualViewSample:
    """A synthetic sample with both distilled and COT views."""

    # Core content
    id: str
    question: str
    answer: str

    # Dual views
    distilled_view: str  # <dist> content
    think_view: str | None  # <think> content (None if COT not allowed)

    # Metadata
    skill_bucket: str
    band: str
    stage: str
    language: str

    # Hard negative (optional)
    hard_negative: dict | None = None  # {reasoning, wrong_answer, error_type}

    # Error correction pair (optional)
    error_correction: dict | None = None  # {error, explanation, correction}

    # Quality signals
    verified: bool = False
    verification_score: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "distilled_view": self.distilled_view,
            "think_view": self.think_view,
            "skill_bucket": self.skill_bucket,
            "band": self.band,
            "stage": self.stage,
            "language": self.language,
            "hard_negative": self.hard_negative,
            "error_correction": self.error_correction,
            "verified": self.verified,
            "verification_score": self.verification_score,
        }

    def format_for_training(
        self,
        view: Literal["distilled", "think", "both"] = "both",
        include_special_tokens: bool = True,
    ) -> str:
        """Format sample for training."""

        parts = []

        if view in ("distilled", "both"):
            if include_special_tokens:
                parts.append(f"{DIST_START}\n{self.distilled_view}\n{DIST_END}")
            else:
                parts.append(self.distilled_view)

        if view in ("think", "both") and self.think_view:
            if include_special_tokens:
                parts.append(f"{THINK_START}\n{self.think_view}\n{THINK_END}")
            else:
                parts.append(self.think_view)

        return "\n\n".join(parts)


# ================================================================
# GENERATOR
# ================================================================


class DualViewGenerator:
    """Generates dual-view samples with COT gating."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        hard_negative_model: str | None = None,  # weaker model for negatives
    ):
        self.model = model
        self.hard_negative_model = hard_negative_model or model

    def generate(
        self,
        question: str,
        skill_bucket: str,
        band: str,
        stage: str = "SFT",
        language: str = "en",
        generate_hard_negative: bool = True,
        generate_error_correction: bool = True,
        sample_id: str | None = None,
    ) -> DualViewSample:
        """Generate a complete dual-view sample."""

        # Get band spec for COT policy
        band_spec = get_band_spec(band)
        cot_allowed = band_spec.cot_allowed
        cot_max_tokens = band_spec.cot_max_tokens or 0

        # Generate distilled view (always)
        distilled = self._generate_distilled(question)
        answer = self._extract_answer(distilled)

        # Generate COT view (if allowed for band)
        think_view = None
        if cot_allowed:
            think_view = self._generate_cot(question, cot_max_tokens)

        # Fallback: if distilled failed but COT succeeded, extract answer from COT
        if (not distilled or "Answer:" not in distilled) and think_view:
            answer_match = re.search(
                r"(?:Final\s+)?Answer[:\s]+(.+?)(?:\n|$)", think_view, re.IGNORECASE
            )
            if answer_match:
                extracted = answer_match.group(1).strip()
                distilled = f"Answer: {extracted}\nJustification: See reasoning above."
                answer = extracted

        # Generate hard negative
        hard_negative = None
        if generate_hard_negative and cot_allowed:
            hard_negative = self._generate_hard_negative(question, answer, skill_bucket)

        # Generate error correction pair
        error_correction = None
        if generate_error_correction and hard_negative:
            error_correction = self._generate_error_correction(question, hard_negative)

        # Validate before returning
        issues = self._validate_sample(distilled, think_view, band)
        if issues:
            print(f"    [WARN] Sample issues: {issues}")

        return DualViewSample(
            id=sample_id or f"{skill_bucket}-{hash(question) % 100000:05d}",
            question=question,
            answer=answer,
            distilled_view=distilled,
            think_view=think_view,
            skill_bucket=skill_bucket,
            band=band,
            stage=stage,
            language=language,
            hard_negative=hard_negative,
            error_correction=error_correction,
        )

    def _validate_sample(
        self, distilled: str, think_view: str | None, band: str
    ) -> list[str]:
        """Validate sample completeness."""
        issues = []

        if not distilled or len(distilled) < 10:
            issues.append("distilled too short")

        if distilled and "Answer:" not in distilled:
            issues.append("distilled missing 'Answer:'")

        # COT required for B3-B5
        if band in ("B3", "B4", "B5"):
            if not think_view or len(think_view) < 50:
                issues.append(f"COT required for {band} but missing/short")

        return issues

    def _generate_distilled(self, question: str, max_retries: int = 2) -> str:
        """Generate distilled view with retry on format failure."""
        prompt = DISTILLED_PROMPT.format(question=question)

        for attempt in range(max_retries + 1):
            response = ollama_chat(
                self.model,
                [{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.3 + (attempt * 0.1),  # Increase temp on retry
            )

            # Strip preambles
            response = self._strip_preamble(response)

            # Validate: must contain "Answer:"
            if "Answer:" in response:
                return response

            if attempt < max_retries:
                print(f"    [Retry {attempt+1}] distilled missing 'Answer:'")

        # Last resort: wrap raw response
        first_sentence = response.split(".")[0] if response else "Unknown"
        return f"Answer: {first_sentence}.\nJustification: {response}"

    def _strip_preamble(self, response: str) -> str:
        """Remove common LLM preambles (Qwen3 chattiness)."""
        if not response:
            return response

        patterns = [
            r"^(?:Okay|OK|Alright|Sure|Let me|I'll|I will|Here's|Here is)[,.]?\s*",
            r"^(?:Let's|Let us)\s+(?:solve|figure|work|think|see|break)[^.]*[.!]\s*",
            r"^(?:I need to|We need to|First,? I)[^.]*[.!]\s*",
        ]
        for pattern in patterns:
            response = re.sub(pattern, "", response, flags=re.IGNORECASE)
        return response.strip()

    def _generate_cot(self, question: str, max_tokens: int) -> str:
        """Generate COT view with step-by-step reasoning."""
        prompt = COT_PROMPT.format(question=question)
        response = ollama_chat(
            self.model,
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.5,
        )

        # Strip common preambles
        response = self._strip_preamble(response)

        # Try to extract <think> content first (backward compatible)
        think_match = re.search(
            r"<think>(.*?)</think>",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if think_match:
            return think_match.group(1).strip()

        # Otherwise, look for "Step 1:" pattern - return as-is if found
        if re.search(r"Step\s*1[:\.]", response, re.IGNORECASE):
            return response

        # Fallback: return cleaned response
        return response

    def _generate_hard_negative(
        self,
        question: str,
        correct_answer: str,
        skill_bucket: str,
    ) -> dict | None:
        """Generate a plausible but incorrect answer."""

        # Get failure modes for this skill
        skill = get_skill_bucket(skill_bucket)
        error_types = skill.failure_modes

        if not error_types:
            return None

        # Pick a random error type
        error_type = random.choice(error_types)
        error_mode = FAILURE_MODES[error_type]

        prompt = HARD_NEGATIVE_PROMPT.format(
            question=question,
            correct_answer=correct_answer,
            error_type=f"{error_mode.description}: {error_mode.detection_signal}",
        )

        try:
            response = ollama_chat(
                self.hard_negative_model,
                [{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.7,  # Higher temp for more varied mistakes
            )

            # Extract wrong answer
            wrong_match = re.search(
                r"Wrong Answer:\s*(.+?)(?:\n|Error Location:|$)",
                response,
                re.DOTALL,
            )
            wrong_answer = wrong_match.group(1).strip() if wrong_match else None

            # Extract reasoning
            think_match = re.search(
                r"<think>(.*?)</think>",
                response,
                re.DOTALL | re.IGNORECASE,
            )
            reasoning = think_match.group(1).strip() if think_match else response

            return {
                "reasoning": reasoning,
                "wrong_answer": wrong_answer,
                "error_type": error_type,
                "error_description": error_mode.description,
            }

        except Exception as e:
            print(f"Hard negative generation failed: {e}")
            return None

    def _generate_error_correction(
        self,
        question: str,
        hard_negative: dict,
    ) -> dict | None:
        """Generate error correction for the hard negative."""

        incorrect_solution = (
            f"Reasoning: {hard_negative['reasoning']}\n"
            f"Answer: {hard_negative['wrong_answer']}"
        )

        prompt = ERROR_CORRECTION_PROMPT.format(
            question=question,
            incorrect_solution=incorrect_solution,
        )

        try:
            response = ollama_chat(
                self.model,
                [{"role": "user", "content": prompt}],
                max_tokens=768,
                temperature=0.3,
            )

            # Extract components
            error_match = re.search(
                r"Error Identified:\s*(.+?)(?:\n|Why It's Wrong:|$)",
                response,
                re.DOTALL,
            )
            why_match = re.search(
                r"Why It's Wrong:\s*(.+?)(?:\n|Correct Solution:|$)",
                response,
                re.DOTALL,
            )

            return {
                "error_identified": (
                    error_match.group(1).strip() if error_match else None
                ),
                "explanation": why_match.group(1).strip() if why_match else None,
                "correct_solution": response,
            }

        except Exception as e:
            print(f"Error correction generation failed: {e}")
            return None

    def _extract_answer(self, distilled: str) -> str:
        """Extract answer from distilled view."""
        match = re.search(r"Answer:\s*(.+?)(?:\n|$)", distilled)
        return match.group(1).strip() if match else distilled.split("\n")[0]


# ================================================================
# BATCH GENERATION
# ================================================================


def generate_batch(
    questions: list[dict],  # [{question, skill_bucket, band, stage, language}]
    model: str = "qwen3:8b",
    hard_negatives: bool = True,
    error_corrections: bool = True,
    verbose: bool = True,
) -> list[DualViewSample]:
    """Generate a batch of dual-view samples."""

    generator = DualViewGenerator(model=model)
    samples = []

    for i, q in enumerate(questions):
        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(questions)}] Generating...")

        try:
            sample = generator.generate(
                question=q["question"],
                skill_bucket=q["skill_bucket"],
                band=q["band"],
                stage=q.get("stage", "SFT"),
                language=q.get("language", "en"),
                generate_hard_negative=hard_negatives,
                generate_error_correction=error_corrections,
                sample_id=q.get("id"),
            )
            samples.append(sample)
        except Exception as e:
            if verbose:
                print(f"  ERROR on {i}: {e}")

    return samples


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate dual-view samples")
    parser.add_argument("--model", default="qwen3:8b", help="Ollama model")
    parser.add_argument("--question", "-q", required=True, help="Question to process")
    parser.add_argument("--skill", "-s", default="RSN-ARITHMETIC", help="Skill bucket")
    parser.add_argument("--band", "-b", default="B3", help="Band (B0-B5)")
    parser.add_argument("--no-negative", action="store_true", help="Skip hard negative")

    args = parser.parse_args()

    generator = DualViewGenerator(model=args.model)
    sample = generator.generate(
        question=args.question,
        skill_bucket=args.skill,
        band=args.band,
        generate_hard_negative=not args.no_negative,
    )

    print("\n" + "=" * 60)
    print("DUAL VIEW SAMPLE")
    print("=" * 60)
    print(f"\nQuestion: {sample.question}")
    print(f"Band: {sample.band} | Skill: {sample.skill_bucket}")
    print(f"\n--- DISTILLED VIEW ---\n{sample.distilled_view}")
    if sample.think_view:
        print(f"\n--- THINK VIEW ---\n{sample.think_view[:500]}...")
    if sample.hard_negative:
        print("\n--- HARD NEGATIVE ---")
        print(f"Error Type: {sample.hard_negative['error_type']}")
        print(f"Wrong Answer: {sample.hard_negative['wrong_answer']}")
