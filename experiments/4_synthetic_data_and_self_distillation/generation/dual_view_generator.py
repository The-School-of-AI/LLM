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

from common.logging_config import get_logger

logger = get_logger("generation.dual_view_generator")

from common.bands import (
    Band,
    get_band_spec,
    cot_allowed_for_band,
    get_cot_max_tokens,
    THINK_START,
    THINK_END,
    DIST_START,
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
# OLD: hardcoded timeout=300 — too short for 70B models on consumer hardware
# NEW: configurable via OLLAMA_TIMEOUT env var (default 600s, use 1800+ for large models)
try:
    OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "600"))
except (ValueError, TypeError):
    OLLAMA_TIMEOUT = 600  # safe fallback if env var is empty or non-numeric


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
    
    logger.debug("ollama_chat -> model=%s, max_tokens=%d, temp=%.2f, timeout=%ds",
                 model, max_tokens, temperature, OLLAMA_TIMEOUT)
    logger.debug("  prompt: %s", messages[-1]["content"][:120].replace("\n", " "))
    
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    
    content = result.get("message", {}).get("content", "").strip()
    logger.debug("  response (%d chars): %s", len(content), content[:100].replace("\n", " "))
    return content


# ================================================================
# GENERATION PROMPTS
# ================================================================

# OLD: Single DISTILLED_PROMPT for all categories (caused "Unknown" for code/indic/translation)
# NEW: Category-specific distilled prompts that match the task type

# Default distilled prompt — math / reasoning / factual (answer + short justification)
DISTILLED_PROMPT_DEFAULT = """Solve this question. You MUST respond in this EXACT format:

Answer: [your answer here]
Justification: [1-2 sentence explanation]

Example:
Question: What is 2+2?
Answer: 4
Justification: Adding 2 and 2 gives 4.

Now solve:
Question: {question}"""

# Code distilled prompt — expects a code block as the answer
DISTILLED_PROMPT_CODE = """Provide the solution to this coding task. You MUST respond in this EXACT format:

Answer:
```python
[your code here]
```
Justification: [1-2 sentence explanation of approach]

Now solve:
{question}"""

# Translation distilled prompt — expects translated text
DISTILLED_PROMPT_TRANSLATION = """Complete this translation task. You MUST respond in this EXACT format:

Answer: [translated text here]
Justification: [1 sentence about the translation approach]

Do NOT respond with "Unknown" or "I don't know". Provide the best translation.

Now translate:
{question}"""

# Indic / multilingual distilled prompt — handles native script questions
DISTILLED_PROMPT_INDIC = """Answer the following question. The question may be in Hindi, Bengali, Tamil, Telugu, or another Indian language. Reply in the SAME language as the question.

You MUST respond in this EXACT format:
Answer: [your answer here]
Justification: [1-2 sentence explanation]

Do NOT respond with "Unknown". Provide the best answer you can.

Question: {question}"""

# Instruction-following distilled prompt
DISTILLED_PROMPT_INSTRUCTION = """Follow the instruction below precisely. You MUST respond in this EXACT format:

Answer: [your complete response following the instruction]
Justification: [1 sentence explaining how you followed the constraints]

Do NOT respond with "Unknown". Complete the task fully.

Instruction: {question}"""

# Code debugging distilled prompt — expects bug identification + fix
DISTILLED_PROMPT_DEBUG = """Identify the bug and provide the fix. You MUST respond in this EXACT format:

Answer: [the fixed code or one-line fix description]
Justification: [1-2 sentence explanation of the bug and fix]

Now fix:
{question}"""

# Code explanation distilled prompt — expects plain-English explanation
DISTILLED_PROMPT_EXPLAIN = """Explain what this code does. You MUST respond in this EXACT format:

Answer: [plain-English description of what the code does and its output]
Justification: [1 sentence supporting detail]

Now explain:
{question}"""

# Map from skill category/prefix to the appropriate distilled prompt
# OLD: single DISTILLED_PROMPT for everything
# NEW: category routing
DISTILLED_PROMPT_MAP = {
    "code_gen": DISTILLED_PROMPT_CODE,
    "code_debug": DISTILLED_PROMPT_DEBUG,
    "code_explain": DISTILLED_PROMPT_EXPLAIN,
    "translation": DISTILLED_PROMPT_TRANSLATION,
    "indic": DISTILLED_PROMPT_INDIC,
    "instruction": DISTILLED_PROMPT_INSTRUCTION,
    "default": DISTILLED_PROMPT_DEFAULT,
}

# Max tokens per category for distilled view
# OLD: hardcoded 256 for all (truncated code)
# NEW: category-appropriate limits
DISTILLED_MAX_TOKENS = {
    "code_gen": 512,
    "code_debug": 384,
    "code_explain": 384,
    "translation": 384,
    "indic": 384,
    "instruction": 384,
    "default": 256,
}


def _get_skill_prompt_category(skill_bucket: str) -> str:
    """Route a skill bucket to the appropriate prompt category.

    OLD: no routing — all skills used the same math-style prompt
    NEW: maps skill prefix/ID to the correct prompt template
    """
    # Normalize to uppercase for matching
    sb = skill_bucket.upper()

    # Code generation skills → code prompt
    if sb in ("CODE-GEN-T1", "CODE-GEN-T2", "CODE-GEN-T3", "CODE-SYN",
              "CODE-ALGO", "CODE-OPT", "CODE-TEST",
              "CODE-COMPLETION"):  # legacy alias
        return "code_gen"

    # Code debugging
    if sb in ("CODE-DBG", "CODE-DEBUG"):  # canonical + legacy
        return "code_debug"

    # Code explanation / comprehension
    if sb in ("CODE-COMP", "CODE-EXPLAIN"):  # canonical + legacy
        return "code_explain"

    # Translation skills
    if sb in ("LANG-TRANS", "INDIC-TRANS"):
        return "translation"

    # Indic language skills (non-translation)
    # OLD: LANG-MIX (Hinglish) was missing — fell through to "default" with 256 tokens
    # NEW: include LANG-MIX in the indic category
    if sb.startswith("INDIC-") or sb in ("LANG-HI-COMP", "LANG-HI-GEN",
                                          "LANG-HI-LOG", "LANG-HINDI",
                                          "LANG-MIX",
                                          "FND-LEX-HI", "RSN-MATH-HI"):
        return "indic"

    # Alignment / instruction-following skills
    if sb.startswith("ALN-"):
        return "instruction"

    # Everything else: math, reasoning, knowledge, foundation, production
    return "default"


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
        logger.debug("Generating distilled view for skill=%s, band=%s, lang=%s", skill_bucket, band, language)
        distilled = self._generate_distilled(question, skill_bucket=skill_bucket)
        answer = self._extract_answer(distilled)
        logger.debug("  Extracted answer (%d chars): %s", len(answer), repr(answer[:80]))

        # Generate COT view (if allowed for band)
        think_view = None
        if cot_allowed:
            logger.debug("  Generating COT view (max_tokens=%d)", cot_max_tokens)
            think_view = self._generate_cot(question, cot_max_tokens)
            logger.debug("  COT view: %d chars", len(think_view) if think_view else 0)
        else:
            logger.debug("  COT not allowed for band %s", band)

        # Fallback: if distilled failed but COT succeeded, extract answer from COT
        # Catches: empty, "Unknown", bare code fences (```python), and other garbage answers
        answer_is_bad = (
            not answer
            or answer.lower().strip().rstrip(".") in ("unknown", "no answer generated", "")
            or answer.strip() in ("```python", "```", "``")
        )
        distilled_is_bad = not distilled or "Answer:" not in distilled or answer_is_bad

        if distilled_is_bad and think_view:
            logger.debug("  Distilled answer is bad (answer=%s), attempting COT fallback", repr(answer[:40]) if answer else "empty")
            extracted = self._extract_answer_from_cot(think_view)
            if extracted:
                logger.info("  COT fallback extracted answer: %s", repr(extracted[:80]))
                if "```" in extracted:
                    distilled = f"Answer:\n{extracted}\nJustification: See step-by-step reasoning in think view."
                else:
                    distilled = f"Answer: {extracted}\nJustification: See reasoning above."
                answer = extracted
            else:
                logger.warning("  COT fallback failed — no usable answer extracted from think_view")

        # Generate hard negative
        hard_negative = None
        # Skip hard negative if we don't have a real answer
        # Catches: empty, "Unknown", bare code fences — prevents garbage "Correct Answer: ```python"
        has_valid_answer = (
            answer
            and answer.lower().strip().rstrip(".") not in ("unknown", "no answer generated", "")
            and answer.strip() not in ("```python", "```", "``")
        )
        if generate_hard_negative and cot_allowed and has_valid_answer:
            logger.debug("  Generating hard negative (correct_answer=%s)", repr(answer[:50]))
            hard_negative = self._generate_hard_negative(
                question, answer, skill_bucket
            )
            if hard_negative:
                logger.debug("  Hard negative: wrong_answer=%s", repr(str(hard_negative.get("wrong_answer", ""))[:60]))
            else:
                logger.debug("  Hard negative generation returned None")
        elif not has_valid_answer:
            logger.debug("  Skipping hard negative — no valid answer (answer=%s)", repr(answer[:40]) if answer else "empty")

        # Generate error correction pair
        error_correction = None
        if generate_error_correction and hard_negative:
            error_correction = self._generate_error_correction(question, hard_negative)

        # Validate before returning
        issues = self._validate_sample(distilled, think_view, band)
        if issues:
            logger.warning("  Sample quality issues: %s", issues)

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
    
    def _generate_distilled(
        self,
        question: str,
        skill_bucket: str = "",
        max_retries: int = 2,
    ) -> str:
        """Generate distilled view with retry on format failure.

        OLD: Used single DISTILLED_PROMPT for all skills — caused "Unknown" for code/indic/translation
        NEW: Routes to category-specific prompt via _get_skill_prompt_category()
        """
        category = _get_skill_prompt_category(skill_bucket)
        prompt_template = DISTILLED_PROMPT_MAP[category]
        max_tokens = DISTILLED_MAX_TOKENS[category]

        logger.debug("  _generate_distilled: category=%s, max_tokens=%d, skill=%s", category, max_tokens, skill_bucket)

        prompt = prompt_template.format(question=question)

        for attempt in range(max_retries + 1):
            response = ollama_chat(
                self.model,
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3 + (attempt * 0.1),  # Increase temp on retry
            )

            # Strip preambles
            response = self._strip_preamble(response)

            # Validate: must contain "Answer:" (or code block for code categories)
            if "Answer:" in response:
                return response
            # NEW: For code categories, accept response with code blocks even without "Answer:" prefix
            if category.startswith("code") and ("```" in response or "def " in response):
                return f"Answer:\n{response}"

            if attempt < max_retries:
                logger.info("    [Retry %d/%d] distilled missing 'Answer:' (category=%s)", attempt + 1, max_retries, category)

        # Last resort: wrap raw response
        if not response:
            return "Answer: No answer generated.\nJustification: LLM returned empty response."
        # For code categories, preserve code blocks in the fallback
        if category.startswith("code") and ("```" in response or "def " in response):
            return f"Answer:\n{response}\nJustification: Code generated without standard format."
        first_sentence = response.split('.')[0] if response else "No answer generated"
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
            logger.error("Hard negative generation failed: %s", e)
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
            logger.error("Error correction generation failed: %s", e)
            return None

    def _extract_answer(self, distilled: str) -> str:
        """Extract answer from distilled view.

        For code answers (```python ... ```), extracts the FULL code block.
        For text answers, extracts the single-line value after "Answer:".
        """
        if not distilled:
            return ""

        # Strategy 1: Extract full code block after "Answer:"
        # Handles: "Answer:\n```python\ndef foo():\n    pass\n```"
        code_match = re.search(
            r"Answer:\s*\n?\s*(```(?:python)?\s*\n.+?```)",
            distilled,
            re.DOTALL,
        )
        if code_match:
            return code_match.group(1).strip()

        # Strategy 2: Single-line answer (math, factual, etc.)
        match = re.search(r"Answer:\s*(.+?)(?:\n|$)", distilled)
        if match:
            candidate = match.group(1).strip()
            # Reject bare code fence markers — not a real answer
            if candidate in ("```python", "```", "``"):
                return ""
            return candidate

        return distilled.split("\n")[0]

    @staticmethod
    def _extract_answer_from_cot(think_view: str) -> str | None:
        """Extract a usable answer from CoT text when distilled view failed.

        Tries multiple strategies in order of reliability:
          1. "Final Answer: <value>"  — most explicit
          2. "Answer: <value>" (but NOT preamble text like "Here's the implementation")
          3. Last numeric value in the text (for math/arithmetic)
          4. Code block if present (for code tasks)
        Returns None if no reliable answer can be extracted.
        """
        if not think_view or not think_view.strip():
            return None

        # Strategy 1: "Final Answer:" — strongest signal
        final_match = re.search(
            r"Final\s+Answer[:\s]+(.+?)(?:\n|$)",
            think_view,
            re.IGNORECASE,
        )
        if final_match:
            candidate = final_match.group(1).strip()
            # Accept if it's not a filler phrase
            if candidate and candidate.lower() not in ("see above", "see below", "n/a"):
                return candidate

        # Strategy 2: "Answer:" — but skip preamble phrases and fillers
        answer_match = re.search(
            r"(?<!\w)Answer[:\s]+(.+?)(?:\n|$)",
            think_view,
            re.IGNORECASE,
        )
        if answer_match:
            candidate = answer_match.group(1).strip()
            # OLD BUG: accepted preamble like "Here's the implementation..."
            # NEW: reject common preamble patterns and filler phrases
            preamble_patterns = (
                "here's", "here is", "the following", "below is",
                "this is", "let me", "we can", "to solve",
            )
            filler_phrases = ("see above", "see below", "n/a", "none", "see reasoning")
            candidate_lower = candidate.lower()
            is_preamble = any(candidate_lower.startswith(p) for p in preamble_patterns)
            is_filler = candidate_lower in filler_phrases
            if candidate and not is_preamble and not is_filler:
                return candidate

        # Strategy 3: For math — extract last standalone number
        # Looks for a number at the end of text or after "=" or ":"
        num_match = re.search(
            r"(?:=|:|\bis\b)\s*(-?\d+(?:\.\d+)?(?:/\d+)?)\s*\.?\s*$",
            think_view,
            re.MULTILINE,
        )
        if num_match:
            return num_match.group(1).strip()

        # Strategy 4: For code — extract the last ```python ... ``` block
        code_blocks = re.findall(
            r"```(?:python)?\s*\n(.+?)```",
            think_view,
            re.DOTALL,
        )
        if code_blocks:
            # Return the last (usually most complete) code block
            return f"```python\n{code_blocks[-1].strip()}\n```"

        # No reliable answer found — return None so hard_negative is skipped
        # OLD BUG: used first_line of CoT as answer (grabbed "Step 1: ..." reasoning text)
        return None


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
            logger.info("[%d/%d] Generating...", i + 1, len(questions))
        
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
                logger.error("ERROR on sample %d: %s", i, e)
    
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
    # OLD: no --language flag
    # NEW: allow specifying language for Indic tasks
    parser.add_argument("--language", "-l", default="en", help="Language code (en, hi, bn, ta, te...)")
    parser.add_argument("--no-negative", action="store_true", help="Skip hard negative")

    args = parser.parse_args()

    generator = DualViewGenerator(model=args.model)
    sample = generator.generate(
        question=args.question,
        skill_bucket=args.skill,
        band=args.band,
        generate_hard_negative=not args.no_negative,
        # OLD: language was not passed — always defaulted to "en"
        # NEW: pass CLI language flag
        language=args.language,
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
