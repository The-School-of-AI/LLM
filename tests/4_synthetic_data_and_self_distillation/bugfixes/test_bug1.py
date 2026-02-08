"""
Test Bug Fix #1: Category-Specific Distilled Prompts
=====================================================

BUG (before fix):
    A single DISTILLED_PROMPT was used for ALL skill categories. It was
    hard-coded for math-style Q&A ("Solve this question ... Answer: ... Justification: ...").
    When code, translation, Indic, or instruction-following tasks were passed through this
    prompt, the LLM could not follow the format and returned "Answer: Unknown." or empty
    strings. This caused >60% of generated samples to have broken distilled_view fields.

FIX (after fix):
    Added 7 category-specific prompt templates routed by _get_skill_prompt_category():
      - DISTILLED_PROMPT_DEFAULT   : math, reasoning, factual, knowledge
      - DISTILLED_PROMPT_CODE      : code generation (expects code blocks)
      - DISTILLED_PROMPT_DEBUG     : code debugging (expects bug + fix)
      - DISTILLED_PROMPT_EXPLAIN   : code comprehension (expects English explanation)
      - DISTILLED_PROMPT_TRANSLATION : translation tasks
      - DISTILLED_PROMPT_INDIC     : Indic/multilingual (replies in same script)
      - DISTILLED_PROMPT_INSTRUCTION : alignment/instruction-following

    Also increased max_tokens from 256 (hardcoded) to category-appropriate limits
    via DISTILLED_MAX_TOKENS (512 for code, 384 for others, 256 for default).

FILES CHANGED:
    experiments/4_synthetic_data_and_self_distillation/generation/dual_view_generator.py
"""

import sys
from pathlib import Path

# ── Setup path so we can import from experiments/ ──────────────────────
EXPERIMENT_DIR = Path(__file__).resolve().parents[3] / "experiments" / "4_synthetic_data_and_self_distillation"
sys.path.insert(0, str(EXPERIMENT_DIR))

from generation.dual_view_generator import (
    _get_skill_prompt_category,
    DISTILLED_PROMPT_MAP,
    DISTILLED_MAX_TOKENS,
    DISTILLED_PROMPT_CODE,
    DISTILLED_PROMPT_DEBUG,
    DISTILLED_PROMPT_EXPLAIN,
    DISTILLED_PROMPT_TRANSLATION,
    DISTILLED_PROMPT_INDIC,
    DISTILLED_PROMPT_INSTRUCTION,
    DISTILLED_PROMPT_DEFAULT,
)


# ======================================================================
# TEST 1: Routing — every canonical skill maps to the correct category
# ======================================================================

class TestSkillPromptRouting:
    """Verify _get_skill_prompt_category routes every skill to the right prompt."""

    # ── Code generation skills ────────────────────────────────────────
    def test_code_gen_canonical(self):
        """Canonical code generation IDs route to 'code_gen'."""
        for skill_id in ("CODE-GEN-T1", "CODE-GEN-T2", "CODE-GEN-T3",
                         "CODE-SYN", "CODE-ALGO", "CODE-OPT", "CODE-TEST"):
            result = _get_skill_prompt_category(skill_id)
            assert result == "code_gen", f"{skill_id} → {result}, expected code_gen"

    def test_code_gen_legacy_alias(self):
        """Legacy alias CODE-COMPLETION also routes to 'code_gen'."""
        assert _get_skill_prompt_category("CODE-COMPLETION") == "code_gen"

    def test_code_gen_case_insensitive(self):
        """Routing should be case-insensitive."""
        assert _get_skill_prompt_category("code-gen-t1") == "code_gen"
        assert _get_skill_prompt_category("Code-Algo") == "code_gen"

    # ── Code debugging ────────────────────────────────────────────────
    def test_code_debug_canonical(self):
        assert _get_skill_prompt_category("CODE-DBG") == "code_debug"

    def test_code_debug_legacy(self):
        assert _get_skill_prompt_category("CODE-DEBUG") == "code_debug"

    # ── Code explanation / comprehension ──────────────────────────────
    def test_code_explain_canonical(self):
        assert _get_skill_prompt_category("CODE-COMP") == "code_explain"

    def test_code_explain_legacy(self):
        assert _get_skill_prompt_category("CODE-EXPLAIN") == "code_explain"

    # ── Translation skills ────────────────────────────────────────────
    def test_translation_lang_trans(self):
        assert _get_skill_prompt_category("LANG-TRANS") == "translation"

    def test_translation_indic_trans(self):
        assert _get_skill_prompt_category("INDIC-TRANS") == "translation"

    # ── Indic language skills (non-translation) ───────────────────────
    def test_indic_prefix(self):
        """All INDIC-* skills (except INDIC-TRANS handled above) route to 'indic'."""
        for skill_id in ("INDIC-QA", "INDIC-NLI", "INDIC-SENT", "INDIC-NER"):
            result = _get_skill_prompt_category(skill_id)
            assert result == "indic", f"{skill_id} → {result}, expected indic"

    def test_indic_hindi_skills(self):
        """Hindi-specific skill IDs route to 'indic'."""
        for skill_id in ("LANG-HI-COMP", "LANG-HI-GEN", "LANG-HI-LOG",
                         "LANG-HINDI", "FND-LEX-HI", "RSN-MATH-HI"):
            result = _get_skill_prompt_category(skill_id)
            assert result == "indic", f"{skill_id} → {result}, expected indic"

    # ── Alignment / instruction-following ─────────────────────────────
    def test_alignment_skills(self):
        for skill_id in ("ALN-INST", "ALN-STRUCT", "ALN-HALL", "ALN-SAFE", "ALN-HELP"):
            result = _get_skill_prompt_category(skill_id)
            assert result == "instruction", f"{skill_id} → {result}, expected instruction"

    # ── Default (math, reasoning, knowledge, production) ──────────────
    def test_default_reasoning(self):
        for skill_id in ("RSN-ARITH", "RSN-ALG", "RSN-LOG", "RSN-CAUS",
                         "RSN-WPT", "RSN-ADVMATH", "RSN-CS", "RSN-MH",
                         "RSN-CONTRADICTION", "RSN-ANAL"):
            result = _get_skill_prompt_category(skill_id)
            assert result == "default", f"{skill_id} → {result}, expected default"

    def test_default_foundation(self):
        for skill_id in ("FND-LEX-EN", "FND-SEM", "FND-DIS", "FND-LCX", "FND-FACT"):
            result = _get_skill_prompt_category(skill_id)
            assert result == "default", f"{skill_id} → {result}, expected default"

    def test_default_production(self):
        for skill_id in ("PRD-ROB", "PRD-SUM", "PRD-IE"):
            result = _get_skill_prompt_category(skill_id)
            assert result == "default", f"{skill_id} → {result}, expected default"

    def test_default_legacy_reasoning(self):
        """Legacy reasoning aliases route to default."""
        for skill_id in ("RSN-ARITHMETIC", "RSN-ALGEBRA", "RSN-LOGIC", "RSN-CAUSAL"):
            result = _get_skill_prompt_category(skill_id)
            assert result == "default", f"{skill_id} → {result}, expected default"


# ======================================================================
# TEST 2: Prompt map completeness — every category has a valid template
# ======================================================================

class TestPromptMapCompleteness:
    """Verify DISTILLED_PROMPT_MAP covers all categories returned by the router."""

    def test_all_categories_have_prompts(self):
        """Every category string returned by the router exists in DISTILLED_PROMPT_MAP."""
        all_categories = {"code_gen", "code_debug", "code_explain",
                          "translation", "indic", "instruction", "default"}
        for cat in all_categories:
            assert cat in DISTILLED_PROMPT_MAP, f"Category '{cat}' missing from DISTILLED_PROMPT_MAP"
            assert cat in DISTILLED_MAX_TOKENS, f"Category '{cat}' missing from DISTILLED_MAX_TOKENS"

    def test_prompts_contain_format_placeholder(self):
        """Every prompt template must contain {question} placeholder."""
        for cat, prompt in DISTILLED_PROMPT_MAP.items():
            assert "{question}" in prompt, f"Prompt for '{cat}' missing {{question}} placeholder"

    def test_prompts_require_answer_format(self):
        """Every prompt should instruct the model to include 'Answer:' in its response."""
        for cat, prompt in DISTILLED_PROMPT_MAP.items():
            assert "Answer:" in prompt, f"Prompt for '{cat}' doesn't instruct model to use 'Answer:' format"


# ======================================================================
# TEST 3: Token limits — code gets more, default gets standard
# ======================================================================

class TestTokenLimits:
    """Verify max_tokens are appropriate per category."""

    def test_code_gets_more_tokens(self):
        """Code generation needs >= 512 tokens (was 256 before fix)."""
        assert DISTILLED_MAX_TOKENS["code_gen"] >= 512

    def test_default_stays_reasonable(self):
        """Default (math/reasoning) can stay at 256."""
        assert DISTILLED_MAX_TOKENS["default"] == 256

    def test_all_limits_above_minimum(self):
        """No category should have less than 256 tokens."""
        for cat, limit in DISTILLED_MAX_TOKENS.items():
            assert limit >= 256, f"Category '{cat}' has max_tokens={limit}, expected >= 256"


# ======================================================================
# TEST 4: Prompt content quality — category-specific keywords present
# ======================================================================

class TestPromptContent:
    """Verify each prompt template has category-appropriate instructions."""

    def test_code_prompt_expects_code_block(self):
        """Code prompt should mention python or code block."""
        prompt = DISTILLED_PROMPT_MAP["code_gen"]
        assert "python" in prompt.lower() or "```" in prompt

    def test_translation_prompt_warns_against_unknown(self):
        """Translation prompt should explicitly forbid 'Unknown' answers."""
        prompt = DISTILLED_PROMPT_MAP["translation"]
        assert "Unknown" in prompt or "unknown" in prompt

    def test_indic_prompt_mentions_indian_languages(self):
        """Indic prompt should mention Indian languages by name."""
        prompt = DISTILLED_PROMPT_MAP["indic"]
        assert "Hindi" in prompt or "Bengali" in prompt or "Tamil" in prompt

    def test_indic_prompt_warns_against_unknown(self):
        """Indic prompt should explicitly forbid 'Unknown' answers."""
        prompt = DISTILLED_PROMPT_MAP["indic"]
        assert "Unknown" in prompt

    def test_instruction_prompt_mentions_follow(self):
        """Instruction prompt should mention following instructions."""
        prompt = DISTILLED_PROMPT_MAP["instruction"]
        assert "instruction" in prompt.lower() or "follow" in prompt.lower()

    def test_debug_prompt_mentions_fix(self):
        """Debug prompt should mention identifying bugs or fixing."""
        prompt = DISTILLED_PROMPT_MAP["code_debug"]
        assert "fix" in prompt.lower() or "bug" in prompt.lower()

    def test_explain_prompt_mentions_explain(self):
        """Explain prompt should mention explaining code behavior."""
        prompt = DISTILLED_PROMPT_MAP["code_explain"]
        assert "explain" in prompt.lower() or "what" in prompt.lower()


# ======================================================================
# Run with: python -m pytest tests/4_synthetic_data_and_self_distillation/bugfixes/test_bug1.py -v
# ======================================================================

