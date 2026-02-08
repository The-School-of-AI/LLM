"""
Test Bug Fix #8: max_tokens=256 Truncated Code Output
=======================================================

BUG (before fix):
    _generate_distilled() used a hardcoded max_tokens=256 for ALL categories.
    Code generation tasks (implementing functions, algorithms) need 512+ tokens
    for a complete code block + justification. At 256 tokens, code responses were
    truncated mid-function or returned empty, producing useless distilled views
    for all 9 code skills.

    This affected:
      - CODE-GEN-T1/T2/T3: Function implementations truncated
      - CODE-ALGO: Algorithm implementations cut off
      - CODE-SYN, CODE-OPT, CODE-TEST: Code outputs incomplete
      - CODE-DBG: Bug fix explanations truncated
      - CODE-COMP: Code explanations cut short

FIX (bundled into Bug 1):
    Added DISTILLED_MAX_TOKENS dict with category-appropriate limits:
      - code_gen:    512 tokens (doubled from 256)
      - code_debug:  384 tokens
      - code_explain: 384 tokens
      - translation: 384 tokens
      - indic:       384 tokens
      - instruction: 384 tokens
      - default:     256 tokens (unchanged — math/reasoning answers are short)

    Also found during review: LANG-MIX (Hinglish) was routing to 'default' (256)
    instead of 'indic' (384). Fixed by adding LANG-MIX to the indic category.

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
    DISTILLED_MAX_TOKENS,
)
from common.skills import SKILL_BUCKETS


# ======================================================================
# TEST 1: Code skills get >= 384 tokens (was 256)
# ======================================================================

class TestCodeSkillsGetMoreTokens:
    """All code skills must get >= 384 tokens for distilled view."""

    def test_code_gen_skills_get_512(self):
        """Code generation skills need 512 tokens for function implementations."""
        code_gen_skills = ["CODE-GEN-T1", "CODE-GEN-T2", "CODE-GEN-T3",
                           "CODE-ALGO", "CODE-SYN", "CODE-OPT", "CODE-TEST"]
        for skill_id in code_gen_skills:
            cat = _get_skill_prompt_category(skill_id)
            tokens = DISTILLED_MAX_TOKENS[cat]
            assert tokens >= 512, f"{skill_id} ({cat}) gets {tokens} tokens, need >= 512"

    def test_code_debug_gets_384(self):
        """Code debugging needs 384 tokens for fix + explanation."""
        cat = _get_skill_prompt_category("CODE-DBG")
        tokens = DISTILLED_MAX_TOKENS[cat]
        assert tokens >= 384, f"CODE-DBG ({cat}) gets {tokens} tokens, need >= 384"

    def test_code_explain_gets_384(self):
        """Code explanation needs 384 tokens for description."""
        cat = _get_skill_prompt_category("CODE-COMP")
        tokens = DISTILLED_MAX_TOKENS[cat]
        assert tokens >= 384, f"CODE-COMP ({cat}) gets {tokens} tokens, need >= 384"

    def test_no_code_skill_at_256(self):
        """No code skill should be at the old 256 limit."""
        code_prefixes = ("CODE-",)
        for skill_id in SKILL_BUCKETS:
            if any(skill_id.startswith(p) for p in code_prefixes):
                cat = _get_skill_prompt_category(skill_id)
                tokens = DISTILLED_MAX_TOKENS[cat]
                assert tokens > 256, f"{skill_id} still at 256 tokens (old bug)"


# ======================================================================
# TEST 2: Non-code skills have appropriate limits
# ======================================================================

class TestNonCodeSkillLimits:
    """Non-code categories should have their intended token limits."""

    def test_default_stays_at_256(self):
        """Math/reasoning/knowledge skills stay at 256 (their answers are short)."""
        assert DISTILLED_MAX_TOKENS["default"] == 256

    def test_translation_at_384(self):
        """Translation needs room for the translated text."""
        assert DISTILLED_MAX_TOKENS["translation"] >= 384

    def test_indic_at_384(self):
        """Indic languages need room for Devanagari/other scripts."""
        assert DISTILLED_MAX_TOKENS["indic"] >= 384

    def test_instruction_at_384(self):
        """Instruction-following needs room for formatted output."""
        assert DISTILLED_MAX_TOKENS["instruction"] >= 384


# ======================================================================
# TEST 3: Every category in DISTILLED_MAX_TOKENS is used
# ======================================================================

class TestAllCategoriesMapped:
    """Every category in DISTILLED_MAX_TOKENS must be reachable from at least one skill."""

    def test_all_token_categories_used(self):
        """Every entry in DISTILLED_MAX_TOKENS should be reachable by at least one skill."""
        used_categories = set()
        for skill_id in SKILL_BUCKETS:
            cat = _get_skill_prompt_category(skill_id)
            used_categories.add(cat)
        for cat in DISTILLED_MAX_TOKENS:
            assert cat in used_categories, \
                f"Category '{cat}' in DISTILLED_MAX_TOKENS is never used by any skill"

    def test_every_skill_has_token_limit(self):
        """Every skill must map to a category that exists in DISTILLED_MAX_TOKENS."""
        for skill_id in SKILL_BUCKETS:
            cat = _get_skill_prompt_category(skill_id)
            assert cat in DISTILLED_MAX_TOKENS, \
                f"{skill_id} maps to category '{cat}' with no token limit defined"


# ======================================================================
# TEST 4: No hardcoded max_tokens=256 in _generate_distilled
# ======================================================================

class TestNoHardcodedTokenLimit:
    """The old hardcoded max_tokens=256 must not exist in _generate_distilled."""

    def test_source_code_uses_variable_not_literal(self):
        """_generate_distilled should use DISTILLED_MAX_TOKENS[category], not a literal."""
        import inspect
        from generation.dual_view_generator import DualViewGenerator
        source = inspect.getsource(DualViewGenerator._generate_distilled)
        # Should reference the dict lookup
        assert "DISTILLED_MAX_TOKENS" in source, \
            "_generate_distilled does not reference DISTILLED_MAX_TOKENS"
        # Should NOT have a hardcoded 256 literal in actual code
        # (comments mentioning "OLD: max_tokens=256" are OK — only check non-comment lines)
        import re
        code_lines = [l for l in source.split("\n") if l.strip() and not l.strip().startswith("#")]
        code_only = "\n".join(code_lines)
        hardcoded = re.findall(r'max_tokens\s*=\s*256', code_only)
        assert not hardcoded, \
            f"Found hardcoded max_tokens=256 in _generate_distilled code: {hardcoded}"


# ======================================================================
# TEST 5: LANG-MIX (Hinglish) routes to indic, not default
# ======================================================================

class TestLangMixRouting:
    """LANG-MIX (Hinglish) should route to 'indic' category, not 'default'."""

    def test_lang_mix_is_indic(self):
        """LANG-MIX should get indic prompt and 384 tokens."""
        cat = _get_skill_prompt_category("LANG-MIX")
        assert cat == "indic", f"LANG-MIX routes to '{cat}', expected 'indic'"

    def test_lang_mix_gets_384_tokens(self):
        """LANG-MIX should get 384 tokens, not 256."""
        cat = _get_skill_prompt_category("LANG-MIX")
        tokens = DISTILLED_MAX_TOKENS[cat]
        assert tokens >= 384, f"LANG-MIX gets {tokens} tokens, need >= 384"


# ======================================================================
# TEST 6: Token limits are within reasonable bounds
# ======================================================================

class TestTokenLimitBounds:
    """Token limits should be sane — not too low or absurdly high."""

    def test_minimum_256(self):
        """No category should have less than 256 tokens."""
        for cat, limit in DISTILLED_MAX_TOKENS.items():
            assert limit >= 256, f"Category '{cat}' has {limit} tokens (< 256)"

    def test_maximum_1024(self):
        """No distilled view should need more than 1024 tokens.
        (CoT views can be longer, but distilled is meant to be concise.)"""
        for cat, limit in DISTILLED_MAX_TOKENS.items():
            assert limit <= 1024, f"Category '{cat}' has {limit} tokens (> 1024)"

    def test_code_gen_is_largest(self):
        """Code generation should have the highest distilled token limit."""
        code_gen_limit = DISTILLED_MAX_TOKENS["code_gen"]
        for cat, limit in DISTILLED_MAX_TOKENS.items():
            if cat != "code_gen":
                assert limit <= code_gen_limit, \
                    f"Category '{cat}' ({limit}) exceeds code_gen ({code_gen_limit})"


# ======================================================================
# TEST 7: Verify all 45 skills have correct token allocation
# ======================================================================

class TestFullSkillTokenAudit:
    """Every canonical skill must have an appropriate token allocation."""

    def test_code_skills_above_256(self):
        """All CODE-* skills must be above the old 256 limit."""
        for skill_id in SKILL_BUCKETS:
            if skill_id.startswith("CODE-"):
                cat = _get_skill_prompt_category(skill_id)
                tokens = DISTILLED_MAX_TOKENS[cat]
                assert tokens > 256, f"{skill_id} ({cat}) still at {tokens} tokens"

    def test_indic_skills_above_256(self):
        """All INDIC-* and Hindi skills must be above the old 256 limit."""
        indic_skills = [s for s in SKILL_BUCKETS
                        if s.startswith("INDIC-") or "HI" in s or s == "LANG-MIX"]
        for skill_id in indic_skills:
            cat = _get_skill_prompt_category(skill_id)
            tokens = DISTILLED_MAX_TOKENS[cat]
            assert tokens > 256, f"{skill_id} ({cat}) still at {tokens} tokens"

    def test_translation_skills_above_256(self):
        """Translation skills must be above 256."""
        for skill_id in ["LANG-TRANS", "INDIC-TRANS"]:
            cat = _get_skill_prompt_category(skill_id)
            tokens = DISTILLED_MAX_TOKENS[cat]
            assert tokens > 256, f"{skill_id} ({cat}) still at {tokens} tokens"

    def test_alignment_skills_above_256(self):
        """All ALN-* skills must be above 256."""
        for skill_id in SKILL_BUCKETS:
            if skill_id.startswith("ALN-"):
                cat = _get_skill_prompt_category(skill_id)
                tokens = DISTILLED_MAX_TOKENS[cat]
                assert tokens > 256, f"{skill_id} ({cat}) still at {tokens} tokens"

    def test_math_reasoning_at_256(self):
        """Math/reasoning skills are fine at 256 (short numerical answers)."""
        math_skills = ["RSN-ARITH", "RSN-ALG", "RSN-LOG", "RSN-WPT"]
        for skill_id in math_skills:
            cat = _get_skill_prompt_category(skill_id)
            tokens = DISTILLED_MAX_TOKENS[cat]
            assert tokens == 256, f"{skill_id} ({cat}) at {tokens}, expected 256"


# ======================================================================
# Run with: uv run python -m pytest tests/4_synthetic_data_and_self_distillation/bugfixes/test_bug8.py -v
# ======================================================================

