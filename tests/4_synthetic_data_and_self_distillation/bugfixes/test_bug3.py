"""
Test Bug Fix #3: Language Propagation from Seeds
==================================================

BUG (before fix):
    Indic samples (INDIC-QA, INDIC-TRANS, LANG-HI-COMP, etc.) were all tagged
    with language="en" regardless of the actual question language. The builtin
    seeds had "language": "hi", "language": "bn" etc. but this field was never
    read or passed through to DualViewGenerator.generate(). The generator's
    default parameter `language="en"` was used for every sample.

    This affected 3 code paths:
      1. cmd_generate_bank() — main pipeline (fixed)
      2. cmd_generate() — single-skill CLI (was NOT fixed, now fixed)
      3. dual_view_generator.py CLI __main__ (was NOT fixed, now fixed)

    Additionally:
      - INDIC-TRANS builtin seeds used "source_lang"/"target_lang" but not
        "language", so even with the fix they fell to skill.languages[0]="hi"
        for all samples including HI→EN translations.
      - LANG-TRANS seeds had no "language" key at all.

FIX (after fix):
    1. cmd_generate_bank(): reads seed.get("language") with fallback to
       skill.languages[0] — passes it to generator (already done)
    2. cmd_generate(): same pattern added (was missing)
    3. dual_view_generator.py CLI: added --language flag
    4. INDIC-TRANS builtin seeds: added "language" key set to target_lang
    5. LANG-TRANS builtin seeds: added "language" key set to target language

FILES CHANGED:
    experiments/4_synthetic_data_and_self_distillation/run_pipeline.py
    experiments/4_synthetic_data_and_self_distillation/generation/dual_view_generator.py
    experiments/4_synthetic_data_and_self_distillation/generation/seed_generator.py
"""

import sys
from pathlib import Path

# ── Setup path so we can import from experiments/ ──────────────────────
EXPERIMENT_DIR = Path(__file__).resolve().parents[3] / "experiments" / "4_synthetic_data_and_self_distillation"
sys.path.insert(0, str(EXPERIMENT_DIR))

from common.skills import SKILL_BUCKETS, get_skill_bucket, SKILL_ALIASES
from generation.seed_generator import get_builtin_seeds, BUILTIN_SEEDS


# ======================================================================
# TEST 1: Builtin seeds for Indic skills have "language" field
# ======================================================================

class TestIndicSeedsHaveLanguageField:
    """Verify that builtin seeds for Indic/translation skills carry language metadata."""

    def test_indic_qa_seeds_have_language(self):
        """Every INDIC-QA seed must have a 'language' key."""
        seeds = get_builtin_seeds("INDIC-QA", 100)
        for seed in seeds:
            assert "language" in seed, f"INDIC-QA seed missing 'language': {seed['question'][:50]}"

    def test_indic_qa_seeds_are_not_en(self):
        """INDIC-QA seeds should be in Indian languages, not all 'en'."""
        seeds = get_builtin_seeds("INDIC-QA", 100)
        languages = {s["language"] for s in seeds}
        # Should have at least 2 different languages (hi, bn, ta, te)
        assert len(languages) >= 2, f"Expected multiple languages, got: {languages}"
        assert "en" not in languages or len(languages) > 1, \
            "INDIC-QA seeds should not be all 'en'"

    def test_indic_qa_has_hindi_bengali_tamil(self):
        """INDIC-QA seeds should cover hi, bn, ta, te."""
        seeds = get_builtin_seeds("INDIC-QA", 100)
        languages = {s["language"] for s in seeds}
        for expected in ("hi", "bn", "ta", "te"):
            assert expected in languages, f"Missing language '{expected}' in INDIC-QA seeds: {languages}"

    def test_indic_trans_seeds_have_language(self):
        """Every INDIC-TRANS seed must have a 'language' key."""
        seeds = get_builtin_seeds("INDIC-TRANS", 100)
        for seed in seeds:
            assert "language" in seed, f"INDIC-TRANS seed missing 'language': {seed['question'][:50]}"

    def test_indic_trans_language_matches_target(self):
        """INDIC-TRANS seed 'language' should match 'target_lang' when both present."""
        seeds = get_builtin_seeds("INDIC-TRANS", 100)
        for seed in seeds:
            if "target_lang" in seed:
                assert seed["language"] == seed["target_lang"], (
                    f"INDIC-TRANS seed language '{seed['language']}' != target_lang '{seed['target_lang']}'"
                )

    def test_indic_trans_hindi_to_english_is_en(self):
        """HI→EN translation should have language='en', not 'hi'."""
        seeds = get_builtin_seeds("INDIC-TRANS", 100)
        hi_to_en = [s for s in seeds if s.get("source_lang") == "hi" and s.get("target_lang") == "en"]
        for seed in hi_to_en:
            assert seed["language"] == "en", (
                f"HI→EN translation should have language='en', got '{seed['language']}'"
            )

    def test_indic_sent_seeds_have_language(self):
        """INDIC-SENT seeds must have a 'language' key."""
        if "INDIC-SENT" in BUILTIN_SEEDS:
            seeds = get_builtin_seeds("INDIC-SENT", 100)
            for seed in seeds:
                assert "language" in seed, f"INDIC-SENT seed missing 'language': {seed['question'][:50]}"

    def test_lang_trans_seeds_have_language(self):
        """LANG-TRANS seeds must have a 'language' key."""
        seeds = get_builtin_seeds("LANG-TRANS", 100)
        for seed in seeds:
            assert "language" in seed, f"LANG-TRANS seed missing 'language': {seed['question'][:50]}"

    def test_lang_trans_en_to_hi_is_hi(self):
        """EN→HI translation seeds should have language='hi'."""
        seeds = get_builtin_seeds("LANG-TRANS", 100)
        hindi_seeds = [s for s in seeds if "Hindi" in s["question"] and "Translate to Hindi" in s["question"]]
        for seed in hindi_seeds:
            assert seed["language"] == "hi", (
                f"EN→HI translation should have language='hi', got '{seed['language']}'"
            )

    def test_lang_trans_hi_to_en_is_en(self):
        """HI→EN translation seeds should have language='en'."""
        seeds = get_builtin_seeds("LANG-TRANS", 100)
        english_seeds = [s for s in seeds if "Translate to English" in s["question"]]
        for seed in english_seeds:
            assert seed["language"] == "en", (
                f"HI→EN translation should have language='en', got '{seed['language']}'"
            )


# ======================================================================
# TEST 2: Skill language fallback — when seed has no "language" key
# ======================================================================

class TestSkillLanguageFallback:
    """Verify that the fallback to skill.languages[0] is correct for each skill."""

    def test_english_skills_default_to_en(self):
        """English-primary skills should fall back to 'en'."""
        en_skills = ["RSN-ARITH", "RSN-LOG", "CODE-GEN-T1", "FND-LEX-EN", "PRD-SUM"]
        for skill_id in en_skills:
            skill = get_skill_bucket(skill_id)
            fallback = skill.languages[0] if skill.languages else "en"
            assert fallback == "en", f"{skill_id} fallback is '{fallback}', expected 'en'"

    def test_hindi_skills_default_to_hi(self):
        """Hindi-primary skills should fall back to 'hi'."""
        hi_skills = ["FND-LEX-HI", "RSN-MATH-HI", "LANG-HI-COMP", "LANG-HI-GEN", "LANG-HI-LOG"]
        for skill_id in hi_skills:
            skill = get_skill_bucket(skill_id)
            fallback = skill.languages[0] if skill.languages else "en"
            assert fallback == "hi", f"{skill_id} fallback is '{fallback}', expected 'hi'"

    def test_indic_skills_default_to_hi(self):
        """Indic multi-language skills should fall back to 'hi' (first language)."""
        indic_skills = ["INDIC-QA", "INDIC-NLI", "INDIC-SENT", "INDIC-NER"]
        for skill_id in indic_skills:
            skill = get_skill_bucket(skill_id)
            fallback = skill.languages[0] if skill.languages else "en"
            assert fallback == "hi", f"{skill_id} fallback is '{fallback}', expected 'hi'"

    def test_hinglish_defaults_to_hi_en(self):
        """LANG-MIX (Hinglish) should fall back to 'hi-en'."""
        skill = get_skill_bucket("LANG-MIX")
        fallback = skill.languages[0] if skill.languages else "en"
        assert fallback == "hi-en", f"LANG-MIX fallback is '{fallback}', expected 'hi-en'"


# ======================================================================
# TEST 3: Seed language propagation simulation
# ======================================================================

class TestLanguagePropagation:
    """Simulate the language extraction logic from cmd_generate_bank."""

    def _extract_language(self, seed: dict, skill_id: str) -> str:
        """Replicate the language extraction logic from cmd_generate_bank."""
        skill = get_skill_bucket(skill_id)
        return seed.get("language", skill.languages[0] if skill.languages else "en")

    def test_hindi_seed_returns_hi(self):
        """Seed with language='hi' should return 'hi'."""
        seed = {"question": "भारत की राजधानी क्या है?", "language": "hi"}
        assert self._extract_language(seed, "INDIC-QA") == "hi"

    def test_bengali_seed_returns_bn(self):
        """Seed with language='bn' should return 'bn'."""
        seed = {"question": "ভারতের রাজধানী কী?", "language": "bn"}
        assert self._extract_language(seed, "INDIC-QA") == "bn"

    def test_tamil_seed_returns_ta(self):
        """Seed with language='ta' should return 'ta'."""
        seed = {"question": "இந்தியாவின் தலைநகரம் என்ன?", "language": "ta"}
        assert self._extract_language(seed, "INDIC-QA") == "ta"

    def test_english_seed_explicit(self):
        """Seed with language='en' returns 'en'."""
        seed = {"question": "What is 2+2?", "language": "en"}
        assert self._extract_language(seed, "RSN-ARITH") == "en"

    def test_no_language_key_english_skill(self):
        """Seed without 'language' key on English skill falls back to 'en'."""
        seed = {"question": "Solve: 3x + 7 = 22"}
        assert self._extract_language(seed, "RSN-ARITH") == "en"

    def test_no_language_key_indic_skill(self):
        """Seed without 'language' key on Indic skill falls back to 'hi'."""
        seed = {"question": "Some Indic question"}
        assert self._extract_language(seed, "INDIC-QA") == "hi"

    def test_no_language_key_hindi_skill(self):
        """Seed without 'language' key on Hindi skill falls back to 'hi'."""
        seed = {"question": "गणित का प्रश्न"}
        assert self._extract_language(seed, "RSN-MATH-HI") == "hi"

    def test_placeholder_seed_indic(self):
        """Placeholder seeds (no language key) on Indic skill should get 'hi'."""
        seed = {"question": "Sample question 1 for INDIC-NER"}
        assert self._extract_language(seed, "INDIC-NER") == "hi"

    def test_placeholder_seed_english(self):
        """Placeholder seeds on English skill should get 'en'."""
        seed = {"question": "Sample question 1 for RSN-ARITH"}
        assert self._extract_language(seed, "RSN-ARITH") == "en"

    def test_trans_en_to_hi_seed(self):
        """INDIC-TRANS EN→HI seed with explicit language='hi'."""
        seed = {"question": "Translate to Hindi: 'Knowledge is power.'", "language": "hi"}
        assert self._extract_language(seed, "INDIC-TRANS") == "hi"

    def test_trans_hi_to_en_seed(self):
        """INDIC-TRANS HI→EN seed with explicit language='en'."""
        seed = {"question": "Translate to English: 'जल ही जीवन है।'", "language": "en"}
        assert self._extract_language(seed, "INDIC-TRANS") == "en"

    def test_mixed_language_seed(self):
        """INDIC-SENT code-mixed seed with language='mix'."""
        seed = {"question": "Is this positive? 'Product is bakwas'", "language": "mix"}
        assert self._extract_language(seed, "INDIC-SENT") == "mix"


# ======================================================================
# TEST 4: Every skill has a non-empty languages list
# ======================================================================

class TestAllSkillsHaveLanguages:
    """Every SkillBucket must have at least one language defined."""

    def test_all_skills_have_languages(self):
        """Every canonical skill must have a non-empty languages list."""
        for skill_id, skill in SKILL_BUCKETS.items():
            assert skill.languages, f"{skill_id} has empty languages list"
            assert len(skill.languages) >= 1, f"{skill_id} has no languages"

    def test_all_skills_first_language_is_valid(self):
        """The first language (fallback) should be a recognized code."""
        valid_codes = {"en", "hi", "bn", "ta", "te", "mr", "gu", "kn",
                       "ml", "pa", "or", "as", "hi-en"}
        for skill_id, skill in SKILL_BUCKETS.items():
            first = skill.languages[0]
            assert first in valid_codes, f"{skill_id}.languages[0] = '{first}' not in valid codes"


# ======================================================================
# Run with: uv run python -m pytest tests/4_synthetic_data_and_self_distillation/bugfixes/test_bug3.py -v
# ======================================================================

