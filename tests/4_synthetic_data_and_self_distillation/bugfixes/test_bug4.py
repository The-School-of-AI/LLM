"""
Test Bug Fix #4: Builtin Seeds Alias Resolution
=================================================

BUG (before fix):
    BUILTIN_SEEDS dict used legacy keys (e.g. "RSN-ARITHMETIC", "CODE-DEBUG"),
    but generate-bank --all iterates canonical keys from SKILL_BUCKETS
    (e.g. "RSN-ARITH", "CODE-DBG"). The get_builtin_seeds() function only
    did a direct dict lookup, so canonical IDs got placeholder garbage seeds:
        "Sample question 1 for RSN-ARITH"

    Similarly, SeedGenerator.generate() checked SEED_PROMPTS directly — canonical
    IDs like RSN-ARITH had no entry, falling to the generic prompt even though
    a good template existed under the legacy key RSN-ARITHMETIC.

FIX (after fix):
    1. Built reverse alias map _CANONICAL_TO_LEGACY from SKILL_ALIASES
       (e.g. "RSN-ARITH" → ["RSN-ARITHMETIC"])
    2. get_builtin_seeds() now tries 3 lookup strategies:
       - Step 1: Direct lookup (legacy keys, canonical keys that have entries)
       - Step 2: Canonical → legacy alias resolution
       - Step 3: Legacy → canonical resolution (for completeness)
       - Step 4: Fallback to placeholder (only if no seeds exist at all)
    3. SeedGenerator._resolve_prompt_key() uses same 3-step resolution
       for SEED_PROMPTS lookup.

FILES CHANGED:
    experiments/4_synthetic_data_and_self_distillation/generation/seed_generator.py
"""

import sys
from pathlib import Path

# ── Setup path so we can import from experiments/ ──────────────────────
EXPERIMENT_DIR = Path(__file__).resolve().parents[3] / "experiments" / "4_synthetic_data_and_self_distillation"
sys.path.insert(0, str(EXPERIMENT_DIR))

from generation.seed_generator import (
    BUILTIN_SEEDS,
    SEED_PROMPTS,
    get_builtin_seeds,
    SeedGenerator,
    _CANONICAL_TO_LEGACY,
)
from common.skills import SKILL_BUCKETS, SKILL_ALIASES


# Instantiate SeedGenerator without __init__ (no Ollama needed)
_gen = SeedGenerator.__new__(SeedGenerator)


# ======================================================================
# TEST 1: Reverse alias map correctness
# ======================================================================

class TestReverseAliasMap:
    """Verify _CANONICAL_TO_LEGACY is built correctly from SKILL_ALIASES."""

    def test_every_alias_is_mapped(self):
        """Every legacy alias should appear in the reverse map under its canonical."""
        for legacy, canonical in SKILL_ALIASES.items():
            assert canonical in _CANONICAL_TO_LEGACY, \
                f"Canonical '{canonical}' missing from _CANONICAL_TO_LEGACY"
            assert legacy in _CANONICAL_TO_LEGACY[canonical], \
                f"Legacy '{legacy}' missing from _CANONICAL_TO_LEGACY['{canonical}']"

    def test_rsn_arith_has_rsn_arithmetic(self):
        assert "RSN-ARITHMETIC" in _CANONICAL_TO_LEGACY.get("RSN-ARITH", [])

    def test_rsn_log_has_rsn_logic(self):
        assert "RSN-LOGIC" in _CANONICAL_TO_LEGACY.get("RSN-LOG", [])

    def test_code_dbg_has_code_debug(self):
        assert "CODE-DEBUG" in _CANONICAL_TO_LEGACY.get("CODE-DBG", [])

    def test_rsn_cs_has_both_know_aliases(self):
        """RSN-CS has two legacy aliases: KNOW-SCIENCE and KNOW-COMMONSENSE."""
        aliases = _CANONICAL_TO_LEGACY.get("RSN-CS", [])
        assert "KNOW-SCIENCE" in aliases
        assert "KNOW-COMMONSENSE" in aliases


# ======================================================================
# TEST 2: get_builtin_seeds() — canonical keys find real seeds
# ======================================================================

class TestBuiltinSeedsCanonicalResolution:
    """Canonical skill IDs should find the right builtin seeds via alias resolution."""

    def test_rsn_arith_finds_arithmetic_seeds(self):
        """RSN-ARITH (canonical) should find RSN-ARITHMETIC seeds (15)."""
        seeds = get_builtin_seeds("RSN-ARITH", 100)
        assert not any("Sample question" in s["question"] for s in seeds), \
            "RSN-ARITH got placeholder seeds instead of real arithmetic seeds"
        assert len(seeds) == 15

    def test_rsn_log_finds_logic_seeds(self):
        """RSN-LOG (canonical) should find RSN-LOGIC seeds (12)."""
        seeds = get_builtin_seeds("RSN-LOG", 100)
        assert not any("Sample question" in s["question"] for s in seeds)
        assert len(seeds) == 12

    def test_code_dbg_finds_debug_seeds(self):
        """CODE-DBG (canonical) should find CODE-DEBUG seeds (6)."""
        seeds = get_builtin_seeds("CODE-DBG", 100)
        assert not any("Sample question" in s["question"] for s in seeds)
        assert len(seeds) == 6

    def test_rsn_cs_finds_know_seeds(self):
        """RSN-CS (canonical) should find KNOW-SCIENCE or KNOW-COMMONSENSE seeds."""
        seeds = get_builtin_seeds("RSN-CS", 100)
        assert not any("Sample question" in s["question"] for s in seeds)
        assert len(seeds) == 10

    def test_code_gen_t1_finds_direct_seeds(self):
        """CODE-GEN-T1 has a direct entry in BUILTIN_SEEDS (5 seeds)."""
        seeds = get_builtin_seeds("CODE-GEN-T1", 100)
        assert not any("Sample question" in s["question"] for s in seeds)
        assert len(seeds) == 5


# ======================================================================
# TEST 3: get_builtin_seeds() — legacy keys still work
# ======================================================================

class TestBuiltinSeedsLegacyStillWork:
    """Legacy keys should still find their seeds via direct lookup."""

    def test_rsn_arithmetic_direct(self):
        seeds = get_builtin_seeds("RSN-ARITHMETIC", 100)
        assert not any("Sample question" in s["question"] for s in seeds)
        assert len(seeds) == 15

    def test_rsn_logic_direct(self):
        seeds = get_builtin_seeds("RSN-LOGIC", 100)
        assert not any("Sample question" in s["question"] for s in seeds)
        assert len(seeds) == 12

    def test_code_debug_direct(self):
        seeds = get_builtin_seeds("CODE-DEBUG", 100)
        assert not any("Sample question" in s["question"] for s in seeds)
        assert len(seeds) == 6

    def test_code_completion_direct(self):
        seeds = get_builtin_seeds("CODE-COMPLETION", 100)
        assert not any("Sample question" in s["question"] for s in seeds)
        assert len(seeds) == 8

    def test_know_commonsense_direct(self):
        seeds = get_builtin_seeds("KNOW-COMMONSENSE", 100)
        assert not any("Sample question" in s["question"] for s in seeds)
        assert len(seeds) == 10


# ======================================================================
# TEST 4: get_builtin_seeds() — metadata stamped correctly
# ======================================================================

class TestBuiltinSeedsMetadata:
    """Seeds should have correct id and skill_bucket metadata."""

    def test_canonical_id_stamped_on_canonical_lookup(self):
        """When looked up via canonical key, seeds get canonical skill_bucket."""
        seeds = get_builtin_seeds("RSN-ARITH", 3)
        for s in seeds:
            assert s["skill_bucket"] == "RSN-ARITH"
            assert s["id"].startswith("RSN-ARITH-SEED-")

    def test_legacy_id_stamped_on_legacy_lookup(self):
        """When looked up via legacy key, seeds get legacy skill_bucket."""
        seeds = get_builtin_seeds("RSN-ARITHMETIC", 3)
        for s in seeds:
            assert s["skill_bucket"] == "RSN-ARITHMETIC"
            assert s["id"].startswith("RSN-ARITHMETIC-SEED-")

    def test_ids_are_sequential(self):
        seeds = get_builtin_seeds("RSN-ARITH", 5)
        for i, s in enumerate(seeds):
            assert s["id"] == f"RSN-ARITH-SEED-{i+1:04d}"

    def test_all_seeds_have_question(self):
        """Every seed must have a non-empty 'question' field."""
        for skill_id in SKILL_BUCKETS:
            seeds = get_builtin_seeds(skill_id, 5)
            for s in seeds:
                assert "question" in s and s["question"], \
                    f"{skill_id}: seed missing question"


# ======================================================================
# TEST 5: _resolve_prompt_key() — all 45 canonical skills resolve
# ======================================================================

class TestResolvePromptKey:
    """Every canonical skill must resolve to a SEED_PROMPTS key (no None)."""

    def test_all_canonical_skills_resolve(self):
        """No canonical skill should return None from _resolve_prompt_key."""
        failures = []
        for skill_id in SKILL_BUCKETS:
            key = _gen._resolve_prompt_key(skill_id)
            if key is None:
                failures.append(skill_id)
        assert not failures, f"These canonical skills have no prompt template: {failures}"

    def test_rsn_arith_resolves_to_rsn_arithmetic(self):
        assert _gen._resolve_prompt_key("RSN-ARITH") == "RSN-ARITHMETIC"

    def test_rsn_alg_resolves_to_rsn_algebra(self):
        assert _gen._resolve_prompt_key("RSN-ALG") == "RSN-ALGEBRA"

    def test_rsn_log_resolves_to_rsn_logic(self):
        assert _gen._resolve_prompt_key("RSN-LOG") == "RSN-LOGIC"

    def test_rsn_caus_resolves_to_rsn_causal(self):
        assert _gen._resolve_prompt_key("RSN-CAUS") == "RSN-CAUSAL"

    def test_rsn_anal_resolves_to_rsn_analogical(self):
        assert _gen._resolve_prompt_key("RSN-ANAL") == "RSN-ANALOGICAL"

    def test_direct_match_preferred_over_alias(self):
        """Skills with direct SEED_PROMPTS entries should resolve to themselves."""
        for skill_id in ("CODE-GEN-T1", "INDIC-QA", "RSN-CS", "ALN-INST"):
            key = _gen._resolve_prompt_key(skill_id)
            assert key == skill_id, f"{skill_id} resolved to {key} instead of direct"

    def test_legacy_keys_also_resolve(self):
        """Legacy keys should resolve to themselves (direct match)."""
        for legacy in ("RSN-ARITHMETIC", "CODE-COMPLETION", "KNOW-FACTUAL"):
            key = _gen._resolve_prompt_key(legacy)
            assert key is not None, f"Legacy key {legacy} failed to resolve"


# ======================================================================
# TEST 6: Seed content quality — real seeds have real questions
# ======================================================================

class TestSeedContentQuality:
    """Verify that resolved seeds have meaningful content, not placeholders."""

    def test_rsn_arith_seeds_are_math_questions(self):
        """RSN-ARITH seeds should contain math-related words."""
        seeds = get_builtin_seeds("RSN-ARITH", 15)
        math_words = {"cost", "price", "calculate", "how much", "how many",
                      "divide", "multiply", "speed", "km", "percent", "%"}
        has_math = sum(1 for s in seeds
                       if any(w in s["question"].lower() for w in math_words))
        assert has_math >= 5, f"Only {has_math}/15 RSN-ARITH seeds look like math questions"

    def test_rsn_log_seeds_are_logic_questions(self):
        """RSN-LOG seeds should contain logic-related words."""
        seeds = get_builtin_seeds("RSN-LOG", 12)
        logic_words = {"if", "therefore", "all", "some", "conclude",
                       "implies", "older", "mammals", "square"}
        has_logic = sum(1 for s in seeds
                        if any(w in s["question"].lower() for w in logic_words))
        assert has_logic >= 5, f"Only {has_logic}/12 RSN-LOG seeds look like logic questions"

    def test_code_dbg_seeds_are_debug_problems(self):
        """CODE-DBG seeds should contain code or bug-fix related text."""
        seeds = get_builtin_seeds("CODE-DBG", 6)
        code_words = {"fix", "bug", "error", "should", "def ", "range"}
        has_code = sum(1 for s in seeds
                       if any(w in s["question"].lower() for w in code_words))
        assert has_code >= 3, f"Only {has_code}/6 CODE-DBG seeds look like debug problems"


# ======================================================================
# TEST 7: Edge case — many-to-one alias (KNOW-SCIENCE & KNOW-COMMONSENSE → RSN-CS)
# ======================================================================

class TestManyToOneAlias:
    """When two legacy keys map to one canonical, one set of seeds is used."""

    def test_rsn_cs_gets_seeds_from_one_legacy(self):
        """RSN-CS should get real seeds (not placeholders), from whichever legacy wins."""
        seeds = get_builtin_seeds("RSN-CS", 10)
        assert len(seeds) == 10
        assert not any("Sample question" in s["question"] for s in seeds)

    def test_rsn_cs_seeds_are_either_science_or_commonsense(self):
        """RSN-CS seeds should match either KNOW-SCIENCE or KNOW-COMMONSENSE."""
        rsn_cs_first = get_builtin_seeds("RSN-CS", 1)[0]["question"]
        science_first = BUILTIN_SEEDS["KNOW-SCIENCE"][0]["question"]
        commonsense_first = BUILTIN_SEEDS["KNOW-COMMONSENSE"][0]["question"]
        assert rsn_cs_first == science_first or rsn_cs_first == commonsense_first


# ======================================================================
# TEST 8: Edge case — direct entry takes priority over alias
# ======================================================================

class TestDirectPriority:
    """When a skill has both a direct BUILTIN_SEEDS entry and an alias match,
    the direct entry should win."""

    def test_code_gen_t1_uses_direct_not_alias(self):
        """CODE-GEN-T1 has direct entry (5 seeds). CODE-COMPLETION (8 seeds)
        is a legacy alias that also maps to CODE-GEN-T1, but should NOT be used."""
        seeds = get_builtin_seeds("CODE-GEN-T1", 100)
        # Direct entry has "Write a Python function..." seeds
        assert len(seeds) == 5
        assert any("Write a Python function" in s["question"] for s in seeds)
        # Should NOT have "Complete this function" seeds from CODE-COMPLETION
        assert not any("Complete this function" in s["question"] for s in seeds)


# ======================================================================
# Run with: uv run python -m pytest tests/4_synthetic_data_and_self_distillation/bugfixes/test_bug4.py -v
# ======================================================================

