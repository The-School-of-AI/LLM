"""
Test Bug Fix #5: CODE-COMPLETION Seeds Pre-Solved
===================================================

BUG (before fix):
    The generated CODE-COMPLETION.jsonl contained 5 samples where the "question"
    field had COMPLETE function implementations (is_palindrome, binary_search,
    bubble_sort, etc.). The LLM then responded "The function is already correctly
    implemented" — producing zero training value.

    Root cause analysis revealed TWO issues:
    1. The builtin seeds (BUILTIN_SEEDS["CODE-COMPLETION"]) were actually
       properly incomplete — NOT the problem.
    2. The JSONL was generated using LLM-generated seeds (SeedGenerator.generate()),
       and the LLM ignored the "Partial function definitions" instruction in the
       CODE-COMPLETION prompt, generating complete functions instead.

FIX (resolved by Bugs 4+6):
    With manifest normalization (Bug 6), CODE-COMPLETION → CODE-GEN-T1 (canonical).
    CODE-GEN-T1 has:
      - Its own BUILTIN_SEEDS: 5 "Write a Python function..." seeds (from-scratch)
      - Its own SEED_PROMPT: "Generate simple code generation problems" (no confusion)
    The broken CODE-COMPLETION prompt ("Partial function definitions to complete")
    is no longer reachable through canonical workflows.

    The existing CODE-COMPLETION.jsonl in the bank needs regeneration.

FILES CHANGED:
    Resolved by Bug 4 (alias resolution) and Bug 6 (manifest normalization)
    in generation/seed_generator.py and run_pipeline.py
"""

import sys
from pathlib import Path

# ── Setup path so we can import from experiments/ ──────────────────────
EXPERIMENT_DIR = Path(__file__).resolve().parents[3] / "experiments" / "4_synthetic_data_and_self_distillation"
sys.path.insert(0, str(EXPERIMENT_DIR))

from generation.seed_generator import BUILTIN_SEEDS, SEED_PROMPTS, get_builtin_seeds, SeedGenerator
from common.skills import get_skill_bucket


_gen = SeedGenerator.__new__(SeedGenerator)


# ======================================================================
# TEST 1: Builtin CODE-COMPLETION seeds ARE properly incomplete
# ======================================================================

class TestCodeCompletionBuiltinSeedsAreIncomplete:
    """Verify that the 8 CODE-COMPLETION builtin seeds are properly incomplete
    (end mid-line, NOT complete functions)."""

    def test_all_seeds_end_incomplete(self):
        """Every CODE-COMPLETION seed should end mid-statement (no complete function body)."""
        seeds = BUILTIN_SEEDS["CODE-COMPLETION"]
        for i, seed in enumerate(seeds):
            lines = seed["question"].strip().split("\n")
            last_line = lines[-1].strip()
            # An incomplete seed ends with: bare "return", "return s ==",
            # "if condition:", "while condition:", "elif condition:" etc.
            is_incomplete = (
                last_line == "return"
                or last_line.endswith("==")
                or last_line.endswith(":")
            )
            assert is_incomplete, (
                f"Builtin CODE-COMPLETION seed [{i}] looks complete. "
                f"Last line: '{last_line}'"
            )

    def test_no_seed_has_full_return_value(self):
        """No seed should have a final return with a complete computed value
        (like 'return high' or 'return perms')."""
        seeds = BUILTIN_SEEDS["CODE-COMPLETION"]
        for i, seed in enumerate(seeds):
            lines = seed["question"].strip().split("\n")
            last_line = lines[-1].strip()
            # These would indicate a complete function:
            complete_patterns = [
                "return high", "return perms", "return result",
                "return -1", "return arr", "return dummy",
            ]
            for pattern in complete_patterns:
                assert pattern not in last_line, (
                    f"Builtin seed [{i}] has complete return: '{last_line}'"
                )


# ======================================================================
# TEST 2: Canonical CODE-GEN-T1 gets "from scratch" seeds, not completions
# ======================================================================

class TestCanonicalCodeGenT1Seeds:
    """CODE-GEN-T1 (canonical) should get "write a function" seeds, not
    code completion partials."""

    def test_code_gen_t1_seeds_are_from_scratch(self):
        """CODE-GEN-T1 builtin seeds should be 'Write a function...' style."""
        seeds = get_builtin_seeds("CODE-GEN-T1", 100)
        for seed in seeds:
            q = seed["question"]
            # Should be generation tasks, not completions
            assert "Write" in q or "Implement" in q or "Create" in q, (
                f"CODE-GEN-T1 seed doesn't look like a generation task: '{q[:80]}'"
            )

    def test_code_gen_t1_seeds_not_completions(self):
        """CODE-GEN-T1 seeds should NOT contain partial function definitions."""
        seeds = get_builtin_seeds("CODE-GEN-T1", 100)
        for seed in seeds:
            q = seed["question"]
            assert "Complete this function" not in q, (
                f"CODE-GEN-T1 should not have completion seeds: '{q[:80]}'"
            )
            # Should not contain 'def foo():' partial code
            assert q.count("def ") == 0, (
                f"CODE-GEN-T1 seed contains function definition: '{q[:80]}'"
            )


# ======================================================================
# TEST 3: Alias normalization — CODE-COMPLETION → CODE-GEN-T1
# ======================================================================

class TestCodeCompletionAlias:
    """CODE-COMPLETION should resolve to CODE-GEN-T1 through alias system."""

    def test_alias_resolves_correctly(self):
        """get_skill_bucket('CODE-COMPLETION') should return CODE-GEN-T1."""
        skill = get_skill_bucket("CODE-COMPLETION")
        assert skill.id == "CODE-GEN-T1"

    def test_generate_bank_would_use_code_gen_t1_seeds(self):
        """When generate-bank normalizes CODE-COMPLETION to CODE-GEN-T1,
        it should get the 5 'write a function' seeds, not the 8 completion seeds."""
        skill = get_skill_bucket("CODE-COMPLETION")
        seeds = get_builtin_seeds(skill.id, 100)
        assert len(seeds) == 5, f"Expected 5 CODE-GEN-T1 seeds, got {len(seeds)}"
        # Verify they're generation tasks
        assert all("Write" in s["question"] for s in seeds)

    def test_legacy_key_still_returns_completion_seeds(self):
        """Direct 'CODE-COMPLETION' lookup should still work for backward compat."""
        seeds = get_builtin_seeds("CODE-COMPLETION", 100)
        assert len(seeds) == 8
        assert all("Complete this function" in s["question"] for s in seeds)


# ======================================================================
# TEST 4: LLM seed prompt — CODE-GEN-T1 uses generation prompt, not completion
# ======================================================================

class TestLlmSeedPromptRouting:
    """The LLM seed generation path should use CODE-GEN-T1's prompt,
    not CODE-COMPLETION's prompt."""

    def test_canonical_resolves_to_gen_prompt(self):
        """_resolve_prompt_key('CODE-GEN-T1') should return CODE-GEN-T1 (direct)."""
        key = _gen._resolve_prompt_key("CODE-GEN-T1")
        assert key == "CODE-GEN-T1"

    def test_gen_prompt_asks_for_generation(self):
        """CODE-GEN-T1 prompt should ask for code generation, not completion."""
        prompt = SEED_PROMPTS["CODE-GEN-T1"]
        assert "generation" in prompt.lower() or "write" in prompt.lower()
        assert "partial" not in prompt.lower()
        assert "Complete this function" not in prompt

    def test_completion_prompt_exists_but_orphaned(self):
        """CODE-COMPLETION prompt still exists in SEED_PROMPTS (harmless dead code)."""
        assert "CODE-COMPLETION" in SEED_PROMPTS
        prompt = SEED_PROMPTS["CODE-COMPLETION"]
        assert "Partial function definitions" in prompt

    def test_completion_prompt_not_reachable_from_canonical(self):
        """Resolving CODE-GEN-T1 should NOT reach the CODE-COMPLETION prompt."""
        key = _gen._resolve_prompt_key("CODE-GEN-T1")
        assert key != "CODE-COMPLETION"


# ======================================================================
# TEST 5: Existing bank data needs regeneration
# ======================================================================

class TestExistingBankData:
    """The existing CODE-COMPLETION.jsonl has broken data from old runs.
    Verify it contains complete functions (confirming it needs regeneration)."""

    def test_existing_jsonl_has_complete_functions(self):
        """The old CODE-COMPLETION.jsonl should have complete functions
        (this proves the bug existed and data needs refresh)."""
        import json
        jsonl_path = EXPERIMENT_DIR / "synth_data_bank" / "CODE-COMPLETION.jsonl"
        if not jsonl_path.exists():
            return  # Skip if file doesn't exist (clean environment)

        with open(jsonl_path) as f:
            samples = [json.loads(l) for l in f if l.strip()]

        complete_count = 0
        for s in samples:
            lines = s["question"].strip().split("\n")
            last_line = lines[-1].strip()
            # A complete function has a return with a value as the last line
            if last_line.startswith("return ") and len(last_line) > 8:
                complete_count += 1

        # All 5 samples should have complete functions (the bug)
        assert complete_count >= 3, (
            f"Expected most samples to have complete functions (old bug), "
            f"got {complete_count}/{len(samples)}"
        )


# ======================================================================
# Run with: uv run python -m pytest tests/4_synthetic_data_and_self_distillation/bugfixes/test_bug5.py -v
# ======================================================================

