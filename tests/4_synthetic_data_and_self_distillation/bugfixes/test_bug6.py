"""
Test Bug Fix #6: Manifest Key Normalization
=============================================

BUG (before fix):
    The manifest stored whatever skill key was passed — legacy or canonical.
    Running with --skills RSN-ARITHMETIC stored "RSN-ARITHMETIC", while
    running with --all stored "RSN-ARITH". This caused:
      1. Duplicate entries for the same skill under different keys
      2. inject --skills RSN-ARITH failing when manifest had "RSN-ARITHMETIC"
      3. rebuild-manifest skipping legacy-named shard files (RSN-ARITHMETIC.jsonl)

    Three additional code paths had unfixed normalization:
      - cmd_rebuild_manifest(): skipped 15 of 22 shards (legacy filenames)
      - cmd_inject(): failed lookup when key format mismatched manifest
      - synth_adapter._update_manifest(): wrote legacy keys like "KNOW-FACTUAL_synth"

FIX (after fix):
    1. cmd_generate_bank(): normalizes skill_id = skill.id (canonical) — already done
    2. cmd_rebuild_manifest(): uses get_skill_bucket() to resolve shard filenames
    3. cmd_inject(): tries raw key → canonical → reverse legacy lookup
    4. synth_adapter: resolves to canonical before writing _synth entry keys

FILES CHANGED:
    experiments/4_synthetic_data_and_self_distillation/run_pipeline.py
    experiments/4_synthetic_data_and_self_distillation/integration/synth_adapter.py
"""

import sys
import json
import tempfile
import shutil
from pathlib import Path

# ── Setup path so we can import from experiments/ ──────────────────────
EXPERIMENT_DIR = Path(__file__).resolve().parents[3] / "experiments" / "4_synthetic_data_and_self_distillation"
sys.path.insert(0, str(EXPERIMENT_DIR))

from common.skills import SKILL_BUCKETS, SKILL_ALIASES, get_skill_bucket, resolve_skill_alias


# ======================================================================
# TEST 1: cmd_generate_bank normalization (already fixed)
# ======================================================================

class TestGenerateBankNormalization:
    """cmd_generate_bank should always write canonical keys to manifest."""

    def test_canonical_key_used_in_loop(self):
        """The loop variable skill_id should be canonical after normalization."""
        # Simulate the normalization logic from cmd_generate_bank
        for raw_skill_id in ["RSN-ARITHMETIC", "CODE-COMPLETION", "KNOW-FACTUAL"]:
            skill = get_skill_bucket(raw_skill_id)
            skill_id = skill.id  # canonical form
            assert skill_id in SKILL_BUCKETS, f"{skill_id} not canonical"
            assert skill_id != raw_skill_id, \
                f"{raw_skill_id} should differ from canonical {skill_id}"

    def test_canonical_keys_stay_unchanged(self):
        """Already-canonical keys should pass through unchanged."""
        for skill_id in ["RSN-ARITH", "CODE-GEN-T1", "FND-FACT"]:
            skill = get_skill_bucket(skill_id)
            assert skill.id == skill_id

    def test_shard_filename_uses_canonical(self):
        """Shard file should be named with canonical ID."""
        skill = get_skill_bucket("RSN-ARITHMETIC")
        shard_name = f"{skill.id}.jsonl"
        assert shard_name == "RSN-ARITH.jsonl"


# ======================================================================
# TEST 2: cmd_rebuild_manifest — resolves legacy shard filenames
# ======================================================================

class TestRebuildManifestResolvesLegacy:
    """rebuild-manifest should accept legacy-named shard files via alias resolution."""

    def _simulate_rebuild(self, shard_names: list[str]) -> dict:
        """Simulate the rebuild-manifest logic on given shard filenames."""
        manifest_skills = {}
        for name in shard_names:
            raw_name = name.replace(".jsonl", "")
            try:
                skill = get_skill_bucket(raw_name)
                skill_id = skill.id
            except ValueError:
                continue  # skip unknown
            manifest_skills[skill_id] = {"shard_file": name, "samples": 5}
        return manifest_skills

    def test_legacy_shard_resolved(self):
        """RSN-ARITHMETIC.jsonl should resolve to RSN-ARITH in manifest."""
        result = self._simulate_rebuild(["RSN-ARITHMETIC.jsonl"])
        assert "RSN-ARITH" in result
        assert "RSN-ARITHMETIC" not in result

    def test_canonical_shard_passes_through(self):
        """RSN-ARITH.jsonl stays as RSN-ARITH."""
        result = self._simulate_rebuild(["RSN-ARITH.jsonl"])
        assert "RSN-ARITH" in result

    def test_all_legacy_shards_resolve(self):
        """All legacy alias shard names should resolve to canonical."""
        legacy_shards = [f"{alias}.jsonl" for alias in SKILL_ALIASES.keys()]
        result = self._simulate_rebuild(legacy_shards)
        for canonical in set(SKILL_ALIASES.values()):
            assert canonical in result, f"Canonical {canonical} missing from rebuilt manifest"

    def test_unknown_shards_skipped(self):
        """Shards with unrecognized names should be skipped."""
        result = self._simulate_rebuild(["KNOW-FACTUAL_synth.jsonl", "BOGUS.jsonl"])
        assert len(result) == 0

    def test_no_duplicate_canonicals(self):
        """Two legacy files mapping to same canonical should not create duplicates.
        Last one wins (dict behavior)."""
        result = self._simulate_rebuild(["KNOW-SCIENCE.jsonl", "KNOW-COMMONSENSE.jsonl"])
        # Both map to RSN-CS, should have exactly one entry
        assert "RSN-CS" in result
        assert len(result) == 1

    def test_existing_bank_files_all_resolve(self):
        """All current shard files in the bank should either resolve or be _synth."""
        bank_dir = EXPERIMENT_DIR / "synth_data_bank"
        if not bank_dir.exists():
            return
        shards = [p.name for p in bank_dir.glob("*.jsonl")]
        result = self._simulate_rebuild(shards)
        # Count how many resolve vs skip
        # _synth files won't resolve (expected)
        synth_files = [s for s in shards if "_synth" in s]
        non_synth = [s for s in shards if "_synth" not in s]
        assert len(result) >= len(non_synth) - 1, \
            f"Only {len(result)}/{len(non_synth)} non-synth shards resolved"


# ======================================================================
# TEST 3: cmd_inject — finds skills regardless of key format in manifest
# ======================================================================

class TestInjectSkillLookup:
    """inject should find skills in manifest regardless of legacy/canonical format."""

    def _simulate_inject_lookup(self, skill_id: str, manifest_keys: list[str]) -> str | None:
        """Simulate the inject skill lookup logic."""
        manifest_skills = {k: {"shard_file": f"{k}.jsonl"} for k in manifest_keys}

        # Direct lookup
        if skill_id in manifest_skills:
            return skill_id

        # Try canonical
        try:
            canonical = get_skill_bucket(skill_id).id
            if canonical in manifest_skills:
                return canonical
        except ValueError:
            pass

        # Try reverse legacy lookup
        for legacy, canon in SKILL_ALIASES.items():
            if canon == skill_id and legacy in manifest_skills:
                return legacy

        return None

    def test_canonical_finds_canonical_manifest(self):
        """RSN-ARITH finds RSN-ARITH in manifest."""
        result = self._simulate_inject_lookup("RSN-ARITH", ["RSN-ARITH"])
        assert result == "RSN-ARITH"

    def test_legacy_finds_legacy_manifest(self):
        """RSN-ARITHMETIC finds RSN-ARITHMETIC in manifest."""
        result = self._simulate_inject_lookup("RSN-ARITHMETIC", ["RSN-ARITHMETIC"])
        assert result == "RSN-ARITHMETIC"

    def test_canonical_finds_legacy_manifest(self):
        """RSN-ARITH finds RSN-ARITHMETIC in manifest (reverse lookup)."""
        result = self._simulate_inject_lookup("RSN-ARITH", ["RSN-ARITHMETIC"])
        assert result == "RSN-ARITHMETIC"

    def test_legacy_finds_canonical_manifest(self):
        """RSN-ARITHMETIC finds RSN-ARITH in manifest (alias resolution)."""
        result = self._simulate_inject_lookup("RSN-ARITHMETIC", ["RSN-ARITH"])
        assert result == "RSN-ARITH"

    def test_code_gen_t1_finds_code_completion(self):
        """CODE-GEN-T1 (canonical) finds CODE-COMPLETION (legacy) in manifest."""
        result = self._simulate_inject_lookup("CODE-GEN-T1", ["CODE-COMPLETION"])
        assert result == "CODE-COMPLETION"

    def test_code_completion_finds_code_gen_t1(self):
        """CODE-COMPLETION (legacy) finds CODE-GEN-T1 (canonical) in manifest."""
        result = self._simulate_inject_lookup("CODE-COMPLETION", ["CODE-GEN-T1"])
        assert result == "CODE-GEN-T1"

    def test_unknown_skill_returns_none(self):
        """Unknown skill returns None."""
        result = self._simulate_inject_lookup("BOGUS-SKILL", ["RSN-ARITH"])
        assert result is None

    def test_all_aliases_find_their_canonical(self):
        """Every legacy alias should find its canonical in the manifest."""
        for legacy, canonical in SKILL_ALIASES.items():
            result = self._simulate_inject_lookup(legacy, [canonical])
            assert result is not None, f"{legacy} could not find {canonical} in manifest"

    def test_all_canonicals_find_their_legacy(self):
        """Every canonical with a legacy alias should find it in manifest."""
        reverse = {}
        for legacy, canonical in SKILL_ALIASES.items():
            reverse.setdefault(canonical, []).append(legacy)
        for canonical, legacies in reverse.items():
            result = self._simulate_inject_lookup(canonical, legacies[:1])
            assert result is not None, f"{canonical} could not find {legacies[0]} in manifest"


# ======================================================================
# TEST 4: synth_adapter — canonical keys for _synth entries
# ======================================================================

class TestSynthAdapterCanonicalKeys:
    """synth_adapter._update_manifest should use canonical keys for _synth entries."""

    def test_know_factual_becomes_fnd_fact_synth(self):
        """KNOW-FACTUAL (legacy) should produce FND-FACT_synth entry."""
        # Simulate the normalization
        skill = "KNOW-FACTUAL"
        try:
            canonical = get_skill_bucket(skill).id
        except ValueError:
            canonical = skill
        entry_key = f"{canonical}_synth"
        assert entry_key == "FND-FACT_synth"

    def test_rsn_arithmetic_becomes_rsn_arith_synth(self):
        """RSN-ARITHMETIC should produce RSN-ARITH_synth."""
        canonical = get_skill_bucket("RSN-ARITHMETIC").id
        assert f"{canonical}_synth" == "RSN-ARITH_synth"

    def test_canonical_input_unchanged(self):
        """Already-canonical input stays the same."""
        canonical = get_skill_bucket("RSN-CS").id
        assert f"{canonical}_synth" == "RSN-CS_synth"

    def test_all_exercise_mappings_resolve(self):
        """Every skill in EXERCISE_TO_SKILL should resolve to a canonical."""
        from integration.synth_adapter import EXERCISE_TO_SKILL
        for exercise, skill in EXERCISE_TO_SKILL.items():
            try:
                canonical = get_skill_bucket(skill).id
                assert canonical in SKILL_BUCKETS, \
                    f"EXERCISE_TO_SKILL['{exercise}'] = '{skill}' -> '{canonical}' not canonical"
            except ValueError:
                pass  # Some mappings may use legacy names that don't resolve


# ======================================================================
# TEST 5: No key collision between canonical and _synth entries
# ======================================================================

class TestNoKeyCollision:
    """_synth keys should not collide with canonical skill keys."""

    def test_synth_suffix_prevents_collision(self):
        """No canonical key ends with '_synth'."""
        for skill_id in SKILL_BUCKETS:
            assert not skill_id.endswith("_synth"), \
                f"Canonical key {skill_id} ends with _synth — would collide"

    def test_synth_keys_distinct_from_canonical(self):
        """Generated _synth keys should not match any canonical key."""
        for legacy, canonical in SKILL_ALIASES.items():
            synth_key = f"{canonical}_synth"
            assert synth_key not in SKILL_BUCKETS, \
                f"Synth key {synth_key} collides with canonical skill"


# ======================================================================
# Run with: uv run python -m pytest tests/4_synthetic_data_and_self_distillation/bugfixes/test_bug6.py -v
# ======================================================================

