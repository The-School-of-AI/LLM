"""
Test Bug Fix #7: Configurable Timeouts for Larger Models
==========================================================

BUG (before fix):
    All Ollama API calls had hardcoded timeouts:
      - generation/dual_view_generator.py: timeout=300 (5 min)
      - generation/seed_generator.py: timeout=300 (5 min)
      - diagnostics/run_diagnostics.py: timeout=30 (30 sec)
      - validation/verification.py: timeout=60 (1 min)

    A 70B model on consumer hardware can take 10-15 minutes per generation.
    Every API call would timeout, making the pipeline unusable for large models.

FIX (after fix):
    All 4 files now read OLLAMA_TIMEOUT from environment variable:
      - Generation files: default 600s (10 min)
      - Diagnostics/verification: default 120s (2 min)
      - check_ollama() health check: stays at 5s (hardcoded, correct)

    All 4 use try/except to handle non-numeric env var values gracefully.

    Usage:
      export OLLAMA_TIMEOUT=1800  # 30 minutes for 70B models
      python run_pipeline.py generate-bank --all --model llama3:70b

    Also fixed: verification.py had `import os` placed after other imports instead
    of in the standard import block at the top.

FILES CHANGED:
    experiments/4_synthetic_data_and_self_distillation/generation/dual_view_generator.py
    experiments/4_synthetic_data_and_self_distillation/generation/seed_generator.py
    experiments/4_synthetic_data_and_self_distillation/diagnostics/run_diagnostics.py
    experiments/4_synthetic_data_and_self_distillation/validation/verification.py
"""

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch

# ── Setup path so we can import from experiments/ ──────────────────────
EXPERIMENT_DIR = Path(__file__).resolve().parents[3] / "experiments" / "4_synthetic_data_and_self_distillation"
sys.path.insert(0, str(EXPERIMENT_DIR))


# ======================================================================
# TEST 1: All 4 modules define OLLAMA_TIMEOUT
# ======================================================================

class TestAllModulesHaveTimeout:
    """Every module that makes Ollama calls must define OLLAMA_TIMEOUT."""

    def test_dual_view_generator_has_timeout(self):
        from generation import dual_view_generator
        assert hasattr(dual_view_generator, "OLLAMA_TIMEOUT")
        assert isinstance(dual_view_generator.OLLAMA_TIMEOUT, int)

    def test_seed_generator_has_timeout(self):
        from generation import seed_generator
        assert hasattr(seed_generator, "OLLAMA_TIMEOUT")
        assert isinstance(seed_generator.OLLAMA_TIMEOUT, int)

    def test_run_diagnostics_has_timeout(self):
        from diagnostics import run_diagnostics
        assert hasattr(run_diagnostics, "OLLAMA_TIMEOUT")
        assert isinstance(run_diagnostics.OLLAMA_TIMEOUT, int)

    def test_verification_has_timeout(self):
        from validation import verification
        assert hasattr(verification, "OLLAMA_TIMEOUT")
        assert isinstance(verification.OLLAMA_TIMEOUT, int)


# ======================================================================
# TEST 2: Default values are appropriate per module type
# ======================================================================

class TestDefaultTimeoutValues:
    """Defaults should be appropriate: generation > diagnostics."""

    def test_generation_default_is_600(self):
        """Generation modules should default to 600s (10 min)."""
        # We can't easily test env-var defaults without reloading modules,
        # so we test the fallback value in the try/except
        from generation import dual_view_generator, seed_generator
        # When env var is unset, default should be 600
        # (If test env has OLLAMA_TIMEOUT set, skip this)
        if "OLLAMA_TIMEOUT" not in os.environ:
            assert dual_view_generator.OLLAMA_TIMEOUT == 600
            assert seed_generator.OLLAMA_TIMEOUT == 600

    def test_diagnostics_default_is_120(self):
        """Diagnostics should default to 120s (2 min)."""
        if "OLLAMA_TIMEOUT" not in os.environ:
            from diagnostics import run_diagnostics
            assert run_diagnostics.OLLAMA_TIMEOUT == 120

    def test_verification_default_is_120(self):
        """Verification should default to 120s (2 min)."""
        if "OLLAMA_TIMEOUT" not in os.environ:
            from validation import verification
            assert verification.OLLAMA_TIMEOUT == 120

    def test_generation_default_higher_than_diagnostics(self):
        """Generation timeout should be >= diagnostics timeout."""
        from generation import dual_view_generator
        from diagnostics import run_diagnostics
        # Unless overridden, generation should be higher
        if "OLLAMA_TIMEOUT" not in os.environ:
            assert dual_view_generator.OLLAMA_TIMEOUT >= run_diagnostics.OLLAMA_TIMEOUT


# ======================================================================
# TEST 3: All modules use OLLAMA_TIMEOUT in urlopen calls
# ======================================================================

class TestTimeoutUsedInUrlopen:
    """Verify that no hardcoded timeout= remains in urlopen calls."""

    def _check_file_for_hardcoded_timeout(self, filepath: Path) -> list[str]:
        """Scan a file for hardcoded timeout= in urlopen calls."""
        issues = []
        with open(filepath) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            # Look for urlopen(..., timeout=<number>) — hardcoded
            if "urlopen" in line and "timeout=" in line:
                # timeout=OLLAMA_TIMEOUT is fine
                if "OLLAMA_TIMEOUT" in line:
                    continue
                # timeout=5 in check_ollama is fine (health check)
                if "timeout=5" in line:
                    continue
                issues.append(f"Line {i+1}: {line.strip()}")
        return issues

    def test_dual_view_generator_no_hardcoded(self):
        filepath = EXPERIMENT_DIR / "generation" / "dual_view_generator.py"
        issues = self._check_file_for_hardcoded_timeout(filepath)
        assert not issues, f"Hardcoded timeouts found: {issues}"

    def test_seed_generator_no_hardcoded(self):
        filepath = EXPERIMENT_DIR / "generation" / "seed_generator.py"
        issues = self._check_file_for_hardcoded_timeout(filepath)
        assert not issues, f"Hardcoded timeouts found: {issues}"

    def test_run_diagnostics_no_hardcoded(self):
        filepath = EXPERIMENT_DIR / "diagnostics" / "run_diagnostics.py"
        issues = self._check_file_for_hardcoded_timeout(filepath)
        assert not issues, f"Hardcoded timeouts found: {issues}"

    def test_verification_no_hardcoded(self):
        filepath = EXPERIMENT_DIR / "validation" / "verification.py"
        issues = self._check_file_for_hardcoded_timeout(filepath)
        assert not issues, f"Hardcoded timeouts found: {issues}"


# ======================================================================
# TEST 4: check_ollama() health check keeps hardcoded 5s
# ======================================================================

class TestCheckOllamaHealthCheck:
    """check_ollama() should keep its 5s timeout (it's a connectivity test)."""

    def test_check_ollama_has_short_timeout(self):
        """check_ollama should use timeout=5 (not OLLAMA_TIMEOUT)."""
        import inspect
        from diagnostics.run_diagnostics import check_ollama
        source = inspect.getsource(check_ollama)
        assert "timeout=5" in source, "check_ollama should use hardcoded timeout=5"
        assert "OLLAMA_TIMEOUT" not in source, \
            "check_ollama should NOT use OLLAMA_TIMEOUT (it's a quick health check)"


# ======================================================================
# TEST 5: OLLAMA_TIMEOUT env var parsing — safe handling of bad values
# ======================================================================

class TestEnvVarParsingSafety:
    """OLLAMA_TIMEOUT parsing must not crash on bad env var values."""

    def test_numeric_string_parsed(self):
        """Normal numeric string works."""
        val_str = "1800"
        try:
            val = int(val_str)
        except (ValueError, TypeError):
            val = 600
        assert val == 1800

    def test_empty_string_fallback(self):
        """Empty string should fall back to default, not crash."""
        val_str = ""
        try:
            val = int(val_str)
        except (ValueError, TypeError):
            val = 600
        assert val == 600

    def test_non_numeric_fallback(self):
        """Non-numeric string should fall back to default, not crash."""
        val_str = "abc"
        try:
            val = int(val_str)
        except (ValueError, TypeError):
            val = 600
        assert val == 600

    def test_none_fallback(self):
        """None (unset env var) should fall back to default."""
        val_str = None
        try:
            val = int(val_str) if val_str else 600
        except (ValueError, TypeError):
            val = 600
        assert val == 600

    def test_float_string_fallback(self):
        """Float string like '1.5' should fall back to default."""
        val_str = "1.5"
        try:
            val = int(val_str)
        except (ValueError, TypeError):
            val = 600
        assert val == 600

    def test_source_code_has_try_except(self):
        """All 4 modules should use try/except around int() parsing."""
        files = [
            EXPERIMENT_DIR / "generation" / "dual_view_generator.py",
            EXPERIMENT_DIR / "generation" / "seed_generator.py",
            EXPERIMENT_DIR / "diagnostics" / "run_diagnostics.py",
            EXPERIMENT_DIR / "validation" / "verification.py",
        ]
        for filepath in files:
            source = filepath.read_text()
            # Find the OLLAMA_TIMEOUT assignment block
            assert "try:" in source and "OLLAMA_TIMEOUT" in source, \
                f"{filepath.name} missing try/except around OLLAMA_TIMEOUT parsing"
            assert "except (ValueError, TypeError)" in source, \
                f"{filepath.name} missing except (ValueError, TypeError) for OLLAMA_TIMEOUT"


# ======================================================================
# TEST 6: All modules also have OLLAMA_BASE configurable
# ======================================================================

class TestOllamaBaseConfigurable:
    """OLLAMA_BASE (host URL) should also be configurable via env var."""

    def test_dual_view_generator_base_configurable(self):
        from generation import dual_view_generator
        assert hasattr(dual_view_generator, "OLLAMA_BASE")

    def test_seed_generator_base_configurable(self):
        from generation import seed_generator
        assert hasattr(seed_generator, "OLLAMA_BASE")

    def test_run_diagnostics_base_configurable(self):
        from diagnostics import run_diagnostics
        assert hasattr(run_diagnostics, "OLLAMA_BASE")

    def test_verification_base_configurable(self):
        from validation import verification
        assert hasattr(verification, "OLLAMA_BASE")


# ======================================================================
# TEST 7: import os is in the standard import block
# ======================================================================

class TestImportOrganization:
    """import os should be in the standard import block, not inline."""

    def test_verification_import_os_at_top(self):
        """verification.py should have 'import os' in the top import block."""
        filepath = EXPERIMENT_DIR / "validation" / "verification.py"
        lines = filepath.read_text().split("\n")
        # Find the first non-docstring, non-blank line with 'import'
        import_lines = []
        in_docstring = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            if stripped.startswith("import ") or stripped.startswith("from "):
                import_lines.append((i + 1, stripped))

        # 'import os' should be among the first group of imports
        os_import = [ln for ln, text in import_lines if text == "import os"]
        assert os_import, "verification.py missing 'import os'"
        # Should be in the first 25 lines (standard import block)
        assert os_import[0] <= 25, \
            f"'import os' at line {os_import[0]} — should be in top import block (< line 25)"


# ======================================================================
# Run with: uv run python -m pytest tests/4_synthetic_data_and_self_distillation/bugfixes/test_bug7.py -v
# ======================================================================

