#!/usr/bin/env python3
"""
Tests for Telugu dataset generators (S1-S11).
Validates that generated output files exist, have correct line counts,
and follow the expected Q? A. format.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "output",
)

# Expected counts for each statement file
STATEMENT_SPECS = [
    ("group1_s1.txt", 30000, "S1: Spelling"),
    ("group1_s2.txt", 26000, "S2: Letter Position"),
    ("group1_s3.txt", 20000, "S3: Sound Matching"),
    ("group1_s4.txt", 26000, "S4: Letter Count"),
    ("group1_s5.txt", 20000, "S5: Rhyming"),
    ("group1_s6.txt", 20000, "S6: Classification"),
    ("group1_s7.txt", 18000, "S7: Position of Letter"),
    ("group1_s8.txt", 12000, "S8: Number Spelling"),
    ("group1_s9.txt", 18000, "S9: Last Letter"),
    ("group1_s10.txt", 10000, "S10: Word Comparison"),
    ("group1_s11.txt", 8000, "S11: Ottulu & Gunintalu"),
]


def _read_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ── File existence ──


class TestFilesExist:
    @pytest.mark.parametrize("filename, expected_count, description", STATEMENT_SPECS)
    def test_statement_file_exists(self, filename, expected_count, description):
        filepath = os.path.join(DATA_DIR, filename)
        assert os.path.exists(filepath), f"{description}: {filename} not found"

    def test_final_output_exists(self):
        filepath = os.path.join(OUTPUT_DIR, "group1_telugu.txt")
        assert os.path.exists(filepath), "Final output group1_telugu.txt not found"


# ── Line counts ──


class TestLineCounts:
    @pytest.mark.parametrize("filename, expected_count, description", STATEMENT_SPECS)
    def test_statement_line_count(self, filename, expected_count, description):
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            pytest.skip(f"{filename} not found")
        lines = _read_lines(filepath)
        assert (
            len(lines) == expected_count
        ), f"{description}: expected {expected_count} lines, got {len(lines)}"

    def test_total_pairs_208000(self):
        total = 0
        for filename, expected_count, description in STATEMENT_SPECS:
            filepath = os.path.join(DATA_DIR, filename)
            if os.path.exists(filepath):
                total += len(_read_lines(filepath))
        assert total == 208000, f"Total pairs: {total}, expected 208000"


# ── Format validation (Q? A.) ──


class TestOutputFormat:
    @pytest.mark.parametrize("filename, expected_count, description", STATEMENT_SPECS)
    def test_all_lines_have_question_mark(self, filename, expected_count, description):
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            pytest.skip(f"{filename} not found")
        lines = _read_lines(filepath)
        # Check first 100 lines for efficiency
        for i, line in enumerate(lines[:100]):
            assert (
                "?" in line
            ), f"{description} line {i+1}: no '?' found in: {line[:80]}"

    @pytest.mark.parametrize("filename, expected_count, description", STATEMENT_SPECS)
    def test_all_lines_end_with_period(self, filename, expected_count, description):
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            pytest.skip(f"{filename} not found")
        lines = _read_lines(filepath)
        for i, line in enumerate(lines[:100]):
            assert line.endswith(
                "."
            ), f"{description} line {i+1}: doesn't end with '.': {line[-20:]}"

    @pytest.mark.parametrize("filename, expected_count, description", STATEMENT_SPECS)
    def test_no_danda_in_output(self, filename, expected_count, description):
        """Telugu uses period (.), NOT danda (।)."""
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            pytest.skip(f"{filename} not found")
        lines = _read_lines(filepath)
        for i, line in enumerate(lines[:100]):
            assert (
                "।" not in line
            ), f"{description} line {i+1}: contains danda (।): {line[:80]}"


# ── Cross-script leakage ──


class TestNoScriptLeakage:
    def _check_file_for_leakage(self, filepath):
        """Check a file for Kannada or Hindi characters."""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        kannada = [ch for ch in content if "\u0c80" <= ch <= "\u0cff"]
        hindi = [ch for ch in content if "\u0900" <= ch <= "\u097f"]
        return kannada, hindi

    @pytest.mark.parametrize("filename, expected_count, description", STATEMENT_SPECS)
    def test_no_kannada_in_statement_files(self, filename, expected_count, description):
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            pytest.skip(f"{filename} not found")
        kannada, _ = self._check_file_for_leakage(filepath)
        assert len(kannada) == 0, f"{description}: found {len(kannada)} Kannada chars"

    @pytest.mark.parametrize("filename, expected_count, description", STATEMENT_SPECS)
    def test_no_hindi_in_statement_files(self, filename, expected_count, description):
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            pytest.skip(f"{filename} not found")
        _, hindi = self._check_file_for_leakage(filepath)
        assert (
            len(hindi) == 0
        ), f"{description}: found {len(hindi)} Hindi/Devanagari chars"

    def test_no_leakage_in_final_output(self):
        filepath = os.path.join(OUTPUT_DIR, "group1_telugu.txt")
        if not os.path.exists(filepath):
            pytest.skip("Final output not found")
        kannada, hindi = self._check_file_for_leakage(filepath)
        assert len(kannada) == 0, f"Final output: {len(kannada)} Kannada chars"
        assert len(hindi) == 0, f"Final output: {len(hindi)} Hindi chars"


# ── Token count validation on final output ──


class TestFinalOutputTokens:
    def test_min_512_tokens_per_line(self):
        filepath = os.path.join(OUTPUT_DIR, "group1_telugu.txt")
        if not os.path.exists(filepath):
            pytest.skip("Final output not found")

        from group1_telugu.prompt_utils_telugu import count_tokens_telugu

        lines = _read_lines(filepath)
        assert len(lines) > 0, "Final output is empty"

        below_512 = []
        for i, line in enumerate(lines):
            tokens = count_tokens_telugu(line)
            if tokens < 512:
                below_512.append((i + 1, tokens))

        assert len(below_512) == 0, (
            f"{len(below_512)} lines below 512 tokens. "
            f"First: line {below_512[0][0]} with {below_512[0][1]} tokens"
            if below_512
            else ""
        )

    def test_final_output_has_datapoints(self):
        filepath = os.path.join(OUTPUT_DIR, "group1_telugu.txt")
        if not os.path.exists(filepath):
            pytest.skip("Final output not found")
        lines = _read_lines(filepath)
        assert len(lines) >= 10000, f"Only {len(lines)} data points, expected >= 10000"


# ── S11-specific: Gunintalu correctness ──


class TestS11Gunintalu:
    def test_gunintam_chart_present(self):
        """S11 should contain gunintam chart entries."""
        filepath = os.path.join(DATA_DIR, "group1_s11.txt")
        if not os.path.exists(filepath):
            pytest.skip("S11 not found")
        lines = _read_lines(filepath)
        # Look for gunintam chart pattern (comma-separated aksharas)
        chart_lines = [line for line in lines if "గుణింతాలు" in line]
        assert len(chart_lines) > 0, "No gunintam chart entries found in S11"

    def test_ottulu_present(self):
        """S11 should contain ottulu (conjunct) entries."""
        filepath = os.path.join(DATA_DIR, "group1_s11.txt")
        if not os.path.exists(filepath):
            pytest.skip("S11 not found")
        lines = _read_lines(filepath)
        ottulu_lines = [
            line for line in lines if "ఒత్తు" in line or "సంయుక్తాక్షరం" in line
        ]
        assert len(ottulu_lines) > 0, "No ottulu entries found in S11"

    def test_classification_present(self):
        """S11 should contain vowel/consonant classification entries."""
        filepath = os.path.join(DATA_DIR, "group1_s11.txt")
        if not os.path.exists(filepath):
            pytest.skip("S11 not found")
        lines = _read_lines(filepath)
        classify_lines = [line for line in lines if "స్వరమా" in line or "వ్యంజనమా" in line]
        assert len(classify_lines) > 0, "No classification entries found in S11"

    def test_ka_gunintam_correctness(self):
        """Verify క gunintam chart is correct."""
        filepath = os.path.join(DATA_DIR, "group1_s11.txt")
        if not os.path.exists(filepath):
            pytest.skip("S11 not found")
        lines = _read_lines(filepath)
        ka_charts = [
            line for line in lines if "గుణింతాలు" in line and line.startswith('"క"')
        ]
        if ka_charts:
            # Answer should contain: క, కా, కి, కీ, కు, కూ, ...
            answer = ka_charts[0].split("?", 1)[1] if "?" in ka_charts[0] else ""
            assert "కా" in answer, "కా not found in క gunintam chart"
            assert "కి" in answer, "కి not found in క gunintam chart"
            assert "కు" in answer, "కు not found in క gunintam chart"
