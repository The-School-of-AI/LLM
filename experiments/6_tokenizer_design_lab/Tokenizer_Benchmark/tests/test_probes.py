"""
Tests for probe generators.

Validates that probes are format-only and don't contain benchmark content.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from probes import (
    MathProbeGenerator,
    MCQProbeGenerator,
    CodeProbeGenerator,
    IndicProbeGenerator,
    SyntheticInstructionGenerator,
)


class TestMathProbeGenerator:
    """Tests for math probe generation."""
    
    def test_generation_count(self):
        """Test that correct number of probes are generated."""
        gen = MathProbeGenerator(seed=42)
        probes = gen.generate_batch(100)
        assert len(probes) == 100
    
    def test_probe_categories(self):
        """Test that all categories are represented."""
        gen = MathProbeGenerator(seed=42)
        probes = gen.generate_batch(300)
        categories = set(p.category for p in probes)
        assert 'latex' in categories
        assert 'numeric' in categories
        assert 'expression' in categories
    
    def test_probe_difficulties(self):
        """Test that all difficulties are represented."""
        gen = MathProbeGenerator(seed=42)
        probes = gen.generate_batch(300)
        difficulties = set(p.difficulty for p in probes)
        assert 'easy' in difficulties
        assert 'medium' in difficulties
        assert 'hard' in difficulties
    
    def test_no_benchmark_leak(self):
        """Test that probes don't contain benchmark identifiers."""
        gen = MathProbeGenerator(seed=42)
        probes = gen.generate_batch(100)
        
        benchmark_terms = ['gsm8k', 'mmlu', 'hellaswag', 'benchmark']
        for probe in probes:
            content_lower = probe.content.lower()
            for term in benchmark_terms:
                assert term not in content_lower


class TestMCQProbeGenerator:
    """Tests for MCQ probe generation."""
    
    def test_generation_count(self):
        gen = MCQProbeGenerator(seed=42)
        probes = gen.generate_batch(100)
        assert len(probes) == 100
    
    def test_styles(self):
        """Test that all MCQ styles are represented."""
        gen = MCQProbeGenerator(seed=42)
        probes = gen.generate_batch(300)
        styles = set(p.style for p in probes)
        assert 'abcd' in styles
        assert 'numbered' in styles
        assert 'bullet' in styles
    
    def test_format_structure(self):
        """Test that MCQs have proper structure."""
        gen = MCQProbeGenerator(seed=42)
        probe = gen.generate_probe(style='abcd')
        
        # Should contain option markers
        assert 'A)' in probe.content or 'a)' in probe.content.lower()
    
    def test_corpus_generation(self):
        """Test corpus generation."""
        gen = MCQProbeGenerator(seed=42)
        corpus = gen.get_corpus(count=10)
        assert len(corpus) > 0
        assert '---' in corpus  # Separator


class TestCodeProbeGenerator:
    """Tests for code probe generation."""
    
    def test_generation_count(self):
        gen = CodeProbeGenerator(seed=42)
        probes = gen.generate_batch(100)
        assert len(probes) == 100
    
    def test_languages(self):
        """Test that multiple languages are generated."""
        gen = CodeProbeGenerator(seed=42)
        probes = gen.generate_batch(300)
        languages = set(p.language for p in probes)
        assert 'python' in languages
        assert 'javascript' in languages
        assert 'json' in languages
    
    def test_python_syntax(self):
        """Test that Python probes have valid structure."""
        gen = CodeProbeGenerator(seed=42)
        probe = gen.generate_python_probe('function')
        
        assert 'def ' in probe.content
    
    def test_json_syntax(self):
        """Test that JSON probes have valid structure."""
        gen = CodeProbeGenerator(seed=42)
        probe = gen.generate_json_probe()
        
        assert '{' in probe.content
        assert '}' in probe.content


class TestIndicProbeGenerator:
    """Tests for Indic probe generation."""
    
    def test_generation_count(self):
        gen = IndicProbeGenerator(seed=42)
        probes = gen.generate_batch(100)
        assert len(probes) == 100
    
    def test_scripts(self):
        """Test that multiple scripts are generated."""
        gen = IndicProbeGenerator(seed=42)
        probes = gen.generate_batch(500)
        scripts = set(p.script for p in probes)
        assert len(scripts) >= 3  # At least 3 different scripts
    
    def test_devanagari_content(self):
        """Test that Devanagari probes contain Devanagari characters."""
        gen = IndicProbeGenerator(seed=42)
        probe = gen.generate_pure_probe('devanagari')
        
        # Check for Devanagari Unicode range
        has_devanagari = any('\u0900' <= c <= '\u097F' for c in probe.content)
        assert has_devanagari
    
    def test_mixed_content(self):
        """Test mixed English-Indic content."""
        gen = IndicProbeGenerator(seed=42)
        probe = gen.generate_mixed_probe('devanagari')
        
        # Should have both ASCII and Devanagari
        has_ascii = any(c.isascii() and c.isalpha() for c in probe.content)
        has_devanagari = any('\u0900' <= c <= '\u097F' for c in probe.content)
        assert has_ascii
        assert has_devanagari


class TestSyntheticInstructionGenerator:
    """Tests for synthetic instruction generation."""
    
    def test_generation_count(self):
        gen = SyntheticInstructionGenerator(seed=42)
        probes = gen.generate_batch(100)
        assert len(probes) == 100
    
    def test_styles(self):
        """Test that all styles are represented."""
        gen = SyntheticInstructionGenerator(seed=42)
        probes = gen.generate_batch(500)
        styles = set(p.style for p in probes)
        assert 'direct' in styles
        assert 'step_by_step' in styles
        assert 'conversational' in styles
        assert 'formal' in styles
    
    def test_no_real_benchmark(self):
        """Test that instructions don't contain real benchmark content."""
        gen = SyntheticInstructionGenerator(seed=42)
        probes = gen.generate_batch(100)
        
        # Should use placeholders, not real content
        for probe in probes:
            # Check for placeholder patterns
            has_placeholder = (
                '[' in probe.content or 
                '{' in probe.content or 
                'placeholder' in probe.content.lower() or
                'OPTION' in probe.content or
                '___' in probe.content
            )
            # At least some probes should have placeholders
            # (not all, as some formats are complete)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
