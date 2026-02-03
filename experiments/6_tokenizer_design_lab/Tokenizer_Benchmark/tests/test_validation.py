"""
Tests for validation checks.

Validates neutrality checking, curriculum analysis, and routing skew detection.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from validation import (
    NeutralityChecker,
    check_format_only_compliance,
    CurriculumAnalyzer,
    analyze_difficulty_bands,
    RoutingSkewAnalyzer,
    calculate_routing_entropy,
)


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self, tokens_per_char: float = 0.25):
        self.ratio = tokens_per_char
    
    def encode(self, text: str) -> list:
        n = int(1 / self.ratio) if self.ratio > 0 else 4
        # Generate deterministic token IDs based on text
        return [hash(text[i:i+n]) % 1000 for i in range(0, len(text), n)]
    
    def decode(self, token_ids: list) -> str:
        return "x" * (len(token_ids) * 4)
    
    def vocab_size(self) -> int:
        return 1000


class TestNeutralityChecker:
    """Tests for neutrality checking."""
    
    def test_clean_probes_pass(self):
        """Clean format-only probes should pass."""
        probes = [
            "$\\frac{a}{b} + c = d$",
            "def process_data(x): pass",
            "Question: Which option?\nA) First\nB) Second",
        ]
        
        checker = NeutralityChecker(sensitivity="high")
        result = checker.check_probe_neutrality(probes)
        
        assert result.passed
        assert result.score > 0.8
    
    def test_suspicious_probes_fail(self):
        """Probes with benchmark terms should be flagged."""
        probes = [
            "This is a gsm8k style question",
            "MMLU benchmark test",
            "hellaswag evaluation",
        ]
        
        checker = NeutralityChecker(sensitivity="high")
        result = checker.check_probe_neutrality(probes)
        
        # Should detect suspicious content
        assert len(result.issues) > 0
    
    def test_format_only_compliance(self):
        """Test the convenience function."""
        probes = [
            "[PLACEHOLDER] content here",
            "Option A: ___",
            "def func(): pass",
        ]
        
        result = check_format_only_compliance(probes)
        assert result.passed
    
    def test_sensitivity_levels(self):
        """Test different sensitivity levels."""
        probes = ["Normal text without issues"] * 10
        
        for level in ["low", "medium", "high"]:
            checker = NeutralityChecker(sensitivity=level)
            result = checker.check_probe_neutrality(probes)
            assert result.passed


class TestCurriculumAnalyzer:
    """Tests for curriculum difficulty analysis."""
    
    def test_balanced_difficulties(self):
        """Balanced tokenization across difficulties should pass."""
        tokenizer = MockTokenizer(0.25)
        
        # Similar content at each difficulty
        easy = ["easy text one", "easy text two"] * 10
        medium = ["medium text one", "medium text two"] * 10
        hard = ["hard text one", "hard text two"] * 10
        
        result = analyze_difficulty_bands(tokenizer, easy, medium, hard)
        
        # Should pass with balanced content
        assert result.distortion_score < 0.5
    
    def test_extreme_distortion(self):
        """Extreme differences should be detected."""
        # Create a tokenizer that behaves very differently on different content
        class BiasedTokenizer:
            def encode(self, text: str) -> list:
                if 'easy' in text:
                    return list(range(100))  # Many tokens for easy
                elif 'hard' in text:
                    return list(range(5))  # Few tokens for hard
                return list(range(20))
        
        tokenizer = BiasedTokenizer()
        
        easy = ["easy " * 10] * 10
        medium = ["medium " * 10] * 10
        hard = ["hard " * 10] * 10
        
        result = analyze_difficulty_bands(tokenizer, easy, medium, hard)
        
        # Should detect distortion
        assert result.distortion_score > 0
    
    def test_empty_bands(self):
        """Empty difficulty bands should be handled."""
        tokenizer = MockTokenizer()
        
        result = analyze_difficulty_bands(tokenizer, [], [], [])
        
        # Should not crash
        assert result is not None


class TestRoutingSkewAnalyzer:
    """Tests for routing skew detection."""
    
    def test_balanced_distribution(self):
        """Balanced token distribution should have high entropy."""
        tokenizer = MockTokenizer()
        
        texts = [f"text sample number {i}" for i in range(100)]
        
        analyzer = RoutingSkewAnalyzer()
        result = analyzer.analyze(tokenizer, texts)
        
        # Should have reasonable entropy
        assert result.entropy > 0
        assert result.normalized_entropy > 0
    
    def test_skewed_distribution(self):
        """Highly skewed distribution should be detected."""
        # Create a tokenizer that always returns the same tokens
        class SkewedTokenizer:
            def encode(self, text: str) -> list:
                # Always return token ID 42
                return [42] * (len(text) // 4 + 1)
            
            def vocab_size(self) -> int:
                return 1000
        
        tokenizer = SkewedTokenizer()
        texts = ["test " * 10] * 50
        
        analyzer = RoutingSkewAnalyzer(entropy_threshold=0.8)
        result = analyzer.analyze(tokenizer, texts)
        
        # Should detect skew (low entropy)
        assert result.normalized_entropy < 0.1
        assert result.skew_detected
    
    def test_entropy_calculation(self):
        """Test the entropy convenience function."""
        tokenizer = MockTokenizer()
        texts = ["hello world", "test text", "sample data"]
        
        entropy = calculate_routing_entropy(tokenizer, texts)
        
        assert 0 <= entropy <= 1
    
    def test_top_tokens_reporting(self):
        """Test that top tokens are reported."""
        tokenizer = MockTokenizer()
        texts = ["test text"] * 20
        
        analyzer = RoutingSkewAnalyzer()
        result = analyzer.analyze(tokenizer, texts, top_k=10)
        
        assert len(result.top_tokens) <= 10
        if result.top_tokens:
            assert 'token_id' in result.top_tokens[0]
            assert 'count' in result.top_tokens[0]
            assert 'percentage' in result.top_tokens[0]


class TestValidationIntegration:
    """Integration tests for validation pipeline."""
    
    def test_full_validation_pipeline(self):
        """Test running all validations together."""
        tokenizer = MockTokenizer()
        
        # Generate some test probes
        probes = [
            "def function(): pass",
            "Question: Option?\nA) Yes\nB) No",
            "$x + y = z$",
        ] * 20
        
        difficulty_bands = {
            'easy': [p for p in probes[:20]],
            'medium': [p for p in probes[20:40]],
            'hard': [p for p in probes[40:]],
        }
        
        # Run all validators
        neutrality_checker = NeutralityChecker()
        neutrality_result = neutrality_checker.check_probe_neutrality(probes)
        
        curriculum_analyzer = CurriculumAnalyzer()
        curriculum_result = curriculum_analyzer.analyze(tokenizer, difficulty_bands)
        
        routing_analyzer = RoutingSkewAnalyzer()
        routing_result = routing_analyzer.analyze(tokenizer, probes)
        
        # All should complete without error
        assert neutrality_result is not None
        assert curriculum_result is not None
        assert routing_result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
