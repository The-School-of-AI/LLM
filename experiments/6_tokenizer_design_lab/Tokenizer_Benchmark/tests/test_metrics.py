"""
Tests for metrics calculations.

Validates metric functions with known inputs.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics import (
    tokens_per_byte,
    tokens_per_char,
    compression_ratio,
    compute_compression_metrics,
    fertility,
    word_tokenize,
    per_language_fertility,
)


class MockTokenizer:
    """Simple mock tokenizer for testing."""
    
    def __init__(self, tokens_per_char_ratio: float = 0.25):
        self.ratio = tokens_per_char_ratio
    
    def encode(self, text: str) -> list:
        # Return one token per N characters
        n = int(1 / self.ratio) if self.ratio > 0 else 4
        return list(range(len(text) // n + 1))
    
    def decode(self, token_ids: list) -> str:
        return "x" * (len(token_ids) * 4)
    
    def vocab_size(self) -> int:
        return 50000


class TestCompressionMetrics:
    """Tests for compression metrics."""
    
    def test_tokens_per_byte_basic(self):
        tokenizer = MockTokenizer(0.25)  # 1 token per 4 chars
        text = "hello world"  # 11 chars = 11 bytes (ASCII)
        
        tpb = tokens_per_byte(tokenizer, text)
        # 3 tokens for 11 bytes = 0.27
        assert 0.2 < tpb < 0.4
    
    def test_tokens_per_byte_empty(self):
        tokenizer = MockTokenizer()
        assert tokens_per_byte(tokenizer, "") == 0.0
    
    def test_tokens_per_char_basic(self):
        tokenizer = MockTokenizer(0.25)
        text = "hello world"
        
        tpc = tokens_per_char(tokenizer, text)
        assert 0.2 < tpc < 0.4
    
    def test_compression_ratio_basic(self):
        tokenizer = MockTokenizer(0.25)
        text = "hello world"
        
        cr = compression_ratio(tokenizer, text)
        # bytes per token, should be > 1
        assert cr > 2  # 11 bytes / 3 tokens ≈ 3.67
    
    def test_compute_compression_metrics(self):
        tokenizer = MockTokenizer()
        texts = ["hello", "world", "test"]
        
        metrics = compute_compression_metrics(tokenizer, texts)
        
        assert 'tokens_per_byte' in metrics
        assert 'tokens_per_char' in metrics
        assert 'compression_ratio' in metrics
        assert 'total_tokens' in metrics
        assert 'num_samples' in metrics
        assert metrics['num_samples'] == 3
    
    def test_compression_per_text(self):
        tokenizer = MockTokenizer()
        texts = ["hello", "world"]
        
        metrics = compute_compression_metrics(tokenizer, texts, aggregate=False)
        
        assert 'per_text' in metrics
        assert len(metrics['per_text']) == 2


class TestFertilityMetrics:
    """Tests for fertility metrics."""
    
    def test_word_tokenize_english(self):
        text = "Hello world, this is a test!"
        words = word_tokenize(text, 'english')
        
        assert 'Hello' in words
        assert 'world' in words
        assert 'test' in words
        assert ',' not in words  # Punctuation excluded
    
    def test_word_tokenize_devanagari(self):
        # Simple test with Devanagari characters
        text = "नमस्ते दुनिया"
        words = word_tokenize(text, 'devanagari')
        
        assert len(words) == 2
    
    def test_word_tokenize_generic(self):
        text = "Hello world 123"
        words = word_tokenize(text, 'generic')
        
        assert len(words) == 3
    
    def test_fertility_basic(self):
        tokenizer = MockTokenizer(0.25)
        text = "hello world"  # 2 words
        
        fert = fertility(tokenizer, text, 'english')
        # Should be tokens / words
        assert fert > 0
    
    def test_fertility_empty(self):
        tokenizer = MockTokenizer()
        assert fertility(tokenizer, "", 'english') == 0.0
    
    def test_per_language_fertility(self):
        tokenizer = MockTokenizer()
        corpus = {
            'english': 'Hello world this is a test',
            'generic': 'Some other text here',
        }
        
        results = per_language_fertility(tokenizer, corpus)
        
        assert 'english' in results
        assert 'generic' in results
        assert 'fertility' in results['english']
        assert 'num_words' in results['english']


class TestWordTokenization:
    """Additional tests for word tokenization edge cases."""
    
    def test_empty_text(self):
        assert word_tokenize("", "english") == []
    
    def test_only_punctuation(self):
        words = word_tokenize("...,,,!!!", "english")
        assert len(words) == 0
    
    def test_mixed_scripts(self):
        text = "Hello नमस्ते world"
        
        english_words = word_tokenize(text, 'english')
        devanagari_words = word_tokenize(text, 'devanagari')
        generic_words = word_tokenize(text, 'generic')
        
        assert 'Hello' in english_words
        assert 'world' in english_words
        assert len(devanagari_words) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
