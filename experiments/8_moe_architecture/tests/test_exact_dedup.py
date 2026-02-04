"""Tests for exact deduplication."""

import pytest
from src.deduplication.exact_dedup import ExactDeduplicator


def test_exact_dedup_identical():
    """Test that identical chunks are deduplicated."""
    deduplicator = ExactDeduplicator(seed=42)
    chunks = ["hello world", "hello world", "different text"]
    
    kept, duplicates = deduplicator.deduplicate(chunks)
    
    assert len(kept) == 2
    assert len(duplicates) == 1
    assert 0 in kept
    assert 2 in kept
    assert 1 in duplicates


def test_exact_dedup_all_unique():
    """Test that unique chunks are all kept."""
    deduplicator = ExactDeduplicator(seed=42)
    chunks = ["first", "second", "third"]
    
    kept, duplicates = deduplicator.deduplicate(chunks)
    
    assert len(kept) == 3
    assert len(duplicates) == 0


def test_hash_deterministic():
    """Test that hashing is deterministic."""
    dedup1 = ExactDeduplicator(seed=42)
    dedup2 = ExactDeduplicator(seed=42)
    
    text = "test string"
    assert dedup1.hash_chunk(text) == dedup2.hash_chunk(text)
