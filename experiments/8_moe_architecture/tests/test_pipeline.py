"""Tests for coreset pipeline."""

import pytest
from src.coreset_builder.pipeline import CoresetPipeline


def test_pipeline_initialization():
    """Test pipeline initialization."""
    config = {
        "stage_name": "1B",
        "target_tokens": 20_000_000_000
    }
    pipeline = CoresetPipeline(config, seed=42)
    
    assert pipeline.stage_name == "1B"
    assert pipeline.target_tokens == 20_000_000_000
    assert pipeline.seed == 42


# TODO: Add more comprehensive pipeline tests
