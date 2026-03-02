"""Tests for OpusSelector."""
import torch
from llm.opus.config import OpusConfig
from llm.opus.selector import OpusSelector, SelectionResult


def test_selector_init():
    cfg = OpusConfig(sketch_dim=64, temperature=0.9, sketch_seed=42)
    selector = OpusSelector(cfg)
    assert selector is not None


def test_selection_result_dataclass():
    result = SelectionResult(
        selected_local_indices=torch.tensor([0, 2]),
        selected_global_indices=torch.tensor([0, 2]),
        used_fallback=False,
        metrics={"mean_score": 0.5},
    )
    assert result.selected_local_indices.shape == (2,)
    assert result.used_fallback is False
    assert result.metrics["mean_score"] == 0.5
