"""Tests for GhostCollector hook attachment and capture."""
import torch
import torch.nn as nn
from llm.opus.ghost import GhostCollector


def _make_tiny_model():
    """Build a model with 'layers.' in the module path so _should_track finds it."""
    model = nn.Module()
    block = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16))
    model.add_module("layers", nn.ModuleList([block]))
    return model


def test_ghost_collector_captures_activations():
    model = _make_tiny_model()
    x = torch.randn(2, 16)
    with GhostCollector(model, include_embeddings=False, include_lm_head=False) as gc:
        out = model.layers[0](x)
        loss = out.sum()
        loss.backward()
        captures = gc.captures()
    assert len(captures) > 0, "GhostCollector should capture at least one layer"
    for name, cap in captures.items():
        assert cap.activations is not None
        assert cap.grad_outputs is not None


def test_ghost_collector_cleanup():
    model = _make_tiny_model()
    with GhostCollector(model) as gc:
        # Run a forward+backward so register() succeeds
        out = model.layers[0](torch.randn(2, 16))
        out.sum().backward()
    # After exiting context, all hooks should be removed
    for mod in model.modules():
        assert len(mod._forward_hooks) == 0
        assert len(mod._backward_hooks) == 0
