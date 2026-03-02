"""Tests for CountSketchProjector determinism and correctness."""
import torch
from llm.opus.countsketch import CountSketchProjector


def test_countsketch_deterministic():
    proj = CountSketchProjector(sketch_dim=64, seed=42)
    # project_linear_batch expects activations [B, in_dim], grad_outputs [B, out_dim]
    activations = torch.randn(10, 128)
    grad_outputs = torch.randn(10, 64)
    out1 = proj.project_linear_batch(activations, grad_outputs, preconditioner=None, out_dim=64, in_dim=128)
    out2 = proj.project_linear_batch(activations, grad_outputs, preconditioner=None, out_dim=64, in_dim=128)
    assert torch.allclose(out1, out2), "CountSketch must be deterministic"


def test_countsketch_output_shape():
    proj = CountSketchProjector(sketch_dim=256, seed=42)
    activations = torch.randn(5, 1024)
    grad_outputs = torch.randn(5, 512)
    out = proj.project_linear_batch(activations, grad_outputs, preconditioner=None, out_dim=512, in_dim=1024)
    assert out.shape == (5, 256)


def test_countsketch_different_seeds_differ():
    activations = torch.randn(5, 128)
    grad_outputs = torch.randn(5, 64)
    out1 = CountSketchProjector(sketch_dim=64, seed=1).project_linear_batch(
        activations, grad_outputs, preconditioner=None, out_dim=64, in_dim=128
    )
    out2 = CountSketchProjector(sketch_dim=64, seed=2).project_linear_batch(
        activations, grad_outputs, preconditioner=None, out_dim=64, in_dim=128
    )
    assert not torch.allclose(out1, out2)
