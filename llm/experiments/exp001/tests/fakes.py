"""
Lightweight fakes/stubs for testing the OPUS training pipeline.

These replace heavy dependencies (DeepSpeed, models, OPUS components)
with minimal CPU-only stand-ins.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FakeDeepSpeedEngine(nn.Module):
    """Minimal stand-in for deepspeed.DeepSpeedEngine."""

    def __init__(self, model: nn.Module, device: torch.device | None = None):
        super().__init__()
        self.module = model
        self.device = device or torch.device("cpu")
        self._lr = 3e-4

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        self.module.zero_grad()

    def get_lr(self) -> list[float]:
        return [self._lr]

    def zero_optimization_stage(self) -> int:
        return 0

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class FakeModel(nn.Module):
    """Tiny model that mimics Model1B's forward signature."""

    def __init__(self, hidden_size: int = 32, vocab_size: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.layers = nn.ModuleList()

    def forward(
        self,
        input_ids,
        next_token_ids=None,
        attention_mask=None,
        return_loss=True,
        return_memory=False,
        prev_memory_stream=None,
        return_hidden=True,
    ):
        h = self.embed(input_ids).to(torch.bfloat16)
        aux_loss = torch.tensor(0.0, device=h.device, dtype=h.dtype)
        return h, None, aux_loss


class FakeProxyProvider:
    """Stand-in for RandomInDistributionProxyProvider."""

    def __init__(self, vocab_size: int = 128):
        self.vocab_size = vocab_size

    def sample(
        self, device: torch.device, k: int, seq_len: int
    ) -> torch.Tensor:
        return torch.randint(0, self.vocab_size, (k, seq_len), device=device)


class FakeGhostCollector:
    """Stand-in for OpusGhostCollector (context manager)."""

    def __init__(self, n_candidates: int, sketch_dim: int = 512, **kwargs):
        self.n_candidates = n_candidates
        self.sketch_dim = sketch_dim

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def register(self):
        pass

    def unregister(self):
        pass

    def clear(self):
        pass

    def results(self):
        alignment = torch.randn(self.n_candidates)
        sketches = {"fake_layer": torch.randn(self.n_candidates, self.sketch_dim)}
        return alignment, sketches


class FakeSelector:
    """Stand-in for OpusSelector."""

    def __init__(self, n_to_select: int, **kwargs):
        self.n_to_select = n_to_select

    def select(self, alignment_scores, candidate_sketches, learning_rate):
        from exp.opus import SelectionResult

        n = alignment_scores.size(0)
        k = min(self.n_to_select, n)
        indices = torch.randperm(n)[:k]
        return SelectionResult(
            selected_local_indices=indices,
            selected_global_indices=indices,
            used_fallback=False,
            metrics={
                "alignment": 0.5,
                "redundancy": 0.02,
                "entropy": 2.1,
                "selector_time_s": 0.01,
            },
        )


class FakeFusedCE(nn.Module):
    """Stand-in for FusedLinearCrossEntropyLoss."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, hidden, weight, targets):
        logits = hidden.float() @ weight.float().t()
        return nn.functional.cross_entropy(logits, targets)
