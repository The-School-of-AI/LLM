from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict

import torch

from lightninglm.opus.distributed import get_rank


@dataclass
class SelectionResult:
    # Indices into this GPU's local candidate pool.
    selected_local_indices: torch.Tensor  # (k_local,)

    # Global indices encoding both owner GPU and local position:
    #   global_idx = owner_rank * n_local + local_idx
    selected_global_indices: torch.Tensor  # (k_global,)

    # True if the random fallback was triggered
    used_fallback: bool

    # Diagnostic metrics from the selection
    metrics: Dict[str, float]


class OpusSelector:
    """
    Local top-k selection for OPUS with Gumbel noise for stochasticity.

    Each GPU independently selects its top-k candidates from its local pool
    based on alignment scores + Gumbel noise. No cross-GPU communication
    needed since each GPU has different candidates and different data.

    The alignment scores already incorporate the model's gradient direction
    (via ghost hooks). Each GPU selects the most useful candidates for its
    own training batch.
    """

    def __init__(
        self,
        selection_ratio: float,
        temperature: float,
        seed: int = 42,
        max_selector_time_s: float = 30.0,
        fallback_random_on_error: bool = True,
    ):
        if not (0.0 < selection_ratio <= 1.0):
            raise ValueError(
                f"selection_ratio must be in (0, 1], got {selection_ratio}"
            )
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.selection_ratio = selection_ratio
        self.temperature = max(temperature, 1e-6)
        self.max_selector_time_s = max_selector_time_s
        self.fallback_random_on_error = fallback_random_on_error

        # Per-GPU RNG for Gumbel noise
        rank = get_rank()
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(seed + rank)

    def select(
        self,
        alignment_scores: torch.Tensor,  # (n_local,)
        candidate_sketches: Dict[str, torch.Tensor],  # layer -> (n_local, sketch_dim)
        learning_rate: float,
    ) -> SelectionResult:
        """
        Select top-k candidates from local pool using alignment scores + Gumbel noise.

        Fast, O(n_local log n_local) local sort. No NCCL communication.
        """
        t_start = time.perf_counter()
        device = alignment_scores.device
        lr = float(learning_rate)
        n_local = alignment_scores.shape[0]
        rank = get_rank()

        # How many to select from this GPU's local pool
        k_local = max(1, int(round(self.selection_ratio * n_local)))

        # Scale alignment by learning rate
        scores = lr * alignment_scores.float()

        # Check for non-finite scores
        nonfinite_mask = ~torch.isfinite(scores)
        nonfinite_count = int(nonfinite_mask.sum().item())
        scores = torch.nan_to_num(
            scores,
            nan=torch.finfo(torch.float32).min,
            posinf=torch.finfo(torch.float32).min,
            neginf=torch.finfo(torch.float32).min,
        )

        # Add Gumbel noise for stochastic selection
        u = (
            torch.rand(scores.shape, generator=self._rng, dtype=torch.float32)
            .clamp_(min=1e-6, max=1.0 - 1e-6)
            .to(device)
        )
        gumbel_noise = -torch.log(-torch.log(u))
        noisy_scores = scores / self.temperature + gumbel_noise

        # Check if all scores are non-finite after processing
        if not torch.any(torch.isfinite(noisy_scores)):
            # Fallback to random selection
            perm = torch.randperm(n_local, generator=self._rng, device="cpu")[:k_local]
            local_indices = perm.to(device)
            return SelectionResult(
                selected_local_indices=local_indices,
                selected_global_indices=local_indices + (rank * n_local),
                used_fallback=True,
                metrics={
                    "alignment": 0.0,
                    "redundancy": 0.0,
                    "entropy": 0.0,
                    "nonfinite_scores": float(nonfinite_count),
                    "fallback_no_finite": 1.0,
                    "used_fallback": 1.0,
                    "selector_time_s": float(time.perf_counter() - t_start),
                },
            )

        # Select top-k by noisy scores
        _, top_indices = torch.topk(noisy_scores, k=min(k_local, n_local))
        local_indices = top_indices

        # Compute metrics from selected candidates
        selected_scores = scores[local_indices]
        alignment_acc = float(selected_scores.sum().item())

        # Redundancy: compute pairwise similarity of selected sketches
        layer_names = sorted(candidate_sketches.keys())
        if layer_names and local_indices.numel() > 1:
            sel_sketches = torch.stack(
                [
                    candidate_sketches[name][local_indices].float()
                    for name in layer_names
                ],
                dim=1,
            )  # (k_local, n_layers, sketch_dim)
            # Frobenius norm of sketch matrix as redundancy proxy
            redundancy_acc = float(torch.norm(sel_sketches).item())
        else:
            redundancy_acc = 0.0

        # Entropy from noisy scores
        sel_logits = noisy_scores[local_indices]
        entropy_acc = 0.0
        for s in sel_logits:
            sv = float(s.item())
            if math.isfinite(sv):
                p = math.exp(min(sv, 80.0))
                if p > 0.0:
                    entropy_acc += -p * math.log(p + 1e-12)

        return SelectionResult(
            selected_local_indices=local_indices,
            selected_global_indices=local_indices + (rank * n_local),
            used_fallback=False,
            metrics={
                "alignment": alignment_acc,
                "redundancy": redundancy_acc,
                "entropy": entropy_acc,
                "nonfinite_scores": float(nonfinite_count),
                "fallback_no_finite": 0.0,
                "used_fallback": 0.0,
                "selector_time_s": float(time.perf_counter() - t_start),
            },
        )
