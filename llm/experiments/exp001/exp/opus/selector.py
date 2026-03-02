from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from distributed import (
    all_gather_1d,
    get_world_size,
)


@dataclass
class SelectionResult:
    global_indices: (
        torch.Tensor
    )  # (k,) indices into the globally gathered candidate pool, identical on all GPUs


class OpusSelector:
    """
    Runs the iterative Boltzmann selection loop using outputs from OPUSGhostCollector.

    Inputs from the collector (per GPU, local candidates only):
        alignment_scores:   (n_candidates_local,)
            Already summed over layers: Σ_r ⟨φ^(r)(z), ψ^(r)_proxy⟩

        candidate_sketches: dict[layer_name -> (n_candidates_local, sketch_dim)]
            Per-layer sketches needed for the redundancy penalty.

    The selector:
        1. All-gathers both across GPUs so every GPU has the full candidate pool
        2. All GPUs run the identical deterministic Boltzmann loop independently
        3. All GPUs arrive at the same global_indices — no broadcast needed

    Why no broadcast?
        The Boltzmann loop is fully deterministic given the same inputs and the same
        rng seed. Since all GPUs have identical tensors after all-gather, and all
        OPUSSelector instances are constructed with the same seed, the rng advances
        identically on every GPU. So all GPUs independently pick the same candidates.
    """

    def __init__(
        self,
        selection_ratio: float = 0.5,
        temperature: float = 0.9,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ):
        if not (0.0 < selection_ratio <= 1.0):
            raise ValueError(
                f"selection_ratio must be in (0, 1], got {selection_ratio}"
            )
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.selection_ratio = selection_ratio
        self.temperature = temperature
        self.learning_rate = learning_rate

        # All GPUs must use the same seed so Gumbel sampling is identical everywhere
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(seed)

    # ------------------------------------------------------------------
    # Distributed gathering
    # ------------------------------------------------------------------

    def _gather_alignment_scores(self, local_scores: torch.Tensor) -> torch.Tensor:
        """
        All-gather 1D alignment scores across GPUs.
        (n_local,) on each GPU -> (n_local * world_size,) on all GPUs
        """
        return all_gather_1d(local_scores)

    def _gather_candidate_sketches(
        self, local_sketches: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        All-gather per-layer sketch tensors across GPUs.
        Each layer: (n_local, m) on each GPU -> (n_total, m) on all GPUs

        Flattened to 1D for all_gather_1d, then reshaped afterwards.
        """
        gathered: Dict[str, torch.Tensor] = {}
        n_total = None

        for layer_name, sketches in local_sketches.items():
            n_local, sketch_dim = sketches.shape
            flat = sketches.reshape(-1)  # (n_local * sketch_dim,)
            flat_global = all_gather_1d(flat)  # (n_total * sketch_dim,)
            if n_total is None:
                n_total = n_local * get_world_size()
            gathered[layer_name] = flat_global.reshape(n_total, sketch_dim)

        return gathered

    # ------------------------------------------------------------------
    # Gumbel-argmax sampling
    # ------------------------------------------------------------------

    def _gumbel_argmax(self, logits: torch.Tensor) -> int:
        """
        Equivalent to sampling from categorical(softmax(logits)) but via argmax.

        For each logit i:
            1. Sample U_i ~ Uniform(0, 1)
            2. Compute Gumbel noise: G_i = -log(-log(U_i))
            3. Return argmax(logits + G)

        Since all GPUs use the same rng seed and call this at identical loop
        iterations with identical logits, they all produce the same chosen index.
        No broadcast needed.
        """
        u = torch.rand(logits.shape, generator=self._rng, dtype=torch.float32).clamp_(
            min=1e-6, max=1.0 - 1e-6
        )
        gumbel_noise = -torch.log(-torch.log(u))
        return int(torch.argmax(logits + gumbel_noise).item())

    # ------------------------------------------------------------------
    # Main selection loop
    # ------------------------------------------------------------------

    def select(
        self,
        alignment_scores: torch.Tensor,  # (n_candidates_local,)
        candidate_sketches: Dict[str, torch.Tensor],  # layer -> (n_candidates_local, m)
    ) -> SelectionResult:
        device = alignment_scores.device

        # --- Step 1: All-gather so every GPU has the full candidate pool ---
        global_alignment = self._gather_alignment_scores(
            alignment_scores.to(torch.float32)
        )
        # (n_candidates_total,)

        global_sketches = self._gather_candidate_sketches(candidate_sketches)
        # dict[layer -> (n_candidates_total, m)]

        n_total = global_alignment.shape[0]
        k = max(1, round(self.selection_ratio * n_total))

        # --- Step 2: Stack per-layer sketches into one tensor ---
        # Sorting layer names ensures identical ordering on all GPUs
        layer_names = sorted(global_sketches.keys())
        cand_stack = torch.stack(
            [global_sketches[name].to(torch.float32) for name in layer_names], dim=1
        )
        # (n_candidates_total, n_layers, sketch_dim)

        # --- Step 3: Iterative Boltzmann selection loop ---

        # history = Φ^(r) from the paper: running sum of selected candidate sketches
        # used to penalize redundant picks each iteration
        history = torch.zeros_like(cand_stack[0])  # (n_layers, sketch_dim)

        # mask to prevent re-selecting an already chosen candidate
        already_selected = torch.zeros(n_total, dtype=torch.bool, device=device)

        selected_indices: List[int] = []

        lr = self.learning_rate
        tau = max(self.temperature, 1e-6)

        for _ in range(k):
            # Alignment term: lr * Σ_r ⟨φ^(r)(z), ψ^(r)_proxy⟩
            # global_alignment already contains the layer sum from the collector
            alignment_term = lr * global_alignment
            # (n_candidates_total,)

            # Redundancy term: lr² * Σ_r ⟨φ^(r)(z), Φ^(r)⟩
            # history accumulates sketches of already-selected candidates
            redundancy_term = (lr**2) * torch.einsum("nlm,lm->n", cand_stack, history)
            # (n_candidates_total,)

            # Full utility score per candidate
            utility = alignment_term - redundancy_term
            # (n_candidates_total,)

            # Mask out already-selected candidates so they can't be picked again
            utility[already_selected] = -torch.inf

            # Apply temperature scaling for Boltzmann distribution
            logits = utility / tau

            # Gumbel-argmax: equivalent to sampling from softmax(logits)
            # Runs on CPU since _rng is a CPU generator
            # Identical result on all GPUs — no broadcast needed
            chosen = self._gumbel_argmax(logits.cpu())

            # Update state for next iteration
            selected_indices.append(chosen)
            already_selected[chosen] = True
            history = history + cand_stack[chosen]  # update Φ^(r) for all layers

        return SelectionResult(
            global_indices=torch.tensor(
                selected_indices, device=device, dtype=torch.long
            )
        )
