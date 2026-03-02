from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .countsketch import CountSketchProjector
from .preconditioner import AdamWPreconditionerView


class OpusGhostCollector:
    """
    Collects per-sample ghost factors (activations + grad_outputs) for linear layers
    during a single forward/backward pass, and immediately computes OPUS sketch scores.

    The input batch must be laid out as:
        [proxy_0, proxy_1, ..., proxy_{n_proxy-1}, cand_0, cand_1, ..., cand_{n_cand-1}]

    For each layer, when the backward hook fires we:
      1. Compute proxy sketches (no preconditioner) and average them -> (sketch_dim,)
      2. Compute candidate sketches (with preconditioner)            -> (n_cand, sketch_dim)
      3. Dot candidate sketches against proxy sketch                 -> (n_cand,)
      4. Accumulate into alignment_scores                            -> (n_cand,)
      5. Store candidate sketches for the redundancy term later      -> dict[layer -> (n_cand, sketch_dim)]

    After all hooks fire, call .results() to retrieve:
      - alignment_scores:   (n_cand,)              used directly in utility scoring
      - candidate_sketches: dict[str, (n_cand, m)] used in the Boltzmann selection loop
    """

    def __init__(
        self,
        model: nn.Module,
        n_proxy: int,
        n_candidates: int,
        preconditioner: AdamWPreconditionerView,
        sketcher: CountSketchProjector,
        device: torch.device,
    ):
        self.model = model
        self.n_proxy = n_proxy
        self.n_candidates = n_candidates
        self.preconditioner = preconditioner
        self.sketcher = sketcher
        self.device = device

        # Populated during forward/backward pass
        self._alignment_scores: Tensor | None = None
        self._candidate_sketches: dict[str, Tensor] = {}

        # Hook handles so we can cleanly remove them afterwards
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _discover_layers(self) -> list[tuple[str, nn.Linear]]:
        """Find all linear layers inside transformer blocks."""
        layers = []
        for name, module in self.model.named_modules():
            in_blocks = ("layers." in name) or ("blocks." in name)
            if isinstance(module, nn.Linear) and in_blocks:
                if getattr(module, "weight", None) is not None:
                    layers.append((name, module))
        if not layers:
            raise RuntimeError("OPUSGhostCollector found no scoreable linear layers")
        return layers

    def _make_forward_hook(self, name: str):
        """
        Forward hook: captures input activations for this layer.
        Stored temporarily until the backward hook fires for the same layer.
        """

        def hook(module: nn.Module, args: tuple[Any, ...], output: Tensor):
            if not args or not torch.is_tensor(args[0]):
                return
            # args[0] is the input to this linear layer: (B, T, in_dim) or (B, in_dim)
            self._activations_buffer[name] = args[0].detach()

        return hook

    def _make_backward_hook(self, name: str, module: nn.Linear):
        """
        Backward hook: fires after gradients are computed for this layer.
        At this point we have both activations (from forward hook) and grad_outputs,
        so we can compute all sketches immediately and discard the raw tensors.
        """

        def hook(
            module: nn.Module,
            grad_input: tuple[Tensor, ...],
            grad_output: tuple[Tensor, ...],
        ):
            if not grad_output or not torch.is_tensor(grad_output[0]):
                return
            if name not in self._activations_buffer:
                return

            activations = self._activations_buffer.pop(
                name
            )  # (B, T, in_dim) or (B, in_dim)
            grad_out = grad_output[0].detach()  # (B, T, out_dim) or (B, out_dim)

            # Ensure both are (B, T, D) for the sketcher
            if activations.dim() == 2:
                activations = activations.unsqueeze(1)
            if grad_out.dim() == 2:
                grad_out = grad_out.unsqueeze(1)

            out_dim, in_dim = module.weight.shape  # type: ignore

            # --- Proxy sketches (no preconditioner) ---
            # Shape: (n_proxy, sketch_dim)
            proxy_sketches = self.sketcher.project_linear_batch(
                activations=activations[: self.n_proxy],
                grad_outputs=grad_out[: self.n_proxy],
                preconditioner=None,
                out_dim=out_dim,  # type: ignore
                in_dim=in_dim,  # type: ignore
                sketch_key=name,
            )
            # Average across proxy samples -> (sketch_dim,)
            # This gives us ψ^(r)_proxy = mean gradient direction of proxy distribution
            proxy_sketch_mean = proxy_sketches.mean(dim=0)

            # --- Candidate sketches (with preconditioner) ---
            # Get AdamW preconditioner for this layer's weight: (out_dim, in_dim)
            # Falls back to identity scaling if optimizer state isn't available on this GPU
            prec = self.preconditioner.get(module.weight)  # type: ignore

            # Shape: (n_candidates, sketch_dim)
            cand_sketches = self.sketcher.project_linear_batch(
                activations=activations[self.n_proxy :],
                grad_outputs=grad_out[self.n_proxy :],
                preconditioner=prec,
                out_dim=out_dim,  # type: ignore
                in_dim=in_dim,  # type: ignore
                sketch_key=name,
            )

            # --- Alignment contribution for this layer ---
            # (n_candidates, sketch_dim) @ (sketch_dim,) -> (n_candidates,)
            # This is ⟨φ^(r)(z), ψ^(r)_proxy⟩ for each candidate z
            layer_alignment = cand_sketches @ proxy_sketch_mean

            # Accumulate into running sum across layers: Σ_r ⟨φ^(r)(z), ψ^(r)_proxy⟩
            if self._alignment_scores is None:
                self._alignment_scores = layer_alignment
            else:
                self._alignment_scores = self._alignment_scores + layer_alignment

            # Store candidate sketches for the redundancy term during Boltzmann selection
            # φ^(r)(z) for all candidates at this layer
            self._candidate_sketches[name] = cand_sketches

        return hook

    def register(self) -> None:
        """Attach forward and backward hooks to all linear layers in transformer blocks."""
        self._activations_buffer: dict[str, Tensor] = {}
        self._alignment_scores = None
        self._candidate_sketches = {}

        for name, module in self._discover_layers():
            self._handles.append(
                module.register_forward_hook(self._make_forward_hook(name))
            )
            self._handles.append(
                module.register_full_backward_hook(
                    self._make_backward_hook(name, module)  # type: ignore
                )
            )

    def unregister(self) -> None:
        """Remove all hooks."""
        while self._handles:
            self._handles.pop().remove()

    def results(self) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Returns:
            alignment_scores:   (n_candidates,)
                Sum of per-layer dot products between candidate and proxy sketches.
                Used as the alignment term in OPUS utility scoring.

            candidate_sketches: dict[layer_name -> (n_candidates, sketch_dim)]
                Per-layer candidate sketches needed for the redundancy penalty
                during the Boltzmann selection loop.
        """
        if self._alignment_scores is None:
            raise RuntimeError(
                "No scores collected — did you run a forward/backward pass?"
            )
        return self._alignment_scores, self._candidate_sketches

    def clear(self) -> None:
        """Reset all collected state."""
        self._activations_buffer = {}
        self._alignment_scores = None
        self._candidate_sketches = {}

    def __enter__(self) -> "OpusGhostCollector":
        self.register()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.unregister()
        self.clear()
