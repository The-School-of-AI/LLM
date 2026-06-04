from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .countsketch import CountSketchProjector
from .preconditioner import AdamWPreconditionerView


class OpusGhostCollector:
    """
    Collects per-sample ghost factors (activations + grad_outputs) for linear layers
    during a single forward/backward pass, and computes OPUS sketch scores using
    the Tensor Sketch (FFT convolution) approach from the paper.

    Key optimizations vs previous bmm-based implementation:
      1. Tensor Sketch: CS(b⊗a) = IFFT(FFT(CS_row(b)) · FFT(CS_col(a)))
         Cost: O(d_in + d_out + m·log(m)) per token vs O(d_in·d_out). ~150x speedup.
      2. Factored preconditioner: P[i,j] ≈ p_row[i]·p_col[j] (Kronecker rank-1)
         Preserves tensor sketch separability while keeping optimizer-aware scoring.
      3. Paper-correct split: proxy is UNPRECONDITIONED, candidates are PRECONDITIONED
         (Eq. 25 of the paper).
      4. Deferred all-reduce: accumulate local alignment scores during backward,
         single all_reduce of (n_cand,) floats in results() instead of per-layer.

    Math for deferred all-reduce:
        alignment = Σ_l (cand[l] · global_proxy[l])
                  = Σ_l (cand[l] · (1/W) Σ_r proxy_r[l])
                  = (1/W) all_reduce( Σ_l (cand[l] · local_proxy[l]) )
    """

    # Minimum output features for a layer to be worth scoring
    MIN_OUT_FEATURES = 256

    def __init__(
        self,
        model: nn.Module,
        n_proxy: int,
        n_candidates: int,
        preconditioner: AdamWPreconditionerView,
        sketcher: CountSketchProjector,
        device: torch.device,
        score_layer_stride: int = 1,
    ):
        self.model = model
        self.n_proxy = n_proxy
        self.n_candidates = n_candidates
        self.preconditioner = preconditioner
        self.sketcher = sketcher
        self.device = device
        self.score_layer_stride = max(1, int(score_layer_stride))

        # Populated during forward/backward pass
        self._alignment_scores: Tensor | None = None
        self._candidate_sketches: dict[str, Tensor] = {}

        # Hook handles
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _discover_layers(self) -> list[tuple[str, nn.Linear]]:
        """Find large linear layers inside transformer blocks, applying stride."""
        all_eligible = []
        skipped_small = []
        for name, module in self.model.named_modules():
            in_blocks = ("layers." in name) or ("blocks." in name)
            if isinstance(module, nn.Linear) and in_blocks:
                if getattr(module, "weight", None) is not None:
                    out_f = module.out_features
                    in_f = module.in_features
                    if out_f >= self.MIN_OUT_FEATURES:
                        all_eligible.append((name, module))
                    else:
                        skipped_small.append((name, out_f, in_f))

        if self.score_layer_stride > 1:
            layers = all_eligible[:: self.score_layer_stride]
            skipped_stride = len(all_eligible) - len(layers)
        else:
            layers = all_eligible
            skipped_stride = 0

        if not layers:
            raise RuntimeError("OPUSGhostCollector found no scoreable linear layers")

        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0:
            parts = [f"{len(layers)} layers scored (tensor_sketch_fft)"]
            if skipped_small:
                parts.append(
                    f"{len(skipped_small)} skipped (out < {self.MIN_OUT_FEATURES})"
                )
            if skipped_stride:
                parts.append(
                    f"{skipped_stride} skipped (stride={self.score_layer_stride})"
                )
            print(f"[OPUS] Ghost hooks: {', '.join(parts)}")
        return layers

    def _make_forward_hook(self, name: str):
        """Forward hook: captures input activations for this layer."""

        def hook(module: nn.Module, args: tuple[Any, ...], output: Tensor):
            if not args or not torch.is_tensor(args[0]):
                return
            self._activations_buffer[name] = args[0].detach()

        return hook

    def _make_backward_hook(self, name: str, module: nn.Linear):
        """
        Backward hook using Tensor Sketch (FFT convolution).

        Per the paper (Eq. 24-25):
          - Proxy sketch:     ψ_proxy = Π(1/K Σ_k a_k ⊗ b_k)  [UNPRECONDITIONED]
          - Candidate sketch: ϕ(z) = Π(P · (a_z ⊗ b_z))       [PRECONDITIONED]
          - Alignment:        Σ_l <ϕ^l(z), ψ^l_proxy>

        Uses deferred all-reduce: accumulates local dot products,
        single all_reduce in results().
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

            activations = self._activations_buffer.pop(name)  # (B, T, in_dim)
            grad_out = grad_output[0].detach()  # (B, T, out_dim)

            if activations.dim() == 2:
                activations = activations.unsqueeze(1)
            if grad_out.dim() == 2:
                grad_out = grad_out.unsqueeze(1)

            out_dim, in_dim = module.weight.shape  # type: ignore

            # Split into proxy and candidate portions
            proxy_a = activations[: self.n_proxy]  # [n_proxy, T, in_dim]
            proxy_g = grad_out[: self.n_proxy]  # [n_proxy, T, out_dim]
            cand_a = activations[self.n_proxy :]  # [n_cand, T, in_dim]
            cand_g = grad_out[self.n_proxy :]  # [n_cand, T, out_dim]

            # --- Proxy sketch: UNPRECONDITIONED (paper Eq. 25) ---
            proxy_sketches = self.sketcher.project_linear_batch_fft(
                activations=proxy_a,
                grad_outputs=proxy_g,
                precond_row=None,  # No preconditioner for proxy
                precond_col=None,
                out_dim=out_dim,
                in_dim=in_dim,
                sketch_key=name,
            )  # [n_proxy, sketch_dim]
            proxy_sketch_mean = proxy_sketches.mean(dim=0)  # [sketch_dim]

            # --- Candidate sketch: PRECONDITIONED with factored P ---
            p_row, p_col = self.preconditioner.get_factored(module.weight)  # type: ignore
            cand_sketches = self.sketcher.project_linear_batch_fft(
                activations=cand_a,
                grad_outputs=cand_g,
                precond_row=p_row,
                precond_col=p_col,
                out_dim=out_dim,
                in_dim=in_dim,
                sketch_key=name,
            )  # [n_cand, sketch_dim]

            # --- Alignment: <ϕ(z), ψ_proxy> per candidate ---
            # Using LOCAL proxy only; deferred all_reduce in results()
            layer_alignment = cand_sketches @ proxy_sketch_mean  # [n_cand]

            if self._alignment_scores is None:
                self._alignment_scores = layer_alignment
            else:
                self._alignment_scores = self._alignment_scores + layer_alignment

            # Store candidate sketches for redundancy term in selector
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
        Returns alignment_scores and candidate_sketches.

        Each GPU has DIFFERENT candidates, so alignment scores are LOCAL
        (each GPU's candidates scored against its local proxy). No all_reduce
        needed — the selector does local top-k selection independently per GPU.
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
