"""
Reversible Midpoint Integration Stack
=======================================

Memory-efficient reversible integration for the transformer layer stack.

Instead of storing activations for all layers during backprop, this uses
the midpoint ODE solver with reversible computation to recompute activations
during the backward pass.

Each layer must implement a force(x) method:
    delta, aux_loss = layer.force(x)  # delta = F(x) - x

The midpoint method:
    x_{n+1} = x_n + step_size * f(x_n + step_size/2 * f(x_n))

With euler bootstrap for the first step:
    x_1 = x_0 + step_size * f(x_0)

Reference: Test_Code/model_1b.py lines 1399-1407
"""

from contextlib import nullcontext
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

try:
    import torch._dynamo as _dynamo

    _dynamo_disable = _dynamo.disable
except Exception:

    def _dynamo_disable(fn=None, recursive=True):
        if fn is None:

            def decorator(f):
                return f

            return decorator
        return fn


class ReversibleMidpointStack(nn.Module):
    """
    Reversible Midpoint Integration Stack.

    Wraps a list of transformer layers and applies them using the midpoint
    ODE integration method. Uses gradient checkpointing for memory efficiency.

    Each layer must implement:
        force(x) -> (delta, aux_loss)
    where delta = layer_output - layer_input.

    Args:
        layers: List of nn.Module with force() method
        step_size: Integration step size (default 0.25)
        a: Blending coefficient for midpoint (default 0.5)
        noise_eps: Noise injection for training stability (default 0.0)
        bootstrap: Bootstrap method ("euler" for Euler first step)
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        step_size: float = 0.25,
        a: float = 0.5,
        noise_eps: float = 0.0,
        bootstrap: str = "euler",
        use_checkpoint: bool = True,
        checkpoint_autocast_enabled: bool = False,
        checkpoint_autocast_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.layers = layers
        self.step_size = step_size
        self.a = a
        self.noise_eps = noise_eps
        self.bootstrap = bootstrap
        self.use_checkpoint = use_checkpoint
        self.checkpoint_autocast_enabled = checkpoint_autocast_enabled
        self.checkpoint_autocast_dtype = checkpoint_autocast_dtype

    def _autocast_context(self, x: torch.Tensor):
        if not self.checkpoint_autocast_enabled:
            return nullcontext()
        if x.device.type not in {"cuda", "mps"}:
            return nullcontext()
        return torch.autocast(
            device_type=x.device.type,
            enabled=True,
            dtype=self.checkpoint_autocast_dtype,
        )

    def _euler_step(self, x, layer):
        """Single Euler step: x_{n+1} = x_n + h * f(x_n)"""
        with self._autocast_context(x):
            delta, aux = layer.force(x)
        x_new = x + self.step_size * delta
        return x_new, aux

    def _midpoint_step(self, x, layer):
        """
        Single midpoint step:
            x_mid = x_n + (h/2) * f(x_n)
            x_{n+1} = x_n + h * f(x_mid)

        With blending coefficient a:
            x_{n+1} = x_n + h * ((1-a)*f(x_n) + a*f(x_mid))
        """
        with self._autocast_context(x):
            # Evaluate at current point
            delta_n, aux_n = layer.force(x)

            # Midpoint evaluation
            x_mid = x + (self.step_size / 2.0) * delta_n
            delta_mid, aux_mid = layer.force(x_mid)

        # Blended update
        blended_delta = (1.0 - self.a) * delta_n + self.a * delta_mid
        x_new = x + self.step_size * blended_delta

        # Combine auxiliary losses
        aux = aux_n + aux_mid
        return x_new, aux

    def _checkpointed_euler_step(self, x, layer):
        """Euler step with gradient checkpointing."""

        def step_fn(x_in):
            with self._autocast_context(x_in):
                delta, aux = layer.force(x_in)
            x_out = x_in + self.step_size * delta
            return x_out, aux

        if self.training and self.use_checkpoint:
            # IMPORTANT: use_reentrant=True keeps the original checkpoint
            # semantics (forward under no_grad, backward-time recompute),
            # which this reversible stack + ReferenceGSA path relies on.
            x_new, aux = checkpoint(step_fn, x, use_reentrant=True)
        else:
            x_new, aux = step_fn(x)
        return x_new, aux

    def _checkpointed_midpoint_step(self, x, layer):
        """Midpoint step with gradient checkpointing."""

        def step_fn(x_in):
            with self._autocast_context(x_in):
                delta_n, aux_n = layer.force(x_in)
                x_mid = x_in + (self.step_size / 2.0) * delta_n
                delta_mid, aux_mid = layer.force(x_mid)
            blended_delta = (1.0 - self.a) * delta_n + self.a * delta_mid
            x_out = x_in + self.step_size * blended_delta
            aux = aux_n + aux_mid
            return x_out, aux

        if self.training and self.use_checkpoint:
            x_new, aux = checkpoint(step_fn, x, use_reentrant=True)
        else:
            x_new, aux = step_fn(x)
        return x_new, aux

    @_dynamo_disable
    def forward(self, x_stream: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all layers using midpoint integration.

        Args:
            x_stream: Input tensor (B, T, n_streams, D)

        Returns:
            x_stream: Output tensor (B, T, n_streams, D)
            total_aux_loss: Accumulated auxiliary loss from all layers
        """
        total_aux = x_stream.new_zeros((), dtype=torch.float32)

        for i, layer in enumerate(self.layers):
            # First layer uses euler bootstrap, rest use midpoint
            if i == 0 and self.bootstrap == "euler":
                x_stream, aux = self._checkpointed_euler_step(x_stream, layer)
            else:
                x_stream, aux = self._checkpointed_midpoint_step(x_stream, layer)

            total_aux = total_aux + aux

            # Optional noise injection for training stability
            if self.training and self.noise_eps > 0:
                noise = torch.randn_like(x_stream) * self.noise_eps
                x_stream = x_stream + noise

        return x_stream, total_aux
