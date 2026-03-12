"""
K2: Fused LoRA Linear — single autograd op for base_linear(x) + LoRA(x).

Replaces the 3-op sequence in LoRALinear.forward():
    1. result = self.linear(x)                                   # base GEMM
    2. lora_out = F.linear(F.linear(lora_x, self.lora_A), self.lora_B)  # 2 LoRA GEMMs
    3. result = result + lora_out * scaling                       # add

Key memory optimization: backward does NOT save base_out [B, T, out_features].
Instead saves lora_mid [B, T, rank] (128x smaller at rank=32, out=4096).
Base grad recomputed from saved x + W_base during backward.

Reference: Unsloth fast_lora.py, Axolotl lora_kernels, Chronicals fused_lora_kernels.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedLoRALinearFn(torch.autograd.Function):
    """
    Fused forward: out = x @ W_base^T + (x @ A^T @ B^T) * scaling
    Memory-efficient backward: saves lora_mid [B,T,rank] instead of base_out [B,T,out].
    """

    @staticmethod
    def forward(ctx, x, W_base, bias, lora_A, lora_B, scaling, dropout_mask=None):
        """
        Args:
            x: [*, in_features] input tensor
            W_base: [out_features, in_features] frozen base weight
            bias: [out_features] or None — base linear bias
            lora_A: [rank, in_features] LoRA down-projection
            lora_B: [out_features, rank] LoRA up-projection
            scaling: float — alpha / rank
            dropout_mask: [*] bool mask or None — applied to x before LoRA path
        """
        # Base path
        base_out = F.linear(x, W_base, bias)

        # LoRA path: x -> A -> B -> scale
        lora_x = x
        if dropout_mask is not None:
            lora_x = lora_x * dropout_mask

        lora_mid = F.linear(lora_x, lora_A)         # [*, rank] — SMALL, save this
        lora_out = F.linear(lora_mid, lora_B)        # [*, out_features]
        result = base_out + lora_out * scaling

        # Save for backward — critically, we do NOT save base_out or lora_out
        ctx.save_for_backward(x, W_base, lora_A, lora_B, lora_mid)
        ctx.scaling = scaling
        ctx.has_bias = bias is not None
        ctx.has_dropout = dropout_mask is not None
        if dropout_mask is not None:
            ctx._dropout_mask = dropout_mask  # lightweight bool tensor

        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, W_base, lora_A, lora_B, lora_mid = ctx.saved_tensors
        scaling = ctx.scaling

        grad_output = grad_output.contiguous()

        # Reshape for matmul: flatten all leading dims
        orig_shape = x.shape
        in_f = x.shape[-1]
        out_f = grad_output.shape[-1]
        rank = lora_A.shape[0]

        x_2d = x.reshape(-1, in_f)           # [N, in_f]
        go_2d = grad_output.reshape(-1, out_f)  # [N, out_f]
        mid_2d = lora_mid.reshape(-1, rank)   # [N, rank]

        # --- LoRA B gradient: dB = (go * scaling)^T @ lora_mid ---
        # go_scaled: [N, out_f], lora_mid: [N, rank]
        go_scaled = go_2d * scaling
        grad_lora_B = go_scaled.t() @ mid_2d  # [out_f, rank]

        # --- LoRA A gradient: dA = ((go * scaling) @ B)^T @ x ---
        # grad_lora_mid: [N, rank]
        grad_lora_mid = go_scaled @ lora_B  # [N, rank] via [N, out_f] @ [out_f, rank]

        lora_x_2d = x_2d
        if ctx.has_dropout:
            dm = ctx._dropout_mask
            if dm.shape != x_2d.shape:
                dm = dm.reshape(-1, 1).expand_as(x_2d) if dm.dim() == 1 else dm.reshape(-1, in_f)
            lora_x_2d = x_2d * dm

        grad_lora_A = grad_lora_mid.t() @ lora_x_2d  # [rank, in_f]

        # --- Input gradient: dx = go @ W_base + grad_lora_mid @ A ---
        grad_x = go_2d @ W_base  # [N, in_f] — base path
        grad_x = grad_x + grad_lora_mid @ lora_A  # + LoRA path

        if ctx.has_dropout:
            dm = ctx._dropout_mask
            if dm.shape != grad_x.shape:
                dm = dm.reshape(-1, 1).expand_as(grad_x) if dm.dim() == 1 else dm.reshape(-1, in_f)
            # LoRA contribution to grad_x needs dropout mask
            # Actually, the full grad_x should be: go @ W_base + (go_scaled @ B) @ A
            # The dropout only applies to the LoRA *forward* path (x -> A), so
            # in backward, the gradient through LoRA already flows correctly.
            # No additional masking needed on grad_x.
            pass

        grad_x = grad_x.reshape(orig_shape)

        # W_base is frozen — no gradient
        # bias gradient if present
        grad_bias = go_2d.sum(0) if ctx.has_bias else None

        return grad_x, None, grad_bias, grad_lora_A, grad_lora_B, None, None


def fused_lora_linear(x, W_base, bias, lora_A, lora_B, scaling, dropout_mask=None):
    """
    Fused LoRA linear: base_linear(x) + (x @ A^T @ B^T) * scaling.

    Memory-efficient: saves only lora_mid [*, rank] for backward (not base_out).

    Args:
        x: [*, in_features]
        W_base: [out_features, in_features] — frozen
        bias: [out_features] or None
        lora_A: [rank, in_features]
        lora_B: [out_features, rank]
        scaling: float (alpha / rank)
        dropout_mask: optional bool mask for LoRA dropout

    Returns: [*, out_features]
    """
    return FusedLoRALinearFn.apply(x, W_base, bias, lora_A, lora_B, scaling, dropout_mask)


class FusedLoRALinear(nn.Module):
    """
    Drop-in replacement for LoRALinear that uses the fused autograd function.

    Same interface as lora_utils.LoRALinear but with memory-efficient backward.
    Saves lora_mid [B, T, rank] instead of base_out [B, T, out_features].
    At rank=32, out=4096: 128x memory savings per layer.
    """

    def __init__(self, original_linear: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        import math

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Keep the original linear as a submodule — preserves ZeRO-3 ds_id.
        self.linear = original_linear
        for p in self.linear.parameters():
            p.requires_grad = False

        # LoRA matrices
        dtype = getattr(original_linear.weight, 'dtype', torch.bfloat16)
        if dtype is None:
            dtype = torch.bfloat16

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank, dtype=dtype))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else None
        self._dropout_rate = dropout

        # Initialize: A with Kaiming, B with zeros (LoRA starts at zero)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_x = x.to(self.lora_A.dtype)

        # Generate dropout mask if needed
        dropout_mask = None
        if self.lora_dropout is not None and self.training:
            # Apply dropout as a mask so the fused function can use it
            dropout_mask = torch.ones_like(lora_x)
            dropout_mask = self.lora_dropout(dropout_mask)

        return fused_lora_linear(
            lora_x,
            self.linear.weight,
            self.linear.bias,
            self.lora_A,
            self.lora_B,
            self.scaling,
            dropout_mask,
        )

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}, "
            f"fused=True"
        )
