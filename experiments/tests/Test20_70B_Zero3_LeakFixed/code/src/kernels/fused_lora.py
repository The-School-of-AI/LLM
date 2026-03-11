"""
Fused LoRA kernel — Unsloth-style memory-efficient LoRA.

Inspired by Unsloth's approach:
- Forward: Compute x @ A^T @ B^T * scaling in two matmuls
- Backward: Recompute intermediate (x @ A^T) instead of saving it
- Memory savings: No intermediate storage [B*T, rank]

At 70B scale with rank=16, seq=4096, batch=32:
- Intermediate size per LoRA layer: 32*4096*16*2 bytes = 4 MB (bf16)
- With ~80 LoRA layers (q/k/v/o × 20 layers): ~320 MB saved per GPU

Attribution: Inspired by Unsloth (Apache-2.0)
Repository: https://github.com/unslothai/unsloth
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: Triton for fused matmuls (not critical at 70B - cuBLAS is fast enough)
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# Note: Profiling removed from this module to avoid OOM.
# With 84 LoRA adapters × 7 kernel_region() calls = 588 CUDA events per step,
# the profiler's event tracking exhausted memory during backward pass.
# LoRA uses standard F.linear (cuBLAS), not custom kernels, so profiling here
# adds little value. Profile custom kernels instead (sparse_attn, sinkhorn, etc).


class FusedLoRAFunction(torch.autograd.Function):
    """
    Memory-efficient LoRA forward/backward.

    Forward:
        out = x @ A^T @ B^T * scaling

    Backward:
        - Saves: x, A, B (not the intermediate)
        - Recomputes: intermediate = x @ A^T during backward
        - Computes: d_x, d_A, d_B from recomputed intermediate

    Memory savings: [B*T, rank] intermediate tensor (typically 4-8 MB per layer)
    """

    @staticmethod
    def forward(ctx, x, lora_A, lora_B, scaling):
        """
        Args:
            x: [B*T, in_features]
            lora_A: [rank, in_features]
            lora_B: [out_features, rank]
            scaling: float (alpha / rank)

        Returns:
            out: [B*T, out_features]
        """
        # Two-step matmul: x @ A^T @ B^T
        # intermediate shape: [B*T, rank]
        intermediate = F.linear(x, lora_A)  # x @ A^T
        out = F.linear(intermediate, lora_B) * scaling  # intermediate @ B^T * scaling

        # Save for backward (no intermediate!)
        ctx.save_for_backward(x, lora_A, lora_B)
        ctx.scaling = scaling

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            grad_output: [..., out_features] - may have batch dims

        Returns:
            grad_x, grad_lora_A, grad_lora_B, None
        """
        x, lora_A, lora_B = ctx.saved_tensors
        scaling = ctx.scaling

        # Save original shape and flatten to 2D for matmul
        original_shape = grad_output.shape
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        x_2d = x.reshape(-1, x.shape[-1])

        # Recompute intermediate (this is the key memory-saving trick)
        with torch.no_grad():
            intermediate = F.linear(x_2d, lora_A)  # [N, rank]

        # Gradients w.r.t. B: grad_B = grad_out^T @ intermediate * scaling
        # lora_B is [out_features, rank], grad_output is [N, out_features], intermediate is [N, rank]
        grad_lora_B = grad_output_2d.t().mm(intermediate) * scaling

        # Gradients w.r.t. intermediate: grad_intermediate = grad_out @ B
        # Forward: out = intermediate @ B^T, so backward: grad_int = grad_out @ B (no transpose!)
        grad_intermediate = grad_output_2d.mm(lora_B) * scaling

        # Gradients w.r.t. A: grad_A = grad_intermediate^T @ x
        # lora_A is [rank, in_features], grad_intermediate is [N, rank], x is [N, in_features]
        grad_lora_A = grad_intermediate.t().mm(x_2d)

        # Gradients w.r.t. x: grad_x = grad_intermediate @ A
        # Forward: intermediate = x @ A^T, so backward: grad_x = grad_int @ A (no transpose!)
        grad_x = grad_intermediate.mm(lora_A)

        # Reshape grad_x back to original shape
        grad_x = grad_x.reshape(original_shape)

        return grad_x, grad_lora_A, grad_lora_B, None


def fused_lora_forward(
    x: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """
    Memory-efficient fused LoRA forward + backward.

    Args:
        x: Input tensor [B*T, in_features]
        lora_A: Low-rank adapter A [rank, in_features]
        lora_B: Low-rank adapter B [out_features, rank]
        scaling: Scaling factor (alpha / rank)

    Returns:
        out: [B*T, out_features]
    """
    return FusedLoRAFunction.apply(x, lora_A, lora_B, scaling)


class FusedLoRALinear(nn.Module):
    """
    Fused LoRA linear layer with recompute-in-backward.

    Drop-in replacement for standard LoRALinear with better memory efficiency.

    Usage:
        layer = FusedLoRALinear(original_linear, rank=16, alpha=32.0)
        out = layer(x)  # Same interface as LoRALinear
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_fused: bool = True,
    ):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.use_fused = use_fused

        # Keep the original linear as a submodule (frozen)
        self.linear = original_linear
        for p in self.linear.parameters():
            p.requires_grad = False

        # LoRA matrices: A (rank x in_features), B (out_features x rank)
        dtype = getattr(original_linear.weight, 'dtype', torch.bfloat16)
        if dtype is None:
            dtype = torch.bfloat16

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank, dtype=dtype))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize: A with Kaiming, B with zeros (LoRA starts at zero)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self):
        """Expose base linear weight for compatibility."""
        return self.linear.weight

    @property
    def bias(self):
        """Expose base linear bias for compatibility."""
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen base path
        result = self.linear(x)

        # LoRA path (fused or standard)
        lora_x = self.lora_dropout(x)
        lora_x = lora_x.to(self.lora_A.dtype)

        if self.use_fused:
            # Fused path: recompute intermediate in backward
            lora_out = fused_lora_forward(lora_x, self.lora_A, self.lora_B, self.scaling)
        else:
            # Standard path: saves intermediate (higher memory)
            lora_out = F.linear(F.linear(lora_x, self.lora_A), self.lora_B) * self.scaling

        result = result + lora_out
        return result

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}, "
            f"fused={self.use_fused}"
        )


# Benchmark utility (optional)
def benchmark_fused_vs_standard():
    """
    Benchmark fused vs standard LoRA on realistic 70B shapes.

    Run with:
        python -c "from src.kernels.fused_lora import benchmark_fused_vs_standard; benchmark_fused_vs_standard()"
    """
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # 70B-like dimensions
    batch_size = 32
    seq_len = 4096
    d_model = 4096
    rank = 16
    alpha = 32.0

    BT = batch_size * seq_len  # 131072

    # Create dummy linear layer
    linear = nn.Linear(d_model, d_model, bias=False, dtype=dtype, device=device)

    # Standard LoRA
    print("=" * 60)
    print("Benchmarking LoRA: Standard vs Fused")
    print("=" * 60)
    print(f"Shape: [B={batch_size}, T={seq_len}, D={d_model}], rank={rank}")
    print(f"Memory (intermediate): {BT * rank * 2 / 1024**2:.2f} MB")
    print()

    # Standard LoRA
    lora_A_std = nn.Parameter(torch.randn(rank, d_model, dtype=dtype, device=device))
    lora_B_std = nn.Parameter(torch.randn(d_model, rank, dtype=dtype, device=device))
    scaling = alpha / rank

    x = torch.randn(BT, d_model, dtype=dtype, device=device, requires_grad=True)

    # Warmup
    for _ in range(5):
        out_std = F.linear(F.linear(x, lora_A_std), lora_B_std) * scaling
        out_std.sum().backward()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark standard
    n_iters = 20
    start = time.perf_counter()
    for _ in range(n_iters):
        x.grad = None
        lora_A_std.grad = None
        lora_B_std.grad = None
        out_std = F.linear(F.linear(x, lora_A_std), lora_B_std) * scaling
        out_std.sum().backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    std_time = (time.perf_counter() - start) / n_iters * 1000

    print(f"Standard LoRA:  {std_time:.3f} ms/iter")

    # Fused LoRA
    lora_A_fused = nn.Parameter(lora_A_std.data.clone())
    lora_B_fused = nn.Parameter(lora_B_std.data.clone())

    # Warmup
    for _ in range(5):
        out_fused = fused_lora_forward(x, lora_A_fused, lora_B_fused, scaling)
        out_fused.sum().backward()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark fused
    start = time.perf_counter()
    for _ in range(n_iters):
        x.grad = None
        lora_A_fused.grad = None
        lora_B_fused.grad = None
        out_fused = fused_lora_forward(x, lora_A_fused, lora_B_fused, scaling)
        out_fused.sum().backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / n_iters * 1000

    print(f"Fused LoRA:     {fused_time:.3f} ms/iter")
    print()
    print(f"Speedup: {std_time / fused_time:.2f}×")
    print(f"Note: Fused LoRA saves {BT * rank * 2 / 1024**2:.2f} MB intermediate storage")
    print("=" * 60)


if __name__ == "__main__":
    benchmark_fused_vs_standard()
