"""
Fused QKVG Projection for DeltaNet - Unsloth-style optimization.

Optimization: Instead of 4 separate matmuls:
    q = x @ W_q
    k = x @ W_k
    v = x @ W_v
    g = x @ W_g

Do a single fused matmul:
    qkvg = x @ W_qkvg  (where W_qkvg = [W_q | W_k | W_v | W_g] concatenated)

Benefits:
- Single kernel launch instead of 4
- Better memory coalescing
- Expected speedup: 10-15% on projection time (~50-70ms per step at 70B)

At 70B scale:
- Current: 4 matmuls × 11.5ms = 46ms
- Fused: 1 matmul × ~40ms expected
- Savings: ~6ms per call × 145 calls = ~870ms total per step
  BUT most time is in actual compute, so realistic: ~50-70ms savings

Attribution: Inspired by Unsloth's fused attention projections
Repository: https://github.com/unslothai/unsloth
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedQKVGProjection(nn.Module):
    """
    Fused Q/K/V/G projection for DeltaNet.

    Combines 4 separate linear layers into a single fused matmul for efficiency.

    Usage:
        # Replace:
        # q = q_proj(x)
        # k = k_proj(x)
        # v = v_proj(x)
        # g = g_proj(x)

        # With:
        # fused_proj = FusedQKVGProjection(hidden_size, key_dim, value_dim)
        # q, k, v, g = fused_proj(x)
    """

    def __init__(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.key_dim = key_dim
        self.value_dim = value_dim

        # Single fused weight matrix [hidden_size, key_dim + key_dim + value_dim + value_dim]
        total_out_dim = key_dim + key_dim + value_dim + value_dim
        self.qkvg_weight = nn.Parameter(torch.empty(total_out_dim, hidden_size, dtype=dtype))

        if bias:
            self.qkvg_bias = nn.Parameter(torch.empty(total_out_dim, dtype=dtype))
        else:
            self.register_parameter('qkvg_bias', None)

        self._init_weights()

    def _init_weights(self):
        """Initialize using same scheme as separate projections."""
        # Standard initialization for attention projections
        nn.init.normal_(self.qkvg_weight, mean=0.0, std=0.02)
        if self.qkvg_bias is not None:
            nn.init.zeros_(self.qkvg_bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, hidden_size]

        Returns:
            q: [B, T, key_dim]
            k: [B, T, key_dim]
            v: [B, T, value_dim]
            g: [B, T, value_dim]
        """
        # Single fused matmul
        qkvg = F.linear(x, self.qkvg_weight, self.qkvg_bias)

        # Split into Q, K, V, G
        q, k, v, g = torch.split(
            qkvg,
            [self.key_dim, self.key_dim, self.value_dim, self.value_dim],
            dim=-1
        )

        return q, k, v, g

    @staticmethod
    def from_separate_projections(
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        g_proj: nn.Linear,
    ) -> 'FusedQKVGProjection':
        """
        Create a fused projection from existing separate projections.

        This is useful for converting a pre-trained model to use fused projections.

        Args:
            q_proj, k_proj, v_proj, g_proj: Existing separate projection layers

        Returns:
            FusedQKVGProjection with weights copied from separate projections
        """
        hidden_size = q_proj.in_features
        key_dim = q_proj.out_features
        value_dim = v_proj.out_features
        bias = q_proj.bias is not None
        dtype = q_proj.weight.dtype

        fused = FusedQKVGProjection(hidden_size, key_dim, value_dim, bias=bias, dtype=dtype)

        # Copy weights by concatenating
        with torch.no_grad():
            fused.qkvg_weight.copy_(torch.cat([
                q_proj.weight,
                k_proj.weight,
                v_proj.weight,
                g_proj.weight,
            ], dim=0))

            if bias:
                fused.qkvg_bias.copy_(torch.cat([
                    q_proj.bias,
                    k_proj.bias,
                    v_proj.bias,
                    g_proj.bias,
                ], dim=0))

        return fused

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"key_dim={self.key_dim}, "
            f"value_dim={self.value_dim}, "
            f"bias={self.qkvg_bias is not None}"
        )


def benchmark_fused_vs_separate():
    """
    Benchmark fused QKVG vs separate projections.

    Run with:
        python -c "from src.kernels.fused_qkvg_proj import benchmark_fused_vs_separate; benchmark_fused_vs_separate()"
    """
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # 70B-like dimensions
    batch_size = 32
    seq_len = 4096
    hidden_size = 4096
    num_heads = 32
    head_dim = 128
    key_dim = num_heads * head_dim
    value_dim = num_heads * head_dim

    print("=" * 70)
    print("Benchmarking QKVG Projections: Separate vs Fused")
    print("=" * 70)
    print(f"Shape: [B={batch_size}, T={seq_len}, H={hidden_size}]")
    print(f"Projections: Q/K={key_dim}, V/G={value_dim}")
    print()

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)

    # Separate projections
    q_proj = nn.Linear(hidden_size, key_dim, bias=False, dtype=dtype, device=device)
    k_proj = nn.Linear(hidden_size, key_dim, bias=False, dtype=dtype, device=device)
    v_proj = nn.Linear(hidden_size, value_dim, bias=False, dtype=dtype, device=device)
    g_proj = nn.Linear(hidden_size, value_dim, bias=False, dtype=dtype, device=device)

    # Warmup
    for _ in range(10):
        q = q_proj(x)
        k = k_proj(x)
        v = v_proj(x)
        g = g_proj(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark separate
    n_iters = 100
    start = time.perf_counter()
    for _ in range(n_iters):
        q = q_proj(x)
        k = k_proj(x)
        v = v_proj(x)
        g = g_proj(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    separate_time = (time.perf_counter() - start) / n_iters * 1000

    print(f"Separate projections:  {separate_time:.3f} ms/iter")

    # Fused projection
    fused_proj = FusedQKVGProjection.from_separate_projections(
        q_proj, k_proj, v_proj, g_proj
    )

    # Warmup
    for _ in range(10):
        q_f, k_f, v_f, g_f = fused_proj(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark fused
    start = time.perf_counter()
    for _ in range(n_iters):
        q_f, k_f, v_f, g_f = fused_proj(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / n_iters * 1000

    print(f"Fused projection:      {fused_time:.3f} ms/iter")
    print()
    print(f"Speedup: {separate_time / fused_time:.2f}×")
    print(f"Savings: {separate_time - fused_time:.3f} ms per call")
    print()

    # Verify correctness
    q_f, k_f, v_f, g_f = fused_proj(x)
    q = q_proj(x)
    k = k_proj(x)
    v = v_proj(x)
    g = g_proj(x)

    print("Correctness check:")
    print(f"  Q max diff: {(q_f - q).abs().max().item():.6e}")
    print(f"  K max diff: {(k_f - k).abs().max().item():.6e}")
    print(f"  V max diff: {(v_f - v).abs().max().item():.6e}")
    print(f"  G max diff: {(g_f - g).abs().max().item():.6e}")
    print("=" * 70)


if __name__ == "__main__":
    benchmark_fused_vs_separate()
