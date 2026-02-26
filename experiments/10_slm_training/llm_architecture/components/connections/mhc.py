"""
Manifold-Constrained Hyper-Connections (mHC)
=============================================

CORRECT Implementation based on DeepSeek paper: arXiv:2512.24880v2

Key Concepts:
1. Expand residual stream width from C to n×C (n=4 typically)
2. Use TINY learnable matrices to control information flow:
   - H_pre ∈ R^{1×n}: Aggregates n streams → 1 for layer input
   - H_post ∈ R^{1×n}: Distributes layer output → n streams  
   - H_res ∈ R^{n×n}: Mixes streams (doubly stochastic)
3. Constrain H_res via Sinkhorn-Knopp to preserve signal stability

Parameter overhead per mHC module: ~200K (NOT millions!)
- φ_pre: nC × n = 4 × 2048 × 4 = 32,768
- φ_post: nC × n = 32,768
- φ_res: nC × n² = 4 × 2048 × 16 = 131,072
- Total: ~196K per mHC module

This is fundamentally different from expanding hidden dimensions!
"""

from typing import Tuple

import torch
import torch.nn as nn

# Use Triton-fused Sinkhorn when available
try:
    from components.kernels.triton_sinkhorn import HAS_TRITON as HAS_TRITON_SINKHORN
    from components.kernels.triton_sinkhorn import triton_sinkhorn_knopp

    _USE_TRITON_SINKHORN = HAS_TRITON_SINKHORN
except ImportError:
    _USE_TRITON_SINKHORN = False


def _sinkhorn_knopp_pytorch(
    H: torch.Tensor, num_iters: int = 20, eps: float = 1e-8
) -> torch.Tensor:
    """
    PyTorch Sinkhorn-Knopp (torch.compile compatible).

    Projects matrices onto Birkhoff polytope (doubly stochastic).
    Used when Triton is unavailable or input is not on CUDA.

    Note: Removed @torch.jit.script as it causes graph breaks with torch.compile.
    torch.compile's inductor backend already optimizes this loop effectively.

    Args:
        H: Input matrix [..., n, n] (will be exponentiated)
        num_iters: Number of iterations (paper uses 20)
        eps: Numerical stability constant

    Returns:
        Doubly stochastic matrix [..., n, n]
    """
    M = torch.exp(H)
    for _ in range(num_iters):
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
    return M


def sinkhorn_knopp(
    H: torch.Tensor, num_iters: int = 20, eps: float = 1e-8
) -> torch.Tensor:
    """
    Sinkhorn-Knopp algorithm for projecting to doubly stochastic matrices.

    Uses fused Triton kernel on CUDA (single launch for all iterations),
    falls back to JIT-compiled PyTorch otherwise.

    A doubly stochastic matrix has:
    - All elements >= 0
    - All rows sum to 1
    - All columns sum to 1

    This ensures stable signal propagation (no explosion/vanishing).

    Args:
        H: Input matrix [..., n, n] (will be exponentiated)
        num_iters: Number of iterations (paper uses 20)
        eps: Numerical stability constant

    Returns:
        Doubly stochastic matrix [..., n, n]
    """
    if _USE_TRITON_SINKHORN and H.is_cuda:
        try:
            return triton_sinkhorn_knopp(H, num_iters, eps)
        except Exception:
            pass
    return _sinkhorn_knopp_pytorch(H, num_iters, eps)


class mHCMapping(nn.Module):
    """
    Computes the three learnable mappings for mHC.

    From Eq. 7 in the paper:
    - x̄_l = vec(x_l) ∈ R^{1×nC}  (flatten input)
    - x̄'_l = RMSNorm(x̄_l)
    - H̃_pre = α_pre · (x̄'_l @ φ_pre) + b_pre
    - H̃_post = α_post · (x̄'_l @ φ_post) + b_post
    - H̃_res = α_res · mat(x̄'_l @ φ_res) + b_res

    From Eq. 8:
    - H_pre = σ(H̃_pre)
    - H_post = 2σ(H̃_post)
    - H_res = Sinkhorn-Knopp(H̃_res)
    """

    def __init__(
        self,
        hidden_size: int,
        expansion_rate: int = 4,
        alpha_init: float = 0.01,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.hidden_size = hidden_size  # C
        self.n = int(expansion_rate)  # n (typically 4)
        self.sinkhorn_iters = sinkhorn_iters

        # Flattened dimension: n * C
        self.flat_dim = self.n * hidden_size

        # Linear projections for dynamic mappings (Eq. 7)
        # φ_pre, φ_post ∈ R^{nC × n}
        self.phi_pre = nn.Linear(self.flat_dim, self.n, bias=False)
        self.phi_post = nn.Linear(self.flat_dim, self.n, bias=False)
        # φ_res ∈ R^{nC × n²}
        self.phi_res = nn.Linear(self.flat_dim, self.n * self.n, bias=False)

        # Static biases
        # b_pre, b_post ∈ R^{1×n}
        self.b_pre = nn.Parameter(torch.zeros(1, self.n))
        self.b_post = nn.Parameter(torch.zeros(1, self.n))
        # b_res ∈ R^{n×n}, initialized to identity-like for stable start
        self.b_res = nn.Parameter(torch.eye(self.n) * 2.0)  # Diagonal dominant

        # Learnable gating factors (initialized small, α=0.01 per paper)
        self.alpha_pre = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_post = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_res = nn.Parameter(torch.tensor(alpha_init))

        # RMSNorm weight
        self.rms_weight = nn.Parameter(torch.ones(self.flat_dim))
        self.rms_eps = 1e-6

        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights with small values."""
        nn.init.normal_(self.phi_pre.weight, std=0.02)
        nn.init.normal_(self.phi_post.weight, std=0.02)
        nn.init.normal_(self.phi_res.weight, std=0.02)

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm to flattened input."""
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.rms_eps)
        return x_normed * self.rms_weight

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute H_pre, H_post, H_res from input hidden states.

        Args:
            x: Hidden states [batch, seq_len, n, hidden_size]

        Returns:
            H_pre: [batch, seq_len, 1, n] - aggregation weights (non-negative)
            H_post: [batch, seq_len, 1, n] - distribution weights (non-negative)
            H_res: [batch, seq_len, n, n] - mixing matrix (doubly stochastic)
        """
        batch_size, seq_len, n, hidden_size = x.shape

        # Flatten: [batch, seq, n, C] -> [batch, seq, n*C]
        x_flat = x.reshape(batch_size, seq_len, -1)

        # RMSNorm on flattened input
        x_normed = self._rms_norm(x_flat)

        # Dynamic mappings via linear projections
        # [batch, seq, n*C] @ [n*C, n] -> [batch, seq, n]
        H_pre_dynamic = self.phi_pre(x_normed)
        H_post_dynamic = self.phi_post(x_normed)
        # [batch, seq, n*C] @ [n*C, n²] -> [batch, seq, n²]
        H_res_dynamic = self.phi_res(x_normed)

        # Combine dynamic + static with gating (Eq. 7)
        H_pre_raw = self.alpha_pre * H_pre_dynamic + self.b_pre
        H_post_raw = self.alpha_post * H_post_dynamic + self.b_post
        H_res_raw = (
            self.alpha_res * H_res_dynamic.view(batch_size, seq_len, self.n, self.n)
            + self.b_res
        )

        # Apply constraints (Eq. 8)
        # H_pre = σ(H̃_pre) - sigmoid for non-negativity
        H_pre = torch.sigmoid(H_pre_raw).unsqueeze(2)  # [batch, seq, 1, n]
        # H_post = 2σ(H̃_post) - scaled sigmoid
        H_post = 2 * torch.sigmoid(H_post_raw).unsqueeze(2)  # [batch, seq, 1, n]
        # H_res = Sinkhorn-Knopp(H̃_res) - doubly stochastic
        H_res = sinkhorn_knopp(H_res_raw, self.sinkhorn_iters)  # [batch, seq, n, n]

        return H_pre, H_post, H_res


class ManifoldConstrainedHyperConnection(nn.Module):
    """
    Manifold-Constrained Hyper-Connections (mHC).

    Replaces standard residual connection with multi-stream architecture:

    Standard residual: x_{l+1} = x_l + F(x_l)

    mHC (Eq. 3): x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)

    Where:
    - x_l ∈ R^{n×C} is the expanded residual stream (n=4 streams)
    - H_pre ∈ R^{1×n} aggregates streams for layer input
    - H_post ∈ R^{1×n} distributes layer output to streams
    - H_res ∈ R^{n×n} mixes streams (doubly stochastic for stability)

    Key properties of doubly stochastic H_res:
    1. Norm preservation: ||H_res||_2 ≤ 1 (prevents gradient explosion)
    2. Compositional closure: product of doubly stochastic is doubly stochastic
    3. Signal conservation: row/column sums = 1 (mean preserving)
    """

    def __init__(
        self,
        hidden_size: int,
        expansion_rate: int = 4,
        alpha_init: float = 0.01,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n = int(expansion_rate)

        # Mapping module computes H_pre, H_post, H_res
        self.mapping = mHCMapping(
            hidden_size=hidden_size,
            expansion_rate=expansion_rate,
            alpha_init=alpha_init,
            sinkhorn_iters=sinkhorn_iters,
        )

    def expand_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand input from [batch, seq, C] to [batch, seq, n, C].

        Initial expansion: replicate input across n streams.
        """
        return x.unsqueeze(2).expand(-1, -1, self.n, -1).contiguous()

    def collapse_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Collapse output from [batch, seq, n, C] to [batch, seq, C].

        Final aggregation: mean across streams.
        """
        return x.mean(dim=2)

    def get_layer_input(self, x: torch.Tensor, H_pre: torch.Tensor) -> torch.Tensor:
        """
        Aggregate n streams into single layer input.

        h_in = H_pre @ x_l (from Eq. 3)

        Args:
            x: Multi-stream state [batch, seq, n, C]
            H_pre: Aggregation weights [batch, seq, 1, n]

        Returns:
            Layer input [batch, seq, C]
        """
        # H_pre: [batch, seq, 1, n], x: [batch, seq, n, C]
        # Result: [batch, seq, 1, C] -> squeeze -> [batch, seq, C]
        return torch.matmul(H_pre, x).squeeze(2)

    def apply_residual(
        self,
        x: torch.Tensor,
        layer_output: torch.Tensor,
        H_res: torch.Tensor,
        H_post: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply mHC residual connection.

        x_{l+1} = H_res @ x_l + H_post^T @ F(·) (Eq. 3)

        Args:
            x: Current multi-stream state [batch, seq, n, C]
            layer_output: Output from layer F [batch, seq, C]
            H_res: Mixing matrix [batch, seq, n, n] (doubly stochastic)
            H_post: Distribution weights [batch, seq, 1, n]

        Returns:
            New multi-stream state [batch, seq, n, C]
        """
        # H_res @ x: Mix streams
        # [batch, seq, n, n] @ [batch, seq, n, C] -> [batch, seq, n, C]
        mixed = torch.matmul(H_res, x)

        # H_post^T @ layer_output: Distribute output to streams
        # H_post: [batch, seq, 1, n] -> transpose -> [batch, seq, n, 1]
        # layer_output: [batch, seq, C] -> [batch, seq, 1, C]
        # Result: [batch, seq, n, 1] @ [batch, seq, 1, C] -> [batch, seq, n, C]
        H_post_T = H_post.transpose(-2, -1)  # [batch, seq, n, 1]
        layer_out_expanded = layer_output.unsqueeze(2)  # [batch, seq, 1, C]
        distributed = torch.matmul(H_post_T, layer_out_expanded)  # [batch, seq, n, C]

        # Combine
        return mixed + distributed

    def forward(self, x: torch.Tensor, layer_output: torch.Tensor) -> torch.Tensor:
        """
        Full mHC forward pass.

        Args:
            x: Multi-stream state [batch, seq, n, C]
            layer_output: Output from layer F [batch, seq, C]

        Returns:
            Updated multi-stream state [batch, seq, n, C]
        """
        H_pre, H_post, H_res = self.mapping(x)
        return self.apply_residual(x, layer_output, H_res, H_post)

    def get_aggregated_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        Get aggregated input for layer and cache mappings for later use.

        Args:
            x: Multi-stream state [batch, seq, n, C]

        Returns:
            layer_input: Aggregated input [batch, seq, C]
            cache: (H_pre, H_post, H_res) for apply_cached
        """
        H_pre, H_post, H_res = self.mapping(x)
        layer_input = self.get_layer_input(x, H_pre)
        return layer_input, (H_pre, H_post, H_res)

    def apply_cached(
        self,
        x: torch.Tensor,
        layer_output: torch.Tensor,
        cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply residual with cached mappings (avoids recomputing).

        Args:
            x: Multi-stream state [batch, seq, n, C]
            layer_output: Output from layer [batch, seq, C]
            cache: (H_pre, H_post, H_res) from get_aggregated_input

        Returns:
            Updated multi-stream state [batch, seq, n, C]
        """
        H_pre, H_post, H_res = cache
        return self.apply_residual(x, layer_output, H_res, H_post)


class RMSNorm(nn.Module):
    """RMSNorm layer."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


# =============================================================================
# Standard residual for comparison
# =============================================================================


class ResidualConnection(nn.Module):
    """Standard residual connection: x + F(x)"""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(sublayer_output)


# =============================================================================
# Per-sublayer mHC module (used by MHCTransformerBlock)
# =============================================================================


class MHCSublayerConnection(nn.Module):
    """
    Single mHC connection for one sublayer (attention or FFN).

    Implements the paper's Eq. 3 on persistent n-stream state:
        layer_input = H_pre @ x_l          (aggregate n streams -> C)
        x_{l+1} = H_res @ x_l + H_post^T @ F(layer_input)

    The n-stream state [B, S, n, C] persists across layers —
    this module never expands or collapses the stream.
    """

    def __init__(
        self,
        hidden_size: int,
        expansion_rate: int = 4,
        alpha_init: float = 0.01,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.mhc = ManifoldConstrainedHyperConnection(
            hidden_size=hidden_size,
            expansion_rate=expansion_rate,
            alpha_init=alpha_init,
            sinkhorn_iters=sinkhorn_iters,
        )

    def get_layer_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        Aggregate n-stream state into single-stream layer input.

        Args:
            x: Multi-stream state [batch, seq, n, C]

        Returns:
            layer_input: [batch, seq, C]
            cache: Mapping tensors for apply_residual
        """
        return self.mhc.get_aggregated_input(x)

    def apply_residual(
        self,
        x: torch.Tensor,
        sublayer_output: torch.Tensor,
        cache: Tuple,
    ) -> torch.Tensor:
        """
        Apply mHC residual to produce updated n-stream state.

        Args:
            x: Multi-stream state [batch, seq, n, C]
            sublayer_output: Output of F [batch, seq, C]
            cache: Mapping tensors from get_layer_input

        Returns:
            Updated multi-stream state [batch, seq, n, C]
        """
        return self.mhc.apply_cached(x, sublayer_output, cache)


# =============================================================================
# Parameter counting utilities
# =============================================================================


def count_mhc_parameters_per_module(hidden_size: int, expansion_rate: int = 4) -> dict:
    """
    Count parameters in a single mHC module.

    Based on paper's parameterization:
    - φ_pre: nC × n
    - φ_post: nC × n
    - φ_res: nC × n²
    - b_pre: n, b_post: n, b_res: n²
    - α: 3 scalars
    - RMSNorm weight: nC
    """
    n = expansion_rate
    C = hidden_size

    breakdown = {
        "phi_pre": n * C * n,  # nC × n
        "phi_post": n * C * n,  # nC × n
        "phi_res": n * C * n * n,  # nC × n²
        "b_pre": n,
        "b_post": n,
        "b_res": n * n,
        "alpha_scalars": 3,
        "rms_weight": n * C,
    }
    breakdown["total"] = sum(breakdown.values())

    return breakdown


def print_mhc_overhead(
    hidden_size: int = 2048, num_layers: int = 24, expansion_rate: int = 4
):
    """Print mHC parameter overhead analysis."""
    breakdown = count_mhc_parameters_per_module(hidden_size, expansion_rate)

    print(f"\n{'='*60}")
    print("mHC Parameter Analysis")
    print(f"Configuration: n={expansion_rate}, C={hidden_size}, layers={num_layers}")
    print(f"{'='*60}")

    print("\nPer mHC module breakdown:")
    for key, val in breakdown.items():
        if key != "total":
            print(f"  {key:15s}: {val:>10,}")
    print(f"  {'TOTAL':15s}: {breakdown['total']:>10,}")

    # 2 mHC per layer (attention + FFN)
    per_layer = breakdown["total"] * 2
    total = per_layer * num_layers

    print(f"\nPer transformer layer (2 mHC): {per_layer:,}")
    print(f"Total for {num_layers} layers: {total:,}")
    print(f"Total in millions: {total / 1e6:.2f}M")

    # Compare to 1B model
    model_1b = 1.1e9
    print(f"\nOverhead vs 1B model: {100 * total / model_1b:.2f}%")
    print(f"{'='*60}\n")


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing mHC Implementation")
    print("=" * 60)

    # Config
    batch_size = 2
    seq_len = 128
    hidden_size = 2048
    n = 4

    # --- Test core ManifoldConstrainedHyperConnection ---
    mhc = ManifoldConstrainedHyperConnection(hidden_size=hidden_size, expansion_rate=n)

    actual_params = sum(p.numel() for p in mhc.parameters())
    expected = count_mhc_parameters_per_module(hidden_size, n)["total"]

    print(f"Actual parameters: {actual_params:,}")
    print(f"Expected from formula: {expected:,}")
    print(f"Match: {actual_params == expected}")

    x = torch.randn(batch_size, seq_len, n, hidden_size)
    layer_output = torch.randn(batch_size, seq_len, hidden_size)

    layer_input, cache = mhc.get_aggregated_input(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Layer input shape: {layer_input.shape}")

    x_new = mhc.apply_cached(x, layer_output, cache)
    print(f"Output shape: {x_new.shape}")

    # Verify doubly stochastic property
    H_pre, H_post, H_res = mhc.mapping(x)
    print("\nH_res verification:")
    print(f"  Shape: {H_res.shape}")
    row_sums = H_res[0, 0].sum(dim=-1)
    col_sums = H_res[0, 0].sum(dim=-2)
    print(f"  Row sums (should be ~1): {row_sums.tolist()}")
    print(f"  Col sums (should be ~1): {col_sums.tolist()}")
    print(f"  All non-negative: {(H_res >= 0).all().item()}")

    # --- Test MHCSublayerConnection (used by transformer) ---
    print("\n--- MHCSublayerConnection ---")
    conn = MHCSublayerConnection(hidden_size=hidden_size, expansion_rate=n)
    layer_in, cache = conn.get_layer_input(x)
    print(f"  Aggregated input: {layer_in.shape}")
    x_updated = conn.apply_residual(x, layer_output, cache)
    print(f"  Updated n-stream: {x_updated.shape}")
    assert x_updated.shape == x.shape, "n-stream shape must be preserved!"
    print("  Shape preserved across sublayer: OK")

    # Full overhead analysis
    print_mhc_overhead(hidden_size=2048, num_layers=24, expansion_rate=4)
