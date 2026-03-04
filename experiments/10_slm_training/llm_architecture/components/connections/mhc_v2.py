"""
Manifold-Constrained Hyper-Connections V2
==========================================

Implementation matching Test_Code/model_1b.py lines 1059-1148.

Key differences from mhc.py (V1):
- alpha_init = 0.1 (not 0.01)
- b_pre/b_post are 1D tensors (not 2D)
- b_res is zero-initialized (not diagonal-dominant)
- Norm is INSIDE MHCSublayerV2 (after H_pre aggregation, before sublayer)
- Uses weighted sum for H_pre aggregation (not matmul)
- Uses einsum for H_res mixing
- Sinkhorn via Triton kernel (CUDA) or JIT fallback (CPU)
- Handles tuple returns from MoEFFN (output, aux_loss)

Performance optimizations:
- Triton Sinkhorn kernel: 40 kernel launches -> 1
- TritonRMSNorm: fused variance + rsqrt + mul
"""

import torch
import torch.nn as nn

# ============================================================================
# Triton imports with graceful fallback
# ============================================================================

try:
    from components.kernels.triton_normalization import TritonRMSNorm

    _HAS_TRITON_NORM = True
except ImportError:
    _HAS_TRITON_NORM = False

try:
    from components.kernels.triton_sinkhorn import triton_sinkhorn_knopp

    _HAS_TRITON_SINKHORN = True
except ImportError:
    _HAS_TRITON_SINKHORN = False


# ============================================================================
# Sinkhorn-Knopp (JIT-compiled fallback for CPU / non-CUDA)
# ============================================================================


@torch.jit.script
def sinkhorn_knopp(
    logits: torch.Tensor, iters: int = 20, eps: float = 1e-6
) -> torch.Tensor:
    """Doubly-stochastic matrix via Sinkhorn-Knopp."""
    M = torch.exp(logits).clamp_min(eps)
    for _ in range(iters):
        M = M / (M.sum(dim=-1, keepdim=True).clamp_min(eps))
        M = M / (M.sum(dim=-2, keepdim=True).clamp_min(eps))
    return M


def _sinkhorn_dispatch(
    logits: torch.Tensor, iters: int, eps: float = 1e-6
) -> torch.Tensor:
    """Dispatch Sinkhorn to Triton (CUDA) or JIT (CPU) backend."""
    if _HAS_TRITON_SINKHORN and logits.is_cuda:
        try:
            return triton_sinkhorn_knopp(logits, num_iters=iters, eps=eps)
        except Exception:
            pass
    return sinkhorn_knopp(logits, iters=iters, eps=eps)


# ============================================================================
# RMSNorm (auto-selects Triton on CUDA, PyTorch on CPU)
# ============================================================================

if _HAS_TRITON_NORM:
    RMSNorm = TritonRMSNorm
else:

    class RMSNorm(nn.Module):
        """RMS Layer Normalization."""

        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            norm = x.pow(2).mean(dim=-1, keepdim=True)
            x = x * torch.rsqrt(norm + self.eps)
            return self.weight * x


# ============================================================================
# MHC Coefficients (matching Test_Code MHCCoeffs)
# ============================================================================


class MHCCoeffsV2(nn.Module):
    """
    Produces routing coefficients for mHC.

    Matching Test_Code MHCCoeffs (lines 1073-1115):
    - alpha_init = 0.1
    - b_pre/b_post are 1D tensors
    - b_res is zero-initialized
    - RMSNorm on flattened n*D input
    """

    def __init__(self, d_model: int, n_streams: int = 4, iters: int = 20):
        super().__init__()
        self.d_model = d_model
        self.n = n_streams
        self.iters = iters

        d_in = self.n * d_model

        self.phi_pre = nn.Linear(d_in, self.n, bias=False)
        self.phi_post = nn.Linear(d_in, self.n, bias=False)
        self.phi_res = nn.Linear(d_in, self.n * self.n, bias=False)

        # 1D biases (matching Test_Code)
        self.b_pre = nn.Parameter(torch.zeros(self.n))
        self.b_post = nn.Parameter(torch.zeros(self.n))
        # Zero-initialized (NOT diagonal-dominant like V1)
        self.b_res = nn.Parameter(torch.zeros(self.n, self.n))

        # Alpha init = 0.1 (matching Test_Code, NOT 0.01 like V1)
        self.alpha_pre = nn.Parameter(torch.tensor(0.1))
        self.alpha_post = nn.Parameter(torch.tensor(0.1))
        self.alpha_res = nn.Parameter(torch.tensor(0.1))

        self.rms = RMSNorm(d_in)

        for m in [self.phi_pre, self.phi_post, self.phi_res]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x_stream: torch.Tensor):
        """
        Compute H_pre, H_post, H_res from input streams.

        Args:
            x_stream: (B, T, n, D)

        Returns:
            H_pre: (B, T, n) - aggregation weights
            H_post: (B, T, n) - distribution weights (scaled 0-2)
            H_res: (B, T, n, n) - mixing matrix (doubly stochastic)
        """
        B, T, n, D = x_stream.shape
        x_flat = x_stream.reshape(B, T, n * D)
        x_flat = self.rms(x_flat)

        pre_logits = self.alpha_pre * self.phi_pre(x_flat) + self.b_pre
        post_logits = self.alpha_post * self.phi_post(x_flat) + self.b_post

        res_logits = self.alpha_res * self.phi_res(x_flat)
        res_logits = res_logits.view(B, T, n, n) + self.b_res

        H_pre = torch.sigmoid(pre_logits)  # (B, T, n)
        H_post = 2.0 * torch.sigmoid(post_logits)  # (B, T, n) scaled to [0, 2]
        H_res = _sinkhorn_dispatch(res_logits, iters=self.iters)  # (B, T, n, n)

        return H_pre, H_post, H_res


# ============================================================================
# MHC Sublayer Wrapper V2 (matching Test_Code MHCSublayer)
# ============================================================================


class MHCSublayerV2(nn.Module):
    """
    Wrap sublayer with mHC residual routing.

    CRITICAL: Norm is INSIDE this module (after H_pre aggregation, before sublayer).
    This differs from V1 where norm is in the transformer block.

    Handles both:
    - Single tensor returns (GatedDeltaNet, ReferenceGSA)
    - Tuple returns (LightningMLP: output, aux_loss)

    Flow:
    1. Compute H_pre, H_post, H_res from coefficients
    2. Aggregate streams: x_in = (x_stream * H_pre.unsqueeze(-1)).sum(dim=2)
    3. Apply norm: x_in = self.norm(x_in)
    4. Run sublayer: out = sublayer(x_in) or sublayer(x_in, attention_mask)
    5. Distribute: y_stream = y.unsqueeze(2) * H_post.unsqueeze(-1)
    6. Mix: x_res = einsum("btij,btjd->btid", H_res, x_stream)
    7. Return: x_res + y_stream, aux_loss
    """

    def __init__(
        self,
        d_model: int,
        n_streams: int,
        sublayer: nn.Module,
        norm: nn.Module,
        iters: int = 20,
    ):
        super().__init__()
        self.d_model = d_model
        self.n = n_streams
        self.sublayer = sublayer
        self.norm = norm  # Norm stored INSIDE (critical difference from V1)
        self.coeffs = MHCCoeffsV2(d_model=d_model, n_streams=n_streams, iters=iters)

    def forward(self, x_stream: torch.Tensor, attention_mask=None):
        """
        Forward pass with mHC routing.

        Args:
            x_stream: (B, T, n, D) - multi-stream state
            attention_mask: Optional attention mask (passed to sublayer if needed)

        Returns:
            x_stream: (B, T, n, D) - updated multi-stream state
            aux_loss: Scalar auxiliary loss (or None)
        """
        H_pre, H_post, H_res = self.coeffs(x_stream)

        # Aggregate streams via weighted sum
        # H_pre: (B, T, n) -> (B, T, n, 1) for broadcasting with (B, T, n, D)
        x_in = (x_stream * H_pre.unsqueeze(-1)).sum(dim=2)  # (B, T, D)

        # Apply norm INSIDE the mHC wrapper
        x_in = self.norm(x_in)

        # Run sublayer
        aux_loss = None
        if attention_mask is None:
            out = self.sublayer(x_in)
        else:
            out = self.sublayer(x_in, attention_mask)

        # Handle tuple returns (MoEFFN returns (output, aux_loss))
        if isinstance(out, tuple):
            y, aux_loss = out
        else:
            y = out

        # Distribute output to streams
        # y: (B, T, D) -> (B, T, 1, D)
        # H_post: (B, T, n) -> (B, T, n, 1)
        y_stream = y.unsqueeze(2) * H_post.unsqueeze(-1)  # (B, T, n, D)

        # Mix existing streams
        x_res = torch.einsum("btij,btjd->btid", H_res, x_stream)  # (B, T, n, D)

        return x_res + y_stream, aux_loss
