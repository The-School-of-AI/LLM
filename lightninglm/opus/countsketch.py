from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from .triton_countsketch import fused_sketch_scatter


@dataclass
class _ShapeCache:
    row_hash: torch.Tensor
    row_sign: torch.Tensor
    col_hash: torch.Tensor
    col_sign: torch.Tensor


class CountSketchProjector:
    """
    Deterministic CountSketch projection for linear-layer gradients.

    Two projection modes:
      1. project_linear_batch_fft (NEW, default):
         Uses the tensor sketch identity CS(b⊗a) = CS_row(b) ★ CS_col(a)
         computed via FFT circular convolution. Cost: O(B*T*(d_in+d_out+m*log(m)))
         — never materializes the outer product.

      2. project_linear_batch (OLD, kept as fallback):
         Materializes partial outer products via bmm row-chunks then scatter-adds.
         Cost: O(B*T*d_in*d_out). Used only when explicitly requested.
    """

    def __init__(
        self, sketch_dim: int = 8192, seed: int = 42, row_chunk_size: int = 64
    ):
        if sketch_dim <= 0:
            raise ValueError("sketch_dim must be > 0")
        self.sketch_dim = int(sketch_dim)
        self.seed = int(seed)
        self.row_chunk_size = int(row_chunk_size)
        self._cache: Dict[Tuple[int, int, torch.device, int], _ShapeCache] = {}

    @staticmethod
    def _stable_key_hash(sketch_key: str) -> int:
        if not sketch_key:
            return 0
        h = hashlib.blake2b(sketch_key.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, byteorder="little", signed=False)

    def _shape_seed(self, out_dim: int, in_dim: int, key_hash: int) -> int:
        mixed = (
            int(self.seed)
            + (int(out_dim) * 1_000_003)
            + (int(in_dim) * 1_000_033)
            + (int(key_hash) * 1_000_037)
        )
        return int(mixed % (2**63 - 1))

    def _get_cache(
        self, out_dim: int, in_dim: int, device: torch.device, sketch_key: str = ""
    ) -> _ShapeCache:
        key_hash = self._stable_key_hash(sketch_key)
        key = (out_dim, in_dim, device, key_hash)
        if key in self._cache:
            return self._cache[key]

        g = torch.Generator(device="cpu")
        g.manual_seed(self._shape_seed(out_dim, in_dim, key_hash))

        row_hash = torch.randint(
            0, self.sketch_dim, (out_dim,), generator=g, dtype=torch.int64
        )
        row_sign = torch.randint(0, 2, (out_dim,), generator=g, dtype=torch.int8).to(
            torch.float32
        )
        row_sign = row_sign.mul_(2.0).sub_(1.0)

        col_hash = torch.randint(
            0, self.sketch_dim, (in_dim,), generator=g, dtype=torch.int64
        )
        col_sign = torch.randint(0, 2, (in_dim,), generator=g, dtype=torch.int8).to(
            torch.float32
        )
        col_sign = col_sign.mul_(2.0).sub_(1.0)

        cache = _ShapeCache(
            row_hash=row_hash.to(device=device, non_blocking=True),
            row_sign=row_sign.to(device=device, non_blocking=True),
            col_hash=col_hash.to(device=device, non_blocking=True),
            col_sign=col_sign.to(device=device, non_blocking=True),
        )
        self._cache[key] = cache
        return cache

    @staticmethod
    def _ensure_btd(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x
        if x.dim() == 2:
            return x.unsqueeze(1)
        raise ValueError(f"Expected tensor with dim 2 or 3, got shape {tuple(x.shape)}")

    @staticmethod
    def _sanitize_f32(x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # ─────────────────────────────────────────────────────────────────────
    # NEW: Tensor Sketch via FFT (O(d_in + d_out + m log m) per token)
    # ─────────────────────────────────────────────────────────────────────

    def _scatter_to_sketch(
        self,
        x: torch.Tensor,  # [B, T, D]
        hash_table: torch.Tensor,  # [D] int64 — maps each dim to sketch bin
        sign_table: torch.Tensor,  # [D] float32 — ±1 per dim
        precond: torch.Tensor | None,  # [D] float32 — per-dim scaling, or None
    ) -> torch.Tensor:
        """
        Scatter a dense vector into sketch space: CS(x)[k] = Σ_{j: h(j)=k} s(j) * x[j]
        Optionally multiplies by per-dim preconditioner before scattering.

        Args:
            x: [B, T, D] — input vectors
            hash_table: [D] — bin assignment for each dim
            sign_table: [D] — ±1 sign for each dim
            precond: [D] or None — per-dim scaling

        Returns: [B, T, m] — sketch vectors
        """
        B, T, D = x.shape
        m = self.sketch_dim

        # Apply sign (and optional preconditioner) to input
        # x_signed: [B, T, D]
        if precond is not None:
            x_signed = x * (sign_table * precond).unsqueeze(0).unsqueeze(0)
        else:
            x_signed = x * sign_table.unsqueeze(0).unsqueeze(0)

        # Scatter into sketch bins: [B, T, m]
        x_flat = x_signed.reshape(B * T, D)
        sketch_flat = torch.zeros(B * T, m, device=x.device, dtype=torch.float32)
        idx = hash_table.unsqueeze(0).expand(B * T, -1)  # [B*T, D]
        sketch_flat.scatter_add_(1, idx, x_flat)

        return sketch_flat.reshape(B, T, m)

    def project_linear_batch_fft(
        self,
        activations: torch.Tensor,  # [B, T, in_dim] or [B, in_dim]
        grad_outputs: torch.Tensor,  # [B, T, out_dim] or [B, out_dim]
        precond_row: torch.Tensor | None,  # [out_dim] — row factor of preconditioner
        precond_col: torch.Tensor | None,  # [in_dim] — column factor of preconditioner
        out_dim: int,
        in_dim: int,
        out_dtype: torch.dtype = torch.float32,
        sketch_key: str = "",
    ) -> torch.Tensor:
        """
        Tensor sketch of per-sample gradient: CS(b ⊗ a) = CS_row(b) ★ CS_col(a)
        via FFT circular convolution. Never materializes the outer product.

        With factored preconditioner P[i,j] ≈ p_row[i] * p_col[j]:
            CS(P ⊙ (b ⊗ a)) = CS_row(p_row ⊙ b) ★ CS_col(p_col ⊙ a)

        Cost: O(B * T * (d_in + d_out + m * log(m)))
        vs old bmm approach: O(B * T * d_in * d_out)

        Args:
            activations: [B, T, in_dim]  — input to linear layer
            grad_outputs: [B, T, out_dim] — grad of output (error signal)
            precond_row: [out_dim] — row factor of factored preconditioner, or None
            precond_col: [in_dim] — col factor of factored preconditioner, or None
            out_dim, in_dim: dimensions of the weight matrix
            sketch_key: layer name for deterministic hash tables

        Returns: [B, sketch_dim] — per-sample sketch of (preconditioned) gradient
        """
        a = self._ensure_btd(activations).float()
        g = self._ensure_btd(grad_outputs).float()
        if a.shape[0] != g.shape[0] or a.shape[1] != g.shape[1]:
            raise ValueError("Batch/token dims mismatch")

        B, T, _ = a.shape
        device = a.device
        m = self.sketch_dim
        cache = self._get_cache(out_dim, in_dim, device, sketch_key=sketch_key)

        # Step 1: Scatter activations into sketch space with col hash/sign
        # CS_col(a)[b,t,k] = Σ_{j: col_hash(j)=k} col_sign(j) * a[b,t,j]
        # If precond_col is provided: multiply a by p_col first
        cs_a = self._scatter_to_sketch(a, cache.col_hash, cache.col_sign, precond_col)
        # [B, T, m]

        # Step 2: Scatter grad_outputs into sketch space with row hash/sign
        # CS_row(b)[b,t,k] = Σ_{j: row_hash(j)=k} row_sign(j) * b[b,t,j]
        # If precond_row is provided: multiply g by p_row first
        cs_g = self._scatter_to_sketch(g, cache.row_hash, cache.row_sign, precond_row)
        # [B, T, m]

        # Step 3: Circular convolution via FFT
        # CS(b⊗a) = IFFT(FFT(CS_row(b)) * FFT(CS_col(a)))
        # Sum over tokens: Σ_t CS(b_t ⊗ a_t) = IFFT(Σ_t FFT(CS_row(b_t)) * FFT(CS_col(a_t)))
        fft_a = torch.fft.rfft(cs_a, n=m, dim=-1)  # [B, T, m//2+1] complex
        fft_g = torch.fft.rfft(cs_g, n=m, dim=-1)  # [B, T, m//2+1] complex

        # Pointwise multiply and sum over tokens in one go
        fft_prod_sum = (fft_g * fft_a).sum(dim=1)  # [B, m//2+1] complex

        # IFFT to get final sketch
        sketches = torch.fft.irfft(fft_prod_sum, n=m, dim=-1)  # [B, m]

        sketches = torch.nan_to_num(sketches, nan=0.0, posinf=0.0, neginf=0.0)
        return sketches.to(out_dtype)

    # ─────────────────────────────────────────────────────────────────────
    # OLD: bmm-based (kept as fallback, used by project_linear_sample etc.)
    # ─────────────────────────────────────────────────────────────────────

    def project_linear_sample(
        self,
        activations: torch.Tensor,
        grad_outputs: torch.Tensor,
        preconditioner: torch.Tensor | None,
        out_dim: int,
        in_dim: int,
        out_dtype: torch.dtype = torch.float32,
        sketch_key: str = "",
    ) -> torch.Tensor:
        """
        Project one sample's linear weight gradient into CountSketch space.
        (Legacy bmm-based approach, kept for compatibility.)
        """
        if activations.dim() != 2 or grad_outputs.dim() != 2:
            raise ValueError("project_linear_sample expects [T, D] tensors")
        if activations.shape[0] != grad_outputs.shape[0]:
            raise ValueError("activation/grad_output token dimensions must match")

        device = activations.device
        cache = self._get_cache(out_dim, in_dim, device, sketch_key=sketch_key)

        a = self._sanitize_f32(activations)
        g = self._sanitize_f32(grad_outputs)
        sketch = torch.zeros(self.sketch_dim, device=device, dtype=torch.float32)
        p = self._sanitize_f32(preconditioner) if preconditioner is not None else None

        chunk = max(1, self.row_chunk_size)
        for row_start in range(0, out_dim, chunk):
            row_end = min(out_dim, row_start + chunk)
            g_chunk = g[:, row_start:row_end]

            grad_chunk = g_chunk.transpose(0, 1).matmul(a)
            grad_chunk = torch.nan_to_num(grad_chunk, nan=0.0, posinf=0.0, neginf=0.0)

            if p is not None:
                grad_chunk = grad_chunk * p[row_start:row_end]
                grad_chunk = torch.nan_to_num(
                    grad_chunk, nan=0.0, posinf=0.0, neginf=0.0
                )

            pair_hash = (
                cache.row_hash[row_start:row_end].unsqueeze(1)
                + cache.col_hash.unsqueeze(0)
            ) % self.sketch_dim
            pair_sign = cache.row_sign[row_start:row_end].unsqueeze(
                1
            ) * cache.col_sign.unsqueeze(0)

            contrib = (grad_chunk * pair_sign).reshape(-1)
            contrib = torch.nan_to_num(contrib, nan=0.0, posinf=0.0, neginf=0.0)
            sketch.scatter_add_(0, pair_hash.reshape(-1), contrib)

        sketch = torch.nan_to_num(sketch, nan=0.0, posinf=0.0, neginf=0.0)
        return sketch.to(out_dtype)

    def project_linear_batch(
        self,
        activations: torch.Tensor,
        grad_outputs: torch.Tensor,
        preconditioner: torch.Tensor | None,
        out_dim: int,
        in_dim: int,
        out_dtype: torch.dtype = torch.float32,
        sketch_key: str = "",
    ) -> torch.Tensor:
        """
        Legacy bmm-based batch projection. Kept for fallback/comparison.
        Prefer project_linear_batch_fft for production use.
        """
        a = self._ensure_btd(activations)
        g = self._ensure_btd(grad_outputs)
        if a.shape[0] != g.shape[0] or a.shape[1] != g.shape[1]:
            raise ValueError(
                "Batch/token dims mismatch between activations and grad_outputs"
            )

        bsz = a.shape[0]
        device = a.device
        cache = self._get_cache(out_dim, in_dim, device, sketch_key=sketch_key)

        a_f = a.float()
        g_f = g.float()
        p = self._sanitize_f32(preconditioner) if preconditioner is not None else None

        sketches = torch.zeros(bsz, self.sketch_dim, device=device, dtype=torch.float32)

        chunk = max(1, self.row_chunk_size)
        for row_start in range(0, out_dim, chunk):
            row_end = min(out_dim, row_start + chunk)
            g_chunk = g_f[:, :, row_start:row_end]

            grad_chunk = torch.bmm(g_chunk.transpose(1, 2), a_f)

            pair_hash = (
                cache.row_hash[row_start:row_end].unsqueeze(1)
                + cache.col_hash.unsqueeze(0)
            ) % self.sketch_dim
            pair_sign = cache.row_sign[row_start:row_end].unsqueeze(
                1
            ) * cache.col_sign.unsqueeze(0)

            p_slice = p[row_start:row_end] if p is not None else None
            fused_sketch_scatter(grad_chunk, p_slice, pair_sign, pair_hash, sketches)

        sketches = torch.nan_to_num(sketches, nan=0.0, posinf=0.0, neginf=0.0)
        return sketches.to(out_dtype)
