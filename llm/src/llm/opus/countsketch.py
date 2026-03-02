from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class _ShapeCache:
    row_hash: torch.Tensor
    row_sign: torch.Tensor
    col_hash: torch.Tensor
    col_sign: torch.Tensor


class CountSketchProjector:
    """
    Deterministic CountSketch projection for linear-layer gradients.

    The projector avoids materializing the full [out_dim, in_dim] gradient tensor by
    processing row chunks and accumulating into sketch bins with scatter-add.
    """

    def __init__(self, sketch_dim: int = 8192, seed: int = 42, row_chunk_size: int = 64):
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

    def _get_cache(self, out_dim: int, in_dim: int, device: torch.device, sketch_key: str = "") -> _ShapeCache:
        key_hash = self._stable_key_hash(sketch_key)
        key = (out_dim, in_dim, device, key_hash)
        if key in self._cache:
            return self._cache[key]

        g = torch.Generator(device="cpu")
        g.manual_seed(self._shape_seed(out_dim, in_dim, key_hash))

        row_hash = torch.randint(0, self.sketch_dim, (out_dim,), generator=g, dtype=torch.int64)
        row_sign = torch.randint(0, 2, (out_dim,), generator=g, dtype=torch.int8).to(torch.float32)
        row_sign = row_sign.mul_(2.0).sub_(1.0)

        col_hash = torch.randint(0, self.sketch_dim, (in_dim,), generator=g, dtype=torch.int64)
        col_sign = torch.randint(0, 2, (in_dim,), generator=g, dtype=torch.int8).to(torch.float32)
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

        Args:
            activations: [T, in_dim]
            grad_outputs: [T, out_dim]
            preconditioner: [out_dim, in_dim] or None
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

        # Row-chunk accumulation to bound peak memory.
        chunk = max(1, self.row_chunk_size)
        for row_start in range(0, out_dim, chunk):
            row_end = min(out_dim, row_start + chunk)
            g_chunk = g[:, row_start:row_end]  # [T, chunk]

            # Equivalent to sum_t outer(g_t, a_t) for this row slice.
            grad_chunk = g_chunk.transpose(0, 1).matmul(a)  # [chunk, in_dim]
            grad_chunk = torch.nan_to_num(grad_chunk, nan=0.0, posinf=0.0, neginf=0.0)

            if p is not None:
                grad_chunk = grad_chunk * p[row_start:row_end]
                grad_chunk = torch.nan_to_num(grad_chunk, nan=0.0, posinf=0.0, neginf=0.0)

            pair_hash = (cache.row_hash[row_start:row_end].unsqueeze(1) + cache.col_hash.unsqueeze(0)) % self.sketch_dim
            pair_sign = cache.row_sign[row_start:row_end].unsqueeze(1) * cache.col_sign.unsqueeze(0)

            contrib = (grad_chunk * pair_sign).reshape(-1)
            contrib = torch.nan_to_num(contrib, nan=0.0, posinf=0.0, neginf=0.0)
            sketch.scatter_add_(0, pair_hash.reshape(-1), contrib)

        sketch = torch.nan_to_num(sketch, nan=0.0, posinf=0.0, neginf=0.0)
        return sketch.to(out_dtype)

    def project_outer_sum_sample(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        preconditioner: torch.Tensor | None,
        row_dim: int,
        col_dim: int,
        out_dtype: torch.dtype = torch.float32,
        sketch_key: str = "",
    ) -> torch.Tensor:
        """
        Project gradient of matrix parameter with shape [row_dim, col_dim].

        Given per-token/assignment factors `left` [N, row_dim] and `right` [N, col_dim],
        this sketches grad = left^T @ right without materializing the full matrix.
        """
        if left.dim() != 2 or right.dim() != 2:
            raise ValueError("project_outer_sum_sample expects [N, D] tensors")
        if left.shape[0] != right.shape[0]:
            raise ValueError("left/right leading dimensions must match")

        device = left.device
        cache = self._get_cache(row_dim, col_dim, device, sketch_key=sketch_key)

        l = self._sanitize_f32(left)
        r = self._sanitize_f32(right)
        sketch = torch.zeros(self.sketch_dim, device=device, dtype=torch.float32)
        p = self._sanitize_f32(preconditioner) if preconditioner is not None else None

        chunk = max(1, self.row_chunk_size)
        for row_start in range(0, row_dim, chunk):
            row_end = min(row_dim, row_start + chunk)
            l_chunk = l[:, row_start:row_end]  # [N, chunk]

            grad_chunk = l_chunk.transpose(0, 1).matmul(r)  # [chunk, col_dim]
            grad_chunk = torch.nan_to_num(grad_chunk, nan=0.0, posinf=0.0, neginf=0.0)

            if p is not None:
                grad_chunk = grad_chunk * p[row_start:row_end]
                grad_chunk = torch.nan_to_num(grad_chunk, nan=0.0, posinf=0.0, neginf=0.0)

            pair_hash = (
                cache.row_hash[row_start:row_end].unsqueeze(1) + cache.col_hash.unsqueeze(0)
            ) % self.sketch_dim
            pair_sign = cache.row_sign[row_start:row_end].unsqueeze(1) * cache.col_sign.unsqueeze(0)

            contrib = (grad_chunk * pair_sign).reshape(-1)
            contrib = torch.nan_to_num(contrib, nan=0.0, posinf=0.0, neginf=0.0)
            sketch.scatter_add_(0, pair_hash.reshape(-1), contrib)

        sketch = torch.nan_to_num(sketch, nan=0.0, posinf=0.0, neginf=0.0)
        return sketch.to(out_dtype)

    def project_outer_sum_pair_sample(
        self,
        left: torch.Tensor,
        right_a: torch.Tensor,
        right_b: torch.Tensor,
        preconditioner_a: torch.Tensor | None,
        preconditioner_b: torch.Tensor | None,
        row_dim: int,
        col_dim: int,
        out_dtype: torch.dtype = torch.float32,
        sketch_key_a: str = "",
        sketch_key_b: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Project two outer-sum gradients sharing the same left factors.

        Useful for MoE gate/up matrices where both gradients share input activations.
        """
        if left.dim() != 2 or right_a.dim() != 2 or right_b.dim() != 2:
            raise ValueError("project_outer_sum_pair_sample expects [N, D] tensors")
        if left.shape[0] != right_a.shape[0] or left.shape[0] != right_b.shape[0]:
            raise ValueError("left/right leading dimensions must match")

        device = left.device
        cache_a = self._get_cache(row_dim, col_dim, device, sketch_key=sketch_key_a)
        if sketch_key_b == sketch_key_a:
            cache_b = cache_a
        else:
            cache_b = self._get_cache(row_dim, col_dim, device, sketch_key=sketch_key_b)

        l = self._sanitize_f32(left)
        ra = self._sanitize_f32(right_a)
        rb = self._sanitize_f32(right_b)
        sketch_a = torch.zeros(self.sketch_dim, device=device, dtype=torch.float32)
        sketch_b = torch.zeros(self.sketch_dim, device=device, dtype=torch.float32)
        pa = self._sanitize_f32(preconditioner_a) if preconditioner_a is not None else None
        pb = self._sanitize_f32(preconditioner_b) if preconditioner_b is not None else None

        chunk = max(1, self.row_chunk_size)
        for row_start in range(0, row_dim, chunk):
            row_end = min(row_dim, row_start + chunk)
            l_chunk = l[:, row_start:row_end]

            grad_a = l_chunk.transpose(0, 1).matmul(ra)
            grad_b = l_chunk.transpose(0, 1).matmul(rb)
            grad_a = torch.nan_to_num(grad_a, nan=0.0, posinf=0.0, neginf=0.0)
            grad_b = torch.nan_to_num(grad_b, nan=0.0, posinf=0.0, neginf=0.0)

            if pa is not None:
                grad_a = grad_a * pa[row_start:row_end]
                grad_a = torch.nan_to_num(grad_a, nan=0.0, posinf=0.0, neginf=0.0)
            if pb is not None:
                grad_b = grad_b * pb[row_start:row_end]
                grad_b = torch.nan_to_num(grad_b, nan=0.0, posinf=0.0, neginf=0.0)

            pair_hash_a = (
                cache_a.row_hash[row_start:row_end].unsqueeze(1) + cache_a.col_hash.unsqueeze(0)
            ) % self.sketch_dim
            pair_sign_a = cache_a.row_sign[row_start:row_end].unsqueeze(1) * cache_a.col_sign.unsqueeze(0)
            pair_hash_b = (
                cache_b.row_hash[row_start:row_end].unsqueeze(1) + cache_b.col_hash.unsqueeze(0)
            ) % self.sketch_dim
            pair_sign_b = cache_b.row_sign[row_start:row_end].unsqueeze(1) * cache_b.col_sign.unsqueeze(0)

            contrib_a = (grad_a * pair_sign_a).reshape(-1)
            contrib_b = (grad_b * pair_sign_b).reshape(-1)
            contrib_a = torch.nan_to_num(contrib_a, nan=0.0, posinf=0.0, neginf=0.0)
            contrib_b = torch.nan_to_num(contrib_b, nan=0.0, posinf=0.0, neginf=0.0)
            sketch_a.scatter_add_(0, pair_hash_a.reshape(-1), contrib_a)
            sketch_b.scatter_add_(0, pair_hash_b.reshape(-1), contrib_b)

        sketch_a = torch.nan_to_num(sketch_a, nan=0.0, posinf=0.0, neginf=0.0)
        sketch_b = torch.nan_to_num(sketch_b, nan=0.0, posinf=0.0, neginf=0.0)
        return sketch_a.to(out_dtype), sketch_b.to(out_dtype)

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
        Project batch of linear gradients to sketch vectors.

        Args:
            activations: [B, T, in_dim] or [B, in_dim]
            grad_outputs: [B, T, out_dim] or [B, out_dim]
            preconditioner: [out_dim, in_dim] or None
        Returns:
            [B, m]
        """
        a = self._ensure_btd(activations)
        g = self._ensure_btd(grad_outputs)
        if a.shape[0] != g.shape[0] or a.shape[1] != g.shape[1]:
            raise ValueError("Batch/token dims mismatch between activations and grad_outputs")

        bsz = a.shape[0]
        out = []
        for i in range(bsz):
            out.append(
                self.project_linear_sample(
                    activations=a[i],
                    grad_outputs=g[i],
                    preconditioner=preconditioner,
                    out_dim=out_dim,
                    in_dim=in_dim,
                    out_dtype=out_dtype,
                    sketch_key=sketch_key,
                )
            )
        return torch.stack(out, dim=0)
