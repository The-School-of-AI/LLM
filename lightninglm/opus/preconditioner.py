from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class _GroupHyper:
    lr: float
    beta1: float
    beta2: float
    eps: float


class AdamWPreconditionerView:
    """
    Frozen, read-only AdamW preconditioner view for the current step.

    P_t = C_t * diag(1 / (sqrt(v_hat) + eps))
    C_t = lr * (1 - beta1) / (1 - beta1^step)

    If optimizer state is unavailable for a parameter (possible with some sharded
    implementations), this view falls back to a scalar C_t identity scaling.
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, strict_shard_only: bool = False
    ):
        self.optimizer = optimizer
        self.strict_shard_only = bool(strict_shard_only)
        self._param_to_group: Dict[int, _GroupHyper] = {}
        self._build_group_index()
        self._cache: Dict[int, torch.Tensor] = {}
        self._slice_cache: Dict[tuple[int, int], torch.Tensor] = {}

    def _build_group_index(self) -> None:
        self._param_to_group.clear()
        for group in self.optimizer.param_groups:
            lr = float(group.get("lr", 1e-3))
            betas = group.get("betas", (0.9, 0.999))
            eps = float(group.get("eps", 1e-8))
            gh = _GroupHyper(
                lr=lr, beta1=float(betas[0]), beta2=float(betas[1]), eps=eps
            )
            for p in group["params"]:
                self._param_to_group[id(p)] = gh

    def refresh(self) -> None:
        self._build_group_index()
        self._cache.clear()
        self._slice_cache.clear()

    def _scalar_fallback(self, p: torch.nn.Parameter) -> torch.Tensor:
        if self.strict_shard_only:
            return torch.zeros_like(p.data, dtype=torch.float32)
        gh = self._param_to_group.get(id(p), None)
        if gh is None:
            return torch.ones_like(p.data, dtype=torch.float32)
        return torch.full_like(
            p.data, fill_value=gh.lr * (1.0 - gh.beta1), dtype=torch.float32
        )

    def _scalar_fallback_like(
        self, p: torch.nn.Parameter, ref: torch.Tensor
    ) -> torch.Tensor:
        if self.strict_shard_only:
            return torch.zeros_like(ref, dtype=torch.float32)
        gh = self._param_to_group.get(id(p), None)
        if gh is None:
            return torch.ones_like(ref, dtype=torch.float32)
        return torch.full_like(
            ref, fill_value=gh.lr * (1.0 - gh.beta1), dtype=torch.float32
        )

    @staticmethod
    def _resolve_step(step) -> float:
        if torch.is_tensor(step):
            step_f = float(step.item())
        elif step is None:
            step_f = 1.0
        else:
            step_f = float(step)
        if not math.isfinite(step_f):
            step_f = 1.0
        return max(step_f, 1.0)

    @staticmethod
    def _corrections(gh: _GroupHyper, step_f: float) -> tuple[float, float]:
        bias_correction1 = 1.0 - (gh.beta1**step_f)
        bias_correction2 = 1.0 - (gh.beta2**step_f)
        c_t = gh.lr * (1.0 - gh.beta1) / max(bias_correction1, 1e-12)
        return c_t, bias_correction2

    def _finite_or_fallback(
        self, p: torch.nn.Parameter, out: torch.Tensor
    ) -> torch.Tensor:
        if out.shape != p.data.shape:
            return self._scalar_fallback(p)
        if not bool(torch.isfinite(out).all()):
            return self._scalar_fallback(p)
        return out

    def get(self, p: torch.nn.Parameter) -> torch.Tensor:
        pid = id(p)
        if pid in self._cache:
            return self._cache[pid]

        gh = self._param_to_group.get(pid)
        if gh is None:
            out = (
                torch.zeros_like(p.data, dtype=torch.float32)
                if self.strict_shard_only
                else torch.ones_like(p.data, dtype=torch.float32)
            )
            self._cache[pid] = out
            return out

        state = self.optimizer.state.get(p, None)
        if not state:
            out = self._scalar_fallback(p)
            self._cache[pid] = out
            return out

        exp_avg_sq = state.get("exp_avg_sq", None)
        step = state.get("step", None)
        if exp_avg_sq is None:
            out = self._scalar_fallback(p)
            self._cache[pid] = out
            return out

        step_f = self._resolve_step(step)
        c_t, bias_correction2 = self._corrections(gh, step_f)

        v_hat = exp_avg_sq.to(torch.float32) / max(bias_correction2, 1e-12)
        v_hat = torch.nan_to_num(v_hat, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0.0)
        out = c_t / (torch.sqrt(v_hat) + gh.eps)
        out = self._finite_or_fallback(p, out)

        self._cache[pid] = out
        return out

    def get_slice(self, p: torch.nn.Parameter, index) -> torch.Tensor:
        """
        Return a preconditioner slice for `p[index]` without materializing full tensor preconditioner.
        Falls back to scalar C_t scaling if optimizer state is unavailable or incompatible.
        """
        if torch.is_tensor(index):
            idx_int = int(index.item())
        else:
            idx_int = int(index)
        key = (id(p), idx_int)
        if key in self._slice_cache:
            return self._slice_cache[key]

        ref = p.data[index]
        gh = self._param_to_group.get(id(p), None)
        if gh is None:
            out = (
                torch.zeros_like(ref, dtype=torch.float32)
                if self.strict_shard_only
                else torch.ones_like(ref, dtype=torch.float32)
            )
            self._slice_cache[key] = out
            return out

        state = self.optimizer.state.get(p, None)
        if not state:
            out = self._scalar_fallback_like(p, ref)
            self._slice_cache[key] = out
            return out

        exp_avg_sq = state.get("exp_avg_sq", None)
        step = state.get("step", None)
        if exp_avg_sq is None:
            out = self._scalar_fallback_like(p, ref)
            self._slice_cache[key] = out
            return out

        try:
            exp_avg_sq_slice = exp_avg_sq[index]
        except Exception:
            out = self._scalar_fallback_like(p, ref)
            self._slice_cache[key] = out
            return out

        step_f = self._resolve_step(step)
        c_t, bias_correction2 = self._corrections(gh, step_f)

        v_hat = exp_avg_sq_slice.to(torch.float32) / max(bias_correction2, 1e-12)
        v_hat = torch.nan_to_num(v_hat, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0.0)
        out = c_t / (torch.sqrt(v_hat) + gh.eps)

        if out.shape != ref.shape or not bool(torch.isfinite(out).all()):
            out = self._scalar_fallback_like(p, ref)
        self._slice_cache[key] = out
        return out

    def get_slices(self, p: torch.nn.Parameter, indices: torch.Tensor) -> torch.Tensor:
        """
        Batched slice accessor for expert-indexed tensors.
        Returns tensor of shape [len(indices), *p.shape[1:]] in float32.
        """
        if indices.dim() != 1:
            raise ValueError(f"indices must be 1D, got shape={tuple(indices.shape)}")
        if indices.numel() == 0:
            return torch.empty((0, *p.shape[1:]), device=p.device, dtype=torch.float32)

        out = []
        for i in indices.tolist():
            out.append(self.get_slice(p, int(i)))
        return torch.stack(out, dim=0)

    def get_factored(self, p: torch.nn.Parameter) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return a rank-1 Kronecker-factored approximation of the diagonal preconditioner.

        For weight W ∈ R^{out_dim × in_dim}, the full preconditioner is:
            P[i,j] = C_t / (sqrt(v_hat[i,j]) + eps)

        We factor this as P[i,j] ≈ p_row[i] * p_col[j] where:
            p_row[i] = sqrt(mean_j(P[i,j]))   — RMS along columns
            p_col[j] = sqrt(mean_i(P[i,j]))   — RMS along rows
        normalized so that mean(p_row) * mean(p_col) ≈ mean(P).

        Returns:
            (p_row, p_col): tuple of [out_dim] and [in_dim] tensors
        """
        P = self.get(p)  # [out_dim, in_dim]
        if P.dim() != 2:
            # Non-matrix parameter: return scalars
            val = P.mean().sqrt()
            return (
                torch.full(
                    (P.shape[0],), float(val), device=P.device, dtype=torch.float32
                ),
                torch.ones(1, device=P.device, dtype=torch.float32),
            )

        # RMS-based factoring: p_row * p_col ≈ P
        p_row = P.mean(dim=1).clamp_min(1e-12).sqrt()  # [out_dim]
        p_col = P.mean(dim=0).clamp_min(1e-12).sqrt()  # [in_dim]

        # Normalize so that mean(p_row)*mean(p_col) ≈ mean(P)
        global_mean = P.mean().clamp_min(1e-12)
        factor_mean = p_row.mean() * p_col.mean()
        if factor_mean > 1e-12:
            scale = (global_mean / factor_mean).sqrt()
            p_row = p_row * scale
            p_col = p_col * scale

        p_row = torch.nan_to_num(p_row, nan=1.0, posinf=1.0, neginf=1.0)
        p_col = torch.nan_to_num(p_col, nan=1.0, posinf=1.0, neginf=1.0)
        return p_row, p_col
