from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models.recurrence_model_70b as m70


def test_moe_grouped_backend_fail_fast_when_required(monkeypatch):
    monkeypatch.setattr(m70, "HAS_MOE_GROUPED_GEMM", False)
    monkeypatch.setattr(m70, "moe_grouped_gemm", None)
    with pytest.raises(RuntimeError, match="MoE grouped_gemm backend was requested but is unavailable"):
        m70.MoEFFN(
            d_model=16,
            d_hidden=8,
            num_experts=4,
            top_k=1,
            data_sparsity=1.0,
            moe_backend="grouped_gemm",
            require_fused_kernel=True,
        )


def test_moe_auto_backend_falls_back_to_vectorized(monkeypatch):
    monkeypatch.setattr(m70, "HAS_MOE_GROUPED_GEMM", False)
    monkeypatch.setattr(m70, "moe_grouped_gemm", None)
    layer = m70.MoEFFN(
        d_model=16,
        d_hidden=8,
        num_experts=4,
        top_k=1,
        data_sparsity=1.0,
        moe_backend="auto",
        require_fused_kernel=False,
    )
    assert layer.active_moe_backend == "vectorized"


def test_moe_grouped_backend_path_executes_with_kernel(monkeypatch):
    calls = {"count": 0}

    def fake_grouped_gemm(a: torch.Tensor, b: torch.Tensor, m_sizes) -> torch.Tensor:
        calls["count"] += 1
        out = []
        start = 0
        for e, m in enumerate([int(x) for x in m_sizes]):
            if m <= 0:
                continue
            chunk = a[start:start + m]
            out.append(chunk @ b[e])
            start += m
        if out:
            return torch.cat(out, dim=0)
        return a.new_empty((0, b.shape[-1]))

    monkeypatch.setattr(m70, "HAS_MOE_GROUPED_GEMM", True)
    monkeypatch.setattr(m70, "moe_grouped_gemm", fake_grouped_gemm)

    layer = m70.MoEFFN(
        d_model=16,
        d_hidden=8,
        num_experts=4,
        top_k=1,
        data_sparsity=1.0,
        moe_backend="grouped_gemm",
        require_fused_kernel=True,
    )
    x = torch.randn(2, 3, 16)
    y, aux_loss = layer(x)
    assert y.shape == x.shape
    assert aux_loss.ndim == 0
    # gate/up/down grouped GEMM calls
    assert calls["count"] == 3
