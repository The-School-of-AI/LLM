"""Tests for AdamWPreconditionerView."""
import torch
from llm.opus.preconditioner import AdamWPreconditionerView


def test_preconditioner_scalar_fallback():
    """Before first optimizer step, falls back to scalar C_t."""
    model = torch.nn.Linear(8, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    precond = AdamWPreconditionerView(opt)
    p = next(model.parameters())
    result = precond.get(p)
    assert result is not None


def test_preconditioner_after_step():
    """After one optimizer step, returns proper preconditioner vector."""
    model = torch.nn.Linear(8, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    loss = model(torch.randn(2, 8)).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    precond = AdamWPreconditionerView(opt)
    p = next(model.parameters())
    result = precond.get(p)
    assert result.shape == p.shape
    assert torch.isfinite(result).all()


def test_preconditioner_refresh():
    """Refresh rebuilds cache from updated optimizer state."""
    model = torch.nn.Linear(8, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.randn(2, 8)).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    precond = AdamWPreconditionerView(opt)
    p = next(model.parameters())
    v1 = precond.get(p).clone()
    loss = model(torch.randn(2, 8)).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    precond.refresh()
    v2 = precond.get(p)
    assert not torch.allclose(v1, v2), "Preconditioner should change after refresh"
