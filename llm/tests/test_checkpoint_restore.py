"""
Unit tests for verify_optimizer_scheduler_restored.

Tests the actual production function from llm.utils with controlled inputs.
No GPU or DeepSpeed required.

Usage:
    cd llm
    uv run python -m pytest tests/test_checkpoint_restore.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from llm.utils import verify_optimizer_scheduler_restored


class TinyModel(nn.Module):
    def __init__(self, d: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _train_steps(model, optimizer, scheduler, n_steps: int = 5, d: int = 32):
    for _ in range(n_steps):
        x = torch.randn(1, d)
        target = torch.randn(1, d)
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


class TestVerifyOptimizerSchedulerRestored:
    """Tests for the production verify_optimizer_scheduler_restored function."""

    def test_passes_after_round_trip(self, tmp_path):
        """Function passes when optimizer/scheduler state survives save → load."""
        ckpt_path = str(tmp_path / "ckpt.pt")

        model = TinyModel()
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        _train_steps(model, optimizer, scheduler, n_steps=5)

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            ckpt_path,
        )

        model2 = TinyModel()
        optimizer2 = optim.AdamW(model2.parameters(), lr=3e-4, betas=(0.9, 0.95))
        scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=100)

        ckpt = torch.load(ckpt_path, weights_only=False)
        model2.load_state_dict(ckpt["model"])
        optimizer2.load_state_dict(ckpt["optimizer"])
        scheduler2.load_state_dict(ckpt["scheduler"])

        result = verify_optimizer_scheduler_restored(optimizer2, scheduler2, expected_global_step=5)

        assert result["restored_count"] == result["total_count"]
        assert result["restored_count"] > 0
        assert result["last_epoch"] == 5
        assert result["current_lr"] is not None

    def test_raises_on_empty_state(self):
        """Raises RuntimeError when optimizer has no state at all."""
        model = TinyModel()
        optimizer = optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        with pytest.raises(RuntimeError, match="Optimizer state is empty"):
            verify_optimizer_scheduler_restored(optimizer, scheduler, expected_global_step=10)

    def test_raises_on_zero_buffers_after_step_2(self):
        """Raises when all momentum/variance buffers are zeros at step > 2."""
        model = TinyModel()
        optimizer = optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        for param in model.parameters():
            optimizer.state[param] = {
                "exp_avg": torch.zeros_like(param),
                "exp_avg_sq": torch.zeros_like(param),
                "step": torch.tensor(10.0),
            }

        with pytest.raises(RuntimeError, match="non-zero momentum/variance"):
            verify_optimizer_scheduler_restored(optimizer, scheduler, expected_global_step=10)

    def test_tolerates_zero_buffers_at_early_steps(self):
        """At global_step <= 2, all-zero buffers should NOT raise."""
        model = TinyModel()
        optimizer = optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        for param in model.parameters():
            optimizer.state[param] = {
                "exp_avg": torch.zeros_like(param),
                "exp_avg_sq": torch.zeros_like(param),
                "step": torch.tensor(1.0),
            }

        result = verify_optimizer_scheduler_restored(optimizer, scheduler, expected_global_step=1)
        assert result["restored_count"] == 0
        assert result["total_count"] > 0

    def test_lr_preserved_after_restore(self, tmp_path):
        """LR should not reset to initial value after restore."""
        ckpt_path = str(tmp_path / "ckpt.pt")

        model = TinyModel()
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        _train_steps(model, optimizer, scheduler, n_steps=20)

        lr_at_save = scheduler.get_last_lr()[0]

        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, ckpt_path)

        model2 = TinyModel()
        optimizer2 = optim.AdamW(model2.parameters(), lr=3e-4, betas=(0.9, 0.95))
        scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=200)

        ckpt = torch.load(ckpt_path, weights_only=False)
        optimizer2.load_state_dict(ckpt["optimizer"])
        scheduler2.load_state_dict(ckpt["scheduler"])

        result = verify_optimizer_scheduler_restored(optimizer2, scheduler2, expected_global_step=20)

        assert result["current_lr"] == pytest.approx(lr_at_save, rel=1e-6)
        assert result["last_epoch"] == 20

    def test_scheduler_none_accepted(self):
        """Function works when lr_scheduler is None."""
        model = TinyModel()
        optimizer = optim.AdamW(model.parameters(), lr=3e-4)
        _train_steps(
            model,
            optimizer,
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100),
            n_steps=5,
        )

        result = verify_optimizer_scheduler_restored(optimizer, None, expected_global_step=5)

        assert result["restored_count"] > 0
        assert result["current_lr"] is None
        assert result["last_epoch"] is None
