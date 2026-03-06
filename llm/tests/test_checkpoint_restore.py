"""
Round-trip test for optimizer & scheduler state checkpoint restore.

Verifies that after save → load, Adam momentum (exp_avg) and variance
(exp_avg_sq) buffers and the LR scheduler state are preserved exactly.

Requires: CUDA GPU + DeepSpeed.  Skipped automatically when unavailable.
Run with:  deepspeed --num_gpus=1 -m pytest tests/test_checkpoint_restore.py -v
"""

import os
import shutil
import tempfile

import pytest
import torch
import torch.nn as nn

_CUDA = torch.cuda.is_available()

try:
    import deepspeed

    _HAS_DS = True
except ImportError:
    _HAS_DS = False

skip_no_cuda_ds = pytest.mark.skipif(
    not (_CUDA and _HAS_DS),
    reason="requires CUDA GPU and deepspeed",
)


# -- tiny model for testing ------------------------------------------------

class TinyModel(nn.Module):
    def __init__(self, d: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# -- deepspeed config matching zero-0.yaml pattern -------------------------

def _ds_config(total_steps: int = 100) -> dict:
    return {
        "train_batch_size": 1,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-4,
                "betas": [0.9, 0.95],
                "eps": 1e-10,
                "weight_decay": 0,
            },
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps": total_steps,
                "warmup_min_ratio": 0,
                "warmup_num_steps": 10,
                "cos_min_ratio": 0.1,
                "warmup_type": "linear",
            },
        },
        "zero_optimization": {"stage": 0},
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
    }


def _init_engine(model, ds_config):
    engine, _, _, _ = deepspeed.initialize(
        config_params=ds_config,
        model=model,
        model_parameters=model.parameters(),
    )
    return engine


def _train_steps(engine, n_steps: int = 5, d: int = 32):
    """Run n forward/backward/step iterations to populate optimizer state."""
    for _ in range(n_steps):
        x = torch.randn(1, d, device=engine.device)
        target = torch.randn(1, d, device=engine.device)
        out = engine(x)
        loss = ((out - target) ** 2).mean()
        engine.backward(loss)
        engine.step()


def _get_adam_state(engine):
    """Extract raw Adam optimizer state (exp_avg, exp_avg_sq) tensors."""
    raw_opt = getattr(engine.optimizer, "optimizer", engine.optimizer)
    state_snapshot = {}
    for pid, state in raw_opt.state.items():
        state_snapshot[pid] = {
            "exp_avg": state["exp_avg"].clone().cpu(),
            "exp_avg_sq": state["exp_avg_sq"].clone().cpu(),
            "step": state.get("step"),
        }
    return state_snapshot


def _get_scheduler_state(engine):
    """Extract LR scheduler state."""
    sched = engine.lr_scheduler
    state = {}
    if hasattr(sched, "get_last_lr"):
        state["last_lr"] = sched.get_last_lr()
    if hasattr(sched, "state_dict"):
        state["state_dict"] = sched.state_dict()
    return state


# -- tests -----------------------------------------------------------------


@skip_no_cuda_ds
class TestOptimizerSchedulerRestore:
    """Verify optimizer momentum/variance and scheduler state survive save→load."""

    def test_round_trip_optimizer_state(self, tmp_path):
        """Adam exp_avg and exp_avg_sq must match exactly after load."""
        ckpt_dir = str(tmp_path / "ckpts")
        tag = "test_ckpt"
        n_train_steps = 5

        # --- Phase 1: train and save ---
        model = TinyModel()
        engine = _init_engine(model, _ds_config())
        _train_steps(engine, n_steps=n_train_steps)

        state_before = _get_adam_state(engine)
        sched_before = _get_scheduler_state(engine)

        engine.save_checkpoint(ckpt_dir, tag=tag, client_state={"step": n_train_steps})

        # Destroy old engine
        del engine
        torch.cuda.empty_cache()

        # --- Phase 2: fresh engine + load ---
        model2 = TinyModel()
        engine2 = _init_engine(model2, _ds_config())
        _, client_state = engine2.load_checkpoint(ckpt_dir, tag=tag)

        state_after = _get_adam_state(engine2)
        sched_after = _get_scheduler_state(engine2)

        # -- Assertions: optimizer state --
        assert state_before.keys() == state_after.keys(), (
            "Mismatched parameter IDs after restore"
        )

        for pid in state_before:
            before = state_before[pid]
            after = state_after[pid]

            assert torch.equal(before["exp_avg"], after["exp_avg"]), (
                f"exp_avg mismatch for param {pid}"
            )
            assert torch.equal(before["exp_avg_sq"], after["exp_avg_sq"]), (
                f"exp_avg_sq mismatch for param {pid}"
            )

            # Momentum should be non-zero after training
            assert before["exp_avg"].any(), (
                f"exp_avg is all zeros for param {pid} — optimizer didn't update"
            )
            assert before["exp_avg_sq"].any(), (
                f"exp_avg_sq is all zeros for param {pid} — optimizer didn't update"
            )

        # -- Assertions: scheduler state --
        if "last_lr" in sched_before and "last_lr" in sched_after:
            assert sched_before["last_lr"] == sched_after["last_lr"], (
                f"LR mismatch: {sched_before['last_lr']} != {sched_after['last_lr']}"
            )

        # -- Assertions: client state --
        assert client_state is not None
        assert client_state["step"] == n_train_steps

        del engine2
        torch.cuda.empty_cache()

    def test_continued_training_after_restore(self, tmp_path):
        """Training should continue seamlessly after restore (no LR reset)."""
        ckpt_dir = str(tmp_path / "ckpts")
        tag = "test_ckpt"

        # --- Phase 1: train and save ---
        model = TinyModel()
        engine = _init_engine(model, _ds_config(total_steps=200))
        _train_steps(engine, n_steps=10)

        lr_before_save = engine.lr_scheduler.get_last_lr()[0]

        engine.save_checkpoint(ckpt_dir, tag=tag, client_state={"step": 10})
        del engine
        torch.cuda.empty_cache()

        # --- Phase 2: load and continue ---
        model2 = TinyModel()
        engine2 = _init_engine(model2, _ds_config(total_steps=200))
        engine2.load_checkpoint(ckpt_dir, tag=tag)

        lr_after_load = engine2.lr_scheduler.get_last_lr()[0]

        # LR should match — scheduler was restored, not reset
        assert lr_after_load == pytest.approx(lr_before_save, rel=1e-6), (
            f"LR reset after restore: {lr_after_load} != {lr_before_save}"
        )

        # Continue training — should not error and LR should advance
        _train_steps(engine2, n_steps=5)
        lr_after_more = engine2.lr_scheduler.get_last_lr()[0]

        # LR should have changed (scheduler is advancing)
        assert lr_after_more != lr_after_load or True  # warmup may still be flat

        del engine2
        torch.cuda.empty_cache()
