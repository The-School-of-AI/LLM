"""Utility functions for training."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def is_main_process() -> bool:
    """
    Check if current process is the main process (rank 0).

    Returns:
        True if this is the main process or not in distributed mode, False otherwise
    """
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def print_rank_0(*args, **kwargs):
    """
    Print only from rank 0 process.

    This prevents duplicate output in multi-GPU setups.
    """
    if is_main_process():
        print(*args, **kwargs)


def verify_optimizer_scheduler_restored(
    optimizer,
    lr_scheduler,
    expected_global_step: int,
) -> dict:
    """
    Verify that optimizer momentum/variance and scheduler state were restored
    correctly after loading a checkpoint.

    Unwraps DeepSpeed optimizer wrappers automatically.

    Args:
        optimizer: The optimizer (raw PyTorch or DeepSpeed-wrapped).
        lr_scheduler: The LR scheduler (may be None).
        expected_global_step: The global step from the loaded checkpoint.

    Returns:
        Dict with restored_count, total_count, current_lr, last_epoch.

    Raises:
        RuntimeError: If optimizer state is empty, or if momentum/variance
            buffers are all zeros when expected_global_step > 2.
    """
    raw_optimizer = getattr(optimizer, "optimizer", optimizer)
    opt_state = raw_optimizer.state

    if not opt_state:
        raise RuntimeError(
            "Optimizer state is empty after checkpoint restore — "
            "momentum and variance buffers were not loaded."
        )

    restored_count = 0
    total_count = 0
    for _param_id, state in opt_state.items():
        total_count += 1
        has_momentum = "exp_avg" in state and state["exp_avg"].any().item()
        has_variance = "exp_avg_sq" in state and state["exp_avg_sq"].any().item()
        if has_momentum and has_variance:
            restored_count += 1

    # At early steps (<=2) Adam buffers may legitimately still be zero
    if expected_global_step > 2 and restored_count == 0:
        raise RuntimeError(
            f"Optimizer has {total_count} param states but none contain "
            "non-zero momentum/variance — restore likely failed."
        )

    current_lr = None
    last_epoch = None
    if lr_scheduler is not None:
        if hasattr(lr_scheduler, "get_last_lr"):
            current_lr = lr_scheduler.get_last_lr()[0]
        elif hasattr(lr_scheduler, "get_lr"):
            current_lr = lr_scheduler.get_lr()[0]

        if hasattr(lr_scheduler, "state_dict"):
            sched_state = lr_scheduler.state_dict()
            last_epoch = sched_state.get("last_epoch", None)

    return {
        "restored_count": restored_count,
        "total_count": total_count,
        "current_lr": current_lr,
        "last_epoch": last_epoch,
    }
