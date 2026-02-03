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
