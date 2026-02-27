from __future__ import annotations

import datetime as _dt
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist



def distributed_available() -> bool:
    return dist.is_available() and dist.is_initialized()



def get_rank() -> int:
    return dist.get_rank() if distributed_available() else 0



def get_world_size() -> int:
    return dist.get_world_size() if distributed_available() else 1



def is_rank0() -> bool:
    return get_rank() == 0



def set_seed(seed: int) -> None:
    rank = get_rank()
    seed_rank = int(seed) + rank
    random.seed(seed_rank)
    np.random.seed(seed_rank)
    torch.manual_seed(seed_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_rank)



def init_distributed(timeout_minutes: int = 30) -> None:
    if distributed_available():
        return
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        timeout=_dt.timedelta(minutes=timeout_minutes),
    )



def barrier() -> None:
    if distributed_available():
        dist.barrier()



def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if distributed_available():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_reduce_sum_async(tensor: torch.Tensor):
    if distributed_available():
        return dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
    return None



def all_reduce_max(tensor: torch.Tensor) -> torch.Tensor:
    if distributed_available():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor



def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    if distributed_available():
        dist.broadcast(tensor, src=src)
    return tensor



def all_gather_1d(local: torch.Tensor) -> torch.Tensor:
    if local.dim() != 1:
        raise ValueError(f"Expected 1D tensor, got shape {tuple(local.shape)}")
    if not distributed_available():
        return local
    parts = [torch.empty_like(local) for _ in range(get_world_size())]
    dist.all_gather(parts, local)
    return torch.cat(parts, dim=0)
