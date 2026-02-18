"""
Asynchronous GPU-prefetching DataLoader wrapper.

Wraps a standard ``torch.utils.data.DataLoader`` and uses a background thread
with a dedicated CUDA stream to overlap data transfer (CPU → GPU) with
compute.  The training loop just calls ``next(prefetch_loader)`` and receives
a batch that is **already on GPU**.

Pipeline::

    [DataLoader workers: mmap read] → [pin memory] → [CUDA stream: H2D] → [GPU]
          overlapped                    overlapped         overlapped
"""

import logging
import threading
from collections import deque
from typing import Any, Dict, Iterator, Optional

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class PrefetchDataLoader:
    """Wraps a ``DataLoader`` with async GPU prefetching.

    Parameters
    ----------
    loader : DataLoader
        The underlying data loader (should have ``pin_memory=True`` for best
        performance).
    device : torch.device
        Target GPU device.
    prefetch_depth : int
        Number of batches to keep prefetched ahead.  2–3 is typically enough
        to hide the data transfer latency.
    """

    def __init__(
        self,
        loader: DataLoader,
        device: Optional[torch.device] = None,
        prefetch_depth: int = 2,
    ):
        self.loader = loader
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.prefetch_depth = max(1, prefetch_depth)

        # CUDA stream for asynchronous H2D transfers
        self._stream: Optional[torch.cuda.Stream] = None
        if self.device.type == "cuda":
            self._stream = torch.cuda.Stream(device=self.device)

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return _PrefetchIterator(self.loader, self.device, self._stream, self.prefetch_depth)


class _PrefetchIterator:
    """Internal iterator that manages the prefetch queue."""

    def __init__(
        self,
        loader: DataLoader,
        device: torch.device,
        stream: Optional[torch.cuda.Stream],
        depth: int,
    ):
        self._device = device
        self._stream = stream
        self._depth = depth
        self._loader_iter = iter(loader)

        # Prefetch queue: batches already transferred to GPU
        self._queue: deque = deque()
        self._exhausted = False

        # Fill the queue up-front
        for _ in range(self._depth):
            self._prefetch_one()

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, Any]:
        # If queue is empty and loader is exhausted, stop
        if not self._queue and self._exhausted:
            raise StopIteration

        # Grab the next pre-fetched batch
        if not self._queue:
            raise StopIteration

        batch, event = self._queue.popleft()

        # Wait for the H2D transfer to finish on the current (default) stream
        if event is not None:
            event.synchronize()

        # Kick off next prefetch to keep the queue full
        self._prefetch_one()

        return batch

    def _prefetch_one(self) -> None:
        """Fetch one batch from the loader and transfer it to GPU asynchronously."""
        if self._exhausted:
            return

        try:
            batch = next(self._loader_iter)
        except StopIteration:
            self._exhausted = True
            return

        # Transfer to GPU on a separate CUDA stream
        if self._stream is not None:
            event = torch.cuda.Event()
            with torch.cuda.stream(self._stream):
                batch = self._to_device(batch)
            event.record(self._stream)
            self._queue.append((batch, event))
        else:
            # CPU fallback (no async needed)
            batch = self._to_device(batch)
            self._queue.append((batch, None))

    def _to_device(self, batch: Any) -> Any:
        """Recursively move tensors in a batch to the target device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self._device, non_blocking=True)
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(v) for v in batch)
        return batch
