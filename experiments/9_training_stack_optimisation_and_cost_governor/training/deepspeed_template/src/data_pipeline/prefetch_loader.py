"""
Prefetch DataLoader with async GPU transfer.

Wraps a standard PyTorch DataLoader to overlap data loading with
GPU computation. A background thread continuously:
1. Fetches the next batch from the DataLoader (CPU → mmap read)
2. Pins batch tensors to pinned memory
3. Transfers to GPU via a dedicated CUDA stream (non-blocking H2D copy)

The training loop simply calls ``next(prefetch_loader)`` — the batch
is already resident on GPU.

    [Worker procs: mmap read] → [Pin memory] → [CUDA stream: H2D copy] → [GPU: compute]
           overlapped                overlapped          overlapped
"""

import logging
from collections import deque
from typing import Any, Dict, Iterator, Optional

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class PrefetchDataLoader:
    """
    Async prefetching DataLoader wrapper for GPU training.

    Prefetches batches onto GPU in a background CUDA stream so that
    the main training loop never waits for host-to-device transfer.

    Args:
        dataloader: A standard PyTorch DataLoader instance.
        device: Target CUDA device (e.g., torch.device('cuda:0')).
        prefetch_depth: Number of batches to prefetch ahead (default: 2).
    """

    def __init__(
        self,
        dataloader: DataLoader,
        device: torch.device,
        prefetch_depth: int = 2,
    ):
        self.dataloader = dataloader
        self.device = device
        self.prefetch_depth = max(1, prefetch_depth)

        # CUDA stream for async H2D transfers
        self._stream: Optional[torch.cuda.Stream] = None
        if device.type == "cuda":
            self._stream = torch.cuda.Stream(device=device)

        # Internal state
        self._dataloader_iter: Optional[Iterator] = None
        self._prefetch_queue: deque = deque()
        self._exhausted = False

        # Track how many batches have been yielded (for progress tracking)
        self.batches_yielded = 0

    def __iter__(self) -> "PrefetchDataLoader":
        """Start a new iteration, resetting state and filling the prefetch queue."""
        self._dataloader_iter = iter(self.dataloader)
        self._prefetch_queue.clear()
        self._exhausted = False
        self.batches_yielded = 0

        # Fill the prefetch queue
        self._fill_queue()

        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        """
        Yield the next batch, already transferred to GPU.

        Raises:
            StopIteration: When all batches have been yielded.
        """
        if not self._prefetch_queue:
            raise StopIteration

        # Get the next prefetched batch
        batch = self._prefetch_queue.popleft()

        # Synchronize the stream to ensure transfer is complete
        if self._stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self._stream)

        # Record the batch tensors on the current stream so that
        # subsequent compute ops properly depend on the transfer
        if isinstance(batch, dict):
            for v in batch.values():
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    v.record_stream(torch.cuda.current_stream(self.device))
        elif isinstance(batch, (list, tuple)):
            for v in batch:
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    v.record_stream(torch.cuda.current_stream(self.device))

        # Prefetch the next batch to keep the queue full
        self._prefetch_one()

        self.batches_yielded += 1
        return batch

    def __len__(self) -> int:
        """Number of batches in the underlying DataLoader."""
        return len(self.dataloader)

    # ------------------------------------------------------------------
    # Prefetch Internals
    # ------------------------------------------------------------------

    def _fill_queue(self) -> None:
        """Fill the prefetch queue to the configured depth."""
        for _ in range(self.prefetch_depth):
            if not self._prefetch_one():
                break

    def _prefetch_one(self) -> bool:
        """
        Prefetch a single batch from the DataLoader onto GPU.

        Returns:
            True if a batch was prefetched, False if the iterator is exhausted.
        """
        if self._exhausted or self._dataloader_iter is None:
            return False

        try:
            batch = next(self._dataloader_iter)
        except StopIteration:
            self._exhausted = True
            return False

        # Transfer to GPU on the prefetch stream
        batch = self._to_device(batch)
        self._prefetch_queue.append(batch)
        return True

    def _to_device(self, batch: Any) -> Any:
        """
        Transfer a batch to the target device asynchronously.

        Uses a dedicated CUDA stream for non-blocking H2D transfer.
        Falls back to synchronous transfer for CPU-only setups.

        Args:
            batch: A dict, list, tuple, or tensor batch.

        Returns:
            The batch with all tensors on the target device.
        """
        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                return self._recursive_to_device(batch)
        else:
            return self._recursive_to_device(batch)

    def _recursive_to_device(self, batch: Any) -> Any:
        """
        Recursively move tensors in a batch to the target device.

        Handles dicts, lists, tuples, and raw tensors.

        Args:
            batch: Nested structure potentially containing tensors.

        Returns:
            Same structure with tensors moved to device.
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, dict):
            return {
                k: self._recursive_to_device(v) for k, v in batch.items()
            }
        elif isinstance(batch, (list, tuple)):
            transferred = [self._recursive_to_device(v) for v in batch]
            return type(batch)(transferred)
        else:
            return batch

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def dataset(self):
        """Access the underlying dataset."""
        return self.dataloader.dataset

    @property
    def batch_size(self) -> Optional[int]:
        """Batch size of the underlying DataLoader."""
        return self.dataloader.batch_size


def create_prefetch_dataloader(
    dataset,
    device: torch.device,
    batch_size: int = 8,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    prefetch_depth: int = 2,
    pin_memory: bool = True,
    sampler=None,
    shuffle: bool = False,
) -> PrefetchDataLoader:
    """
    Convenience factory to create a PrefetchDataLoader from a dataset.

    Creates the underlying PyTorch DataLoader with optimal settings
    for streaming from memory-mapped data and wraps it with async
    GPU prefetching.

    Args:
        dataset: PyTorch Dataset instance (e.g., StreamingTokenDataset).
        device: Target CUDA device.
        batch_size: Batch size.
        num_workers: DataLoader worker processes.
        prefetch_factor: DataLoader prefetch factor (per worker).
        prefetch_depth: GPU prefetch queue depth.
        pin_memory: Whether to use pinned memory (should be True for GPU).
        sampler: Optional sampler (e.g., DistributedSampler).
        shuffle: Shuffle data (mutually exclusive with sampler).

    Returns:
        Configured PrefetchDataLoader.
    """
    # When using a sampler, shuffle must be False
    if sampler is not None:
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and device.type == "cuda",
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        sampler=sampler,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    return PrefetchDataLoader(
        dataloader=dataloader,
        device=device,
        prefetch_depth=prefetch_depth,
    )
