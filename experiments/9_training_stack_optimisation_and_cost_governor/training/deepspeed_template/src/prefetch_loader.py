"""
GPU-prefetching DataLoader wrapper with dedicated CUDA stream.

Overlaps host-to-device (H2D) transfer of batch N+1 with GPU compute on
batch N by using a separate CUDA stream for the transfers.  This hides
PCIe/NVLink transfer latency behind forward + backward computation.

Usage::

    from src.prefetch_loader import PrefetchDataLoader

    loader = PrefetchDataLoader(
        base_loader,
        device=torch.device("cuda:0"),
        prefetch_depth=2,
    )
    for batch in loader:
        # batch tensors are already on `device`, ready for compute
        loss = model(**batch)

The wrapper is composable with any iterable (DataLoader, SPDL pipeline,
streaming datasets) and is backward-compatible: setting ``prefetch_depth=0``
disables prefetching and falls back to synchronous transfer.
"""

import threading
from collections import deque
from typing import Any, Dict, Iterable

import torch


def _to_device(
    batch: Dict[str, Any],
    device: torch.device,
    stream: torch.cuda.Stream,
) -> Dict[str, Any]:
    """Move every tensor in *batch* to *device* on the given CUDA *stream*."""
    moved = {}
    with torch.cuda.stream(stream):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device, non_blocking=True)
            else:
                moved[key] = value
    return moved


class PrefetchDataLoader:
    """
    Wraps an iterable (typically a ``DataLoader``) and asynchronously
    transfers upcoming batches to a CUDA device on a dedicated stream.

    The prefetch thread pulls batches from the underlying iterable and
    posts H2D copies on ``self._stream``.  The main training thread
    consumes batches from a bounded queue, calling
    ``self._stream.synchronize()`` just before yielding so the tensors
    are guaranteed resident on the device.

    Args:
        base_loader: Any iterable that yields ``Dict[str, Tensor]`` batches.
        device: Target CUDA device (e.g., ``torch.device("cuda:0")``).
        prefetch_depth: Number of batches to keep prefetched ahead of
            consumption.  ``0`` disables prefetching (synchronous fallback).
            Values of 2–3 are recommended for typical training loops.
    """

    def __init__(
        self,
        base_loader: Iterable,
        device: torch.device,
        prefetch_depth: int = 2,
    ):
        if not isinstance(device, torch.device):
            device = torch.device(device)

        self.base_loader = base_loader
        self.device = device
        self.prefetch_depth = max(0, prefetch_depth)

        # Expose common DataLoader attributes for compatibility
        # (e.g., tqdm length estimation, DistributedSampler access)
        self.dataset = getattr(base_loader, "dataset", None)
        self.sampler = getattr(base_loader, "sampler", None)
        self.batch_size = getattr(base_loader, "batch_size", None)
        self.num_workers = getattr(base_loader, "num_workers", None)
        self.pin_memory = getattr(base_loader, "pin_memory", None)

    def __len__(self):
        """Delegate to base_loader.__len__ if available."""
        return len(self.base_loader)

    def __iter__(self):
        if self.prefetch_depth == 0 or not torch.cuda.is_available():
            # Synchronous fallback — no threading, no stream
            yield from self._iter_sync()
        else:
            yield from self._iter_prefetch()

    # ------------------------------------------------------------------
    # Synchronous path (prefetch_depth=0 or CPU-only)
    # ------------------------------------------------------------------

    def _iter_sync(self):
        """Transfer each batch synchronously on the default stream."""
        for batch in self.base_loader:
            moved = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    moved[key] = value.to(self.device, non_blocking=True)
                else:
                    moved[key] = value
            yield moved

    # ------------------------------------------------------------------
    # Prefetch path (dedicated CUDA stream + background thread)
    # ------------------------------------------------------------------

    def _iter_prefetch(self):
        """
        Background thread fills a bounded deque with GPU-resident batches.
        Main thread pops from the front after synchronizing the stream.
        """
        stream = torch.cuda.Stream(device=self.device)
        queue: deque = deque()
        condition = threading.Condition()
        finished = threading.Event()
        error_container: list = []

        def _producer():
            try:
                for batch in self.base_loader:
                    moved = _to_device(batch, self.device, stream)
                    # Record an event so the consumer can synchronize with
                    # exactly the right point on the transfer stream.
                    event = stream.record_event()
                    with condition:
                        # Block if the queue is full
                        while (
                            len(queue) >= self.prefetch_depth and not finished.is_set()
                        ):
                            condition.wait(timeout=0.1)
                        if finished.is_set():
                            break
                        queue.append((moved, event))
                        condition.notify()
            except Exception as exc:
                error_container.append(exc)
            finally:
                finished.set()
                with condition:
                    condition.notify_all()

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()

        try:
            while True:
                with condition:
                    # Wait for data or producer completion
                    while len(queue) == 0 and not finished.is_set():
                        condition.wait(timeout=0.1)

                    if len(queue) == 0:
                        # Producer finished and queue is drained
                        break

                    batch, event = queue.popleft()
                    condition.notify()  # unblock producer if it was waiting

                # GPU synchronization: make the default stream wait for the H2D transfer
                torch.cuda.current_stream().wait_event(event)

                # Tell the allocator about the tensors' use on the default stream
                # so memory is not freed before the compute kernels finish.
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        value.record_stream(torch.cuda.current_stream())

                yield batch

        finally:
            finished.set()
            thread.join(timeout=5.0)

        # Re-raise any exception from the producer thread
        if error_container:
            raise error_container[0]
