"""
Streaming Token Dataset with shard-level progress tracking.

This is the core of the shard-first data loading strategy:
- Data is pre-sharded into .npy files BEFORE being fed to the DataLoader
- Each shard is a named, identifiable unit of data
- Progress is tracked at (shard_idx, seq_offset) granularity
- On failure, we know exactly which shard was active
- On restart, we resume from that exact point with the same data ordering

Key Properties:
- Zero-copy reads via np.memmap (data is paged from NVMe on demand)
- Constant memory usage (~few MB for index metadata)
- Deterministic shard ordering (sorted filenames)
- Exact checkpoint/resume at sequence granularity
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler

logger = logging.getLogger(__name__)


class StreamingTokenDataset(Dataset):
    """
    Memory-mapped dataset that reads pre-tokenized NumPy shards.

    Each shard is a `.npy` file containing a 1-D array of token IDs.
    The dataset slices each shard into non-overlapping windows of
    ``seq_length`` tokens to form individual training sequences.

    Shards are always consumed in **deterministic sorted order** by
    filename (shard-00000, shard-00001, ...). No shuffling is applied
    at the shard level — this guarantees that given the same set of
    shards and the same ``(start_shard_idx, start_seq_offset)``, the
    dataset produces the exact same sequence of samples.

    The DataLoader sits *below* this class and is responsible only
    for batching and optional within-epoch shuffling. Fault tolerance
    and data identity are handled here at the shard level.

    Args:
        shard_paths: Ordered list of paths to .npy shard files.
        seq_length: Number of tokens per training sequence.
        start_shard_idx: Shard index to resume from (from checkpoint).
        start_seq_offset: Sequence offset within the start shard (from checkpoint).
    """

    def __init__(
        self,
        shard_paths: List[str],
        seq_length: int = 4096,
        start_shard_idx: int = 0,
        start_seq_offset: int = 0,
    ):
        super().__init__()

        if not shard_paths:
            raise ValueError("shard_paths must not be empty.")
        if seq_length <= 0:
            raise ValueError(f"seq_length must be positive, got {seq_length}.")

        self.seq_length = seq_length
        self.start_shard_idx = start_shard_idx
        self.start_seq_offset = start_seq_offset

        # Store shard paths in deterministic sorted order
        self.shard_paths = sorted(shard_paths)

        # Memory-map each shard and build global index
        self._mmaps: List[np.memmap] = []
        self._global_index: List[Tuple[int, int]] = []  # (shard_idx, offset)
        self._shard_seq_counts: List[int] = []

        self._build_index()

        logger.info(
            "StreamingTokenDataset initialized: "
            "%d shards, %d total sequences (seq_length=%d), "
            "resuming from shard=%d, offset=%d",
            len(self.shard_paths),
            len(self._global_index),
            self.seq_length,
            self.start_shard_idx,
            self.start_seq_offset,
        )

    # ------------------------------------------------------------------
    # Index Construction
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """
        Memory-map each shard and build the global sequence index.

        For each shard, we compute how many full ``seq_length + 1``
        windows fit (the +1 is for the shifted label). Partial
        sequences at shard boundaries are skipped — this avoids the
        complexity of cross-shard concatenation.

        On resume, sequences before ``(start_shard_idx, start_seq_offset)``
        are excluded from the index.
        """
        self._mmaps.clear()
        self._global_index.clear()
        self._shard_seq_counts.clear()

        total_skipped = 0

        for shard_idx, path in enumerate(self.shard_paths):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Shard file not found: {path}")

            # Memory-map the shard (read-only, zero-copy)
            mmap = np.load(path, mmap_mode="r")

            if mmap.ndim != 1:
                raise ValueError(
                    f"Shard {os.path.basename(path)} has shape {mmap.shape}; "
                    f"expected a 1-D array of token IDs."
                )

            self._mmaps.append(mmap)

            # Number of full sequences in this shard
            # We need seq_length + 1 tokens per sequence (input + shifted label)
            num_tokens = len(mmap)
            tokens_per_seq = self.seq_length + 1
            num_sequences = num_tokens // tokens_per_seq

            self._shard_seq_counts.append(num_sequences)

            # Skip consumed shards/offsets on resume
            for seq_idx in range(num_sequences):
                if shard_idx < self.start_shard_idx:
                    total_skipped += 1
                    continue
                if shard_idx == self.start_shard_idx and seq_idx < self.start_seq_offset:
                    total_skipped += 1
                    continue

                token_offset = seq_idx * tokens_per_seq
                self._global_index.append((shard_idx, token_offset))

        if total_skipped > 0:
            logger.info(
                "Skipped %d already-consumed sequences (resume from shard=%d, offset=%d)",
                total_skipped,
                self.start_shard_idx,
                self.start_seq_offset,
            )

        if not self._global_index:
            logger.warning(
                "No sequences available after resume offset. "
                "All shards may have been consumed."
            )

    # ------------------------------------------------------------------
    # Dataset Interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Total number of available sequences (after resume offset)."""
        return len(self._global_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sequence by global index.

        Returns:
            Dictionary with:
            - ``input_ids``: Token IDs for input (shape: [seq_length])
            - ``labels``: Token IDs shifted by 1 for causal LM (shape: [seq_length])
            - ``attention_mask``: All ones (shape: [seq_length])
        """
        if idx < 0 or idx >= len(self._global_index):
            raise IndexError(
                f"Index {idx} out of range for dataset with "
                f"{len(self._global_index)} sequences."
            )

        shard_idx, token_offset = self._global_index[idx]
        mmap = self._mmaps[shard_idx]

        # Extract seq_length + 1 tokens (for input + label shift)
        tokens = mmap[token_offset : token_offset + self.seq_length + 1]
        tokens = np.array(tokens, dtype=np.int64)  # Copy from mmap

        input_ids = torch.from_numpy(tokens[:-1])   # [0 .. seq_length-1]
        labels = torch.from_numpy(tokens[1:])        # [1 .. seq_length]
        attention_mask = torch.ones(self.seq_length, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    # ------------------------------------------------------------------
    # Progress Tracking (for checkpointing)
    # ------------------------------------------------------------------

    def get_progress(self, consumed_sequences: int) -> Tuple[int, int]:
        """
        Compute the current (shard_idx, seq_offset) given how many
        sequences have been consumed so far.

        This is saved into the training checkpoint so that on restart,
        we can resume from the exact same position.

        Args:
            consumed_sequences: Number of sequences consumed during
                this training session (not counting skipped ones).

        Returns:
            Tuple of (current_shard_idx, current_seq_offset) representing
            the next sequence to consume.
        """
        if consumed_sequences <= 0:
            return (self.start_shard_idx, self.start_seq_offset)

        if consumed_sequences >= len(self._global_index):
            # All sequences consumed — return end marker
            last_shard_idx = len(self.shard_paths) - 1
            last_shard_seqs = self._shard_seq_counts[last_shard_idx]
            return (last_shard_idx, last_shard_seqs)

        # Look up the position of the next unconsumed sequence
        shard_idx, token_offset = self._global_index[consumed_sequences]
        tokens_per_seq = self.seq_length + 1
        seq_offset = token_offset // tokens_per_seq

        return (shard_idx, seq_offset)

    def get_current_shard_info(self, idx: int) -> Dict[str, object]:
        """
        Get details about which shard a given index maps to.

        Useful for logging and debugging.

        Args:
            idx: Global sequence index.

        Returns:
            Dictionary with shard_idx, shard_name, token_offset, seq_in_shard.
        """
        if idx < 0 or idx >= len(self._global_index):
            return {"error": f"Index {idx} out of range"}

        shard_idx, token_offset = self._global_index[idx]
        tokens_per_seq = self.seq_length + 1
        seq_in_shard = token_offset // tokens_per_seq

        return {
            "shard_idx": shard_idx,
            "shard_name": os.path.basename(self.shard_paths[shard_idx]),
            "token_offset": token_offset,
            "seq_in_shard": seq_in_shard,
            "total_seqs_in_shard": self._shard_seq_counts[shard_idx],
        }

    # ------------------------------------------------------------------
    # Shard Management
    # ------------------------------------------------------------------

    def refresh_shards(self, new_shard_paths: List[str]) -> None:
        """
        Replace the current shard list with a new one.

        Called at shard boundaries when the background stager has
        downloaded additional shards. The current progress is preserved.

        Args:
            new_shard_paths: Updated ordered list of shard paths.
        """
        # Get current progress before refresh
        current_progress = self.get_progress(0)

        # Close existing mmaps
        self._mmaps.clear()

        # Update shard paths and rebuild index
        self.shard_paths = sorted(new_shard_paths)
        self.start_shard_idx = current_progress[0]
        self.start_seq_offset = current_progress[1]

        self._build_index()

        logger.info(
            "Shard list refreshed: now %d shards, %d sequences available",
            len(self.shard_paths),
            len(self._global_index),
        )

    @property
    def num_shards(self) -> int:
        """Number of shards in the dataset."""
        return len(self.shard_paths)

    @property
    def total_tokens(self) -> int:
        """Total number of tokens across all mounted shards."""
        return sum(len(m) for m in self._mmaps)


def create_distributed_sampler(
    dataset: StreamingTokenDataset,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
    shuffle: bool = False,
) -> DistributedSampler:
    """
    Create a DistributedSampler for the streaming dataset.

    Each GPU rank sees a disjoint subset of sequences. Shuffle is
    disabled by default to maintain deterministic ordering for
    reproducibility, but can be enabled for within-epoch randomness.

    Args:
        dataset: The StreamingTokenDataset instance.
        num_replicas: Number of processes (default: world size).
        rank: Current process rank (default: current rank).
        shuffle: Whether to shuffle indices within epoch.

    Returns:
        Configured DistributedSampler.
    """
    return DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        drop_last=True,  # Drop incomplete last batch for uniform batch sizes
    )
