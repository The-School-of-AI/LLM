"""
Memory-mapped streaming dataset for pre-tokenized NumPy shards.

Builds a single global index across all shards and supports GPU-count-agnostic
resume via ``skip_samples``.  Works with ``DistributedSampler`` for multi-GPU
training — shard sizes can be uneven because the sampler partitions the
*index*, not the shards.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class StreamingTokenDataset(Dataset):
    """Dataset that reads pre-tokenized ``.npy`` shards via memory-mapping.

    Parameters
    ----------
    shard_paths : list[str]
        Sorted list of ``.npy`` file paths.  Each file should contain a 1-D
        ``int32``/``int64`` array of token IDs.
    seq_length : int
        Number of tokens per sequence (``input_ids`` length).
        Labels are shifted by 1, so each shard needs at least
        ``seq_length + 1`` tokens to produce one sample.
    skip_samples : int
        Number of samples (sequences) to skip from the beginning of the
        global index.  Used for **GPU-count-agnostic resume**: the checkpoint
        stores ``total_samples_consumed = global_step × batch_size × world_size``
        and passes it here.

    Notes
    -----
    * Shard ordering is **deterministic** (sorted by path / filename).
    * Partial sequences at shard boundaries are skipped — no cross-shard
      concatenation.
    * Memory usage is constant: only the index metadata (~few MB) is held
      in RAM; actual token data is paged in from NVMe on demand via ``mmap``.
    """

    def __init__(
        self,
        shard_paths: List[str],
        seq_length: int,
        skip_samples: int = 0,
    ):
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.seq_length = seq_length

        # Memory-map each shard (read-only)
        self._mmaps: List[np.ndarray] = []
        for path in self.shard_paths:
            mmap = np.load(path, mmap_mode="r")
            self._mmaps.append(mmap)

        # Build global index: [(shard_idx, token_offset), ...]
        # Each entry represents one seq_length window.  We need
        # seq_length + 1 tokens so that labels (shifted by 1) are valid.
        self._full_index: List[Tuple[int, int]] = []
        for shard_idx, mmap in enumerate(self._mmaps):
            num_tokens = len(mmap)
            # Number of complete sequences in this shard
            num_seqs = (num_tokens - 1) // self.seq_length  # -1 for label shift
            for seq_i in range(num_seqs):
                offset = seq_i * self.seq_length
                self._full_index.append((shard_idx, offset))

        total_before_skip = len(self._full_index)

        # Apply skip for resume
        if skip_samples > 0:
            if skip_samples > len(self._full_index):
                logger.warning(
                    "skip_samples (%d) > total sequences (%d) — dataset will be empty",
                    skip_samples,
                    len(self._full_index),
                )
                skip_samples = len(self._full_index)
            self._index = self._full_index[skip_samples:]
        else:
            self._index = self._full_index

        logger.info(
            "StreamingTokenDataset: %d shards, %d total sequences, "
            "skipped %d, %d remaining",
            len(self.shard_paths),
            total_before_skip,
            skip_samples,
            len(self._index),
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shard_idx, offset = self._index[idx]
        mmap = self._mmaps[shard_idx]

        # Read seq_length + 1 tokens (extra token for label shift)
        tokens = np.array(mmap[offset : offset + self.seq_length + 1], dtype=np.int64)
        input_ids = torch.from_numpy(tokens[:-1])  # [0 .. seq_length-1]
        labels = torch.from_numpy(tokens[1:])       # [1 .. seq_length]
        attention_mask = torch.ones_like(input_ids)  # All tokens are real

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    # ------------------------------------------------------------------
    # Progress / resume helpers
    # ------------------------------------------------------------------

    def get_total_samples(self) -> int:
        """Total sequences in the full (unskipped) global index."""
        return len(self._full_index)

    def get_first_active_shard_idx(self) -> int:
        """Index of the earliest shard still needed by the active index.

        Useful for telling ``S3Stager`` where to start staging on resume.
        Returns 0 if the dataset is empty after skip.
        """
        if not self._index:
            return 0
        return self._index[0][0]

    # ------------------------------------------------------------------
    # Dynamic shard refresh
    # ------------------------------------------------------------------

    def refresh_shards(self, shard_paths: List[str], skip_samples: int = 0) -> None:
        """Rebuild the dataset with a new set of shard paths.

        Called at shard boundaries when new background-downloaded shards
        become available.  Keeps ``skip_samples`` consistent.
        """
        self.__init__(shard_paths, self.seq_length, skip_samples=skip_samples)
