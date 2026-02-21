import logging
import os

import numpy as np
import torch
from common import BATCH_SIZE, DTYPE, NUM_THREADS, PREFETCH_BUFFER, SEQUENCE_LENGTH
from spdl.pipeline import PipelineBuilder

logger = logging.getLogger("spdl.loader")


# --- Helper Functions ---
def read_idx(idx_path):
    """
    Read the .idx file and return an array of offsets.
    The first 8 bytes are header/version (ignored).
    """
    logger.debug(f"Reading idx file: {idx_path}")
    with open(idx_path, "rb") as f:
        _ = f.read(8)  # Skip header
        offsets = np.frombuffer(f.read(), dtype=np.uint64)
    logger.debug(f"Offsets loaded: {len(offsets)} entries")
    return offsets


def _should_skip_region(bin_path, i, start, end, itemsize, seq_len, logger):
    """
    Helper to check if a region in the bin/idx file should be skipped.
    Returns True if region should be skipped.
    """
    num_bytes = end - start
    if num_bytes <= 0:
        logger.warning(f"Corrupt/empty region: {bin_path} [{i}] {start}-{end}")
        return True
    num_tokens = num_bytes // itemsize
    if num_tokens == 0:
        logger.warning(f"Zero tokens: {bin_path} [{i}] {start}-{end}")
        return True
    num_full = num_tokens // seq_len
    if num_full == 0:
        logger.warning(f"Incomplete sequence: {bin_path} [{i}] {num_tokens} tokens")
        return True
    return False


# --- Main Data Loading Logic ---
def load_tokens_from_bin_idx(bin_path, idx_path, dtype=None, seq_len=None):
    """
    Yield token sequences from a .bin file using .idx offsets.
    Each yielded tensor is of shape [seq_len].
    """
    dtype = np.dtype(dtype) if dtype is not None else np.dtype(DTYPE)
    seq_len = seq_len if seq_len is not None else SEQUENCE_LENGTH
    offsets = read_idx(idx_path)
    itemsize = dtype.itemsize
    logger.info(f"Loading tokens from: {bin_path} with {len(offsets)-1} sequences")
    with open(bin_path, "rb") as f:
        for i in range(len(offsets) - 1):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            if _should_skip_region(bin_path, i, start, end, itemsize, seq_len, logger):
                continue
            num_bytes = end - start
            num_tokens = num_bytes // itemsize
            num_full = num_tokens // seq_len
            read_tokens = num_full * seq_len
            f.seek(start)
            tokens = np.frombuffer(f.read(read_tokens * itemsize), dtype=dtype)
            if tokens.size != read_tokens:
                logger.warning(
                    f"Incomplete read: {bin_path} [{i}] expected {read_tokens}, got {tokens.size}"
                )
                continue
            for j in range(0, len(tokens), seq_len):
                logger.debug(f"Yielding sequence {j//seq_len} from {bin_path}")
                yield torch.from_numpy(tokens[j : j + seq_len])


def bin_idx_source(shard_dir, seq_len=None, dtype=None):
    """
    Generator yielding token sequences from all .bin/.idx shards in a directory.
    """
    dtype = np.dtype(dtype) if dtype is not None else np.dtype(DTYPE)
    seq_len = seq_len if seq_len is not None else SEQUENCE_LENGTH
    files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".bin"))
    logger.info(f"Found {len(files)} .bin files in {shard_dir}")
    for bin_file in files:
        bin_path = os.path.join(shard_dir, bin_file)
        idx_path = bin_path.replace(".bin", ".idx")
        logger.info(f"Processing shard: {bin_path}")
        yield from load_tokens_from_bin_idx(
            bin_path, idx_path, dtype=dtype, seq_len=seq_len
        )


def build_pipeline(shard_dir, seq_len=None, dtype=None):
    """
    Build and return a SPDL pipeline for .bin/.idx token shards.
    """
    return (
        PipelineBuilder()
        .add_source(bin_idx_source(shard_dir, seq_len=seq_len, dtype=dtype))
        .aggregate(BATCH_SIZE)
        .add_sink(PREFETCH_BUFFER)
        .build(num_threads=NUM_THREADS)
    )


# --- Dummy Model (for testing) ---
class DummyModel(torch.nn.Module):
    """
    Dummy model for testing: sums across dim=1.
    """

    def forward(self, x):
        return x.sum(dim=1)
