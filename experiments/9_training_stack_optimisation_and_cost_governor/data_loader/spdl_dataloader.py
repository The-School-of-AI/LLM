
import os
import torch
import numpy as np
import logging
from spdl.pipeline import PipelineBuilder
from common import BATCH_SIZE, NUM_THREADS, PREFETCH_BUFFER, SEQUENCE_LENGTH, DTYPE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("spdl.loader")

def read_idx(idx_path):
    logger.debug(f"Reading idx file: {idx_path}")
    with open(idx_path, "rb") as f:
        header = f.read(8)  # 8 bytes for versioning/dtype (not used here)
        offsets = np.frombuffer(f.read(), dtype=np.uint64)
    logger.debug(f"Offsets loaded: {len(offsets)} entries")
    return offsets

def load_tokens_from_bin_idx(bin_path, idx_path, dtype=None, seq_len=None):
    logger = logging.getLogger("spdl.loader")
    """
    Loads token sequences from a .bin file using .idx for offsets.
    Returns a generator of [seq_len] token tensors.
    """
    if dtype is None:
        dtype = np.dtype(DTYPE)
    if seq_len is None:
        seq_len = SEQUENCE_LENGTH
    offsets = read_idx(idx_path)
    itemsize = np.dtype(dtype).itemsize
    logger.info(f"Loading tokens from: {bin_path} with {len(offsets)-1} sequences")
    with open(bin_path, "rb") as f:
        for i in range(len(offsets) - 1):
            start_offset = int(offsets[i])
            end_offset = int(offsets[i + 1])
            num_bytes = end_offset - start_offset
            if num_bytes <= 0:
                logger.warning(f"Corrupt/empty region: {bin_path} [{i}] {start_offset}-{end_offset}")
                continue
            num_tokens = num_bytes // itemsize
            if num_tokens == 0:
                logger.warning(f"Zero tokens: {bin_path} [{i}] {start_offset}-{end_offset}")
                continue
            num_full = num_tokens // seq_len
            if num_full == 0:
                logger.warning(f"Incomplete sequence: {bin_path} [{i}] {num_tokens} tokens")
                continue
            read_tokens = num_full * seq_len
            f.seek(start_offset)
            tokens = np.frombuffer(f.read(read_tokens * itemsize), dtype=dtype)
            if tokens.size != read_tokens:
                logger.warning(f"Incomplete read: {bin_path} [{i}] expected {read_tokens}, got {tokens.size}")
                continue
            for j in range(0, len(tokens), seq_len):
                logger.debug(f"Yielding sequence {j//seq_len} from {bin_path}")
                yield torch.from_numpy(tokens[j:j+seq_len])

def bin_idx_source(shard_dir, seq_len=None, dtype=None):
    logger = logging.getLogger("spdl.loader")
    """
    Generator yielding token sequences from all .bin/.idx shards in a directory.
    """
    if dtype is None:
        dtype = np.dtype(DTYPE)
    if seq_len is None:
        seq_len = SEQUENCE_LENGTH
    files = sorted([f for f in os.listdir(shard_dir) if f.endswith(".bin")])
    logger.info(f"Found {len(files)} .bin files in {shard_dir}")
    for bin_file in files:
        bin_path = os.path.join(shard_dir, bin_file)
        idx_path = bin_path.replace(".bin", ".idx")
        logger.info(f"Processing shard: {bin_path}")
        yield from load_tokens_from_bin_idx(bin_path, idx_path, dtype=dtype, seq_len=seq_len)

def build_pipeline(shard_dir, seq_len=None, dtype=None):
    """
    Build a SPDL pipeline using .bin/.idx token shards.
    """
    return (
        PipelineBuilder()
        .add_source(bin_idx_source(shard_dir, seq_len=seq_len, dtype=dtype))
        .aggregate(BATCH_SIZE)
        .add_sink(PREFETCH_BUFFER)
        .build(num_threads=NUM_THREADS)
    )

class DummyModel(torch.nn.Module):
    def forward(self, x):
        return x.sum(dim=1)
