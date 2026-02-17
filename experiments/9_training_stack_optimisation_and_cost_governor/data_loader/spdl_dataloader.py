
import os
import torch
import numpy as np
from spdl.pipeline import PipelineBuilder
from common import BATCH_SIZE, NUM_THREADS, PREFETCH_BUFFER, SEQUENCE_LENGTH, DTYPE

def read_idx(idx_path):
    """Read .idx file and return list of document offsets."""
    with open(idx_path, "rb") as f:
        header = f.read(8)  # 8 bytes for versioning/dtype (not used here)
        offsets = np.frombuffer(f.read(), dtype=np.uint64)
    return offsets

def load_tokens_from_bin_idx(bin_path, idx_path, dtype=None, seq_len=None):
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
    with open(bin_path, "rb") as f:
        for i in range(len(offsets) - 1):
            # Cast offsets to Python int for safe arithmetic and file ops
            start_offset = int(offsets[i])
            end_offset = int(offsets[i + 1])
            num_bytes = end_offset - start_offset
            if num_bytes <= 0:
                continue  # skip corrupt or empty regions
            num_tokens = num_bytes // itemsize
            if num_tokens == 0:
                continue
            # Only use full sequences
            num_full = num_tokens // seq_len
            if num_full == 0:
                continue
            read_tokens = num_full * seq_len
            f.seek(start_offset)
            tokens = np.frombuffer(f.read(read_tokens * itemsize), dtype=dtype)
            if tokens.size != read_tokens:
                continue  # skip incomplete reads
            for j in range(0, len(tokens), seq_len):
                yield torch.from_numpy(tokens[j:j+seq_len])

def bin_idx_source(shard_dir, seq_len=None, dtype=None):
    """
    Generator yielding token sequences from all .bin/.idx shards in a directory.
    """
    if dtype is None:
        dtype = np.dtype(DTYPE)
    if seq_len is None:
        seq_len = SEQUENCE_LENGTH
    files = sorted([f for f in os.listdir(shard_dir) if f.endswith(".bin")])
    for bin_file in files:
        bin_path = os.path.join(shard_dir, bin_file)
        idx_path = bin_path.replace(".bin", ".idx")
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
