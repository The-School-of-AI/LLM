"""
Tests for StreamingTokenDataset.

Creates small temporary .npy shards and verifies:
- Correct sequence count and content
- Deterministic ordering (no shuffling)
- Resume via skip_samples
- Partial sequence skip at shard boundaries
- GPU-count-agnostic resume with different skip values
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from data_loader.streaming_dataset import StreamingTokenDataset


@pytest.fixture
def shard_dir():
    """Create a temp directory with small .npy shards."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def _create_shards(shard_dir, shard_specs):
    """Create .npy shards with specified token counts.

    Parameters
    ----------
    shard_dir : str
        Directory to write shards to.
    shard_specs : list[int]
        Number of tokens in each shard.

    Returns
    -------
    list[str]
        Sorted list of shard file paths.
    """
    paths = []
    token_offset = 0
    for i, num_tokens in enumerate(shard_specs):
        tokens = np.arange(token_offset, token_offset + num_tokens, dtype=np.int64)
        path = os.path.join(shard_dir, f"shard-{i:05d}.npy")
        np.save(path, tokens)
        paths.append(path)
        token_offset += num_tokens
    return sorted(paths)


class TestStreamingTokenDataset:
    """Tests for StreamingTokenDataset."""

    def test_basic_length(self, shard_dir):
        """Each shard of 101 tokens with seq_length=10 yields 10 sequences."""
        paths = _create_shards(shard_dir, [101, 101])
        ds = StreamingTokenDataset(paths, seq_length=10)
        # Each shard: (101 - 1) // 10 = 10 sequences
        assert len(ds) == 20

    def test_single_shard_content(self, shard_dir):
        """Verify input_ids and labels are correctly shifted by 1."""
        paths = _create_shards(shard_dir, [21])  # 21 tokens → 2 sequences of length 10
        ds = StreamingTokenDataset(paths, seq_length=10)

        sample = ds[0]
        assert sample["input_ids"].shape == (10,)
        assert sample["labels"].shape == (10,)
        assert sample["attention_mask"].shape == (10,)

        # input_ids = tokens[0:10], labels = tokens[1:11]
        expected_input = torch.arange(0, 10, dtype=torch.int64)
        expected_labels = torch.arange(1, 11, dtype=torch.int64)
        assert torch.equal(sample["input_ids"], expected_input)
        assert torch.equal(sample["labels"], expected_labels)
        assert torch.all(sample["attention_mask"] == 1)

    def test_deterministic_ordering(self, shard_dir):
        """Two instantiations produce identical sequence ordering."""
        paths = _create_shards(shard_dir, [51, 31, 81])
        ds1 = StreamingTokenDataset(paths, seq_length=10)
        ds2 = StreamingTokenDataset(paths, seq_length=10)

        for i in range(len(ds1)):
            s1 = ds1[i]
            s2 = ds2[i]
            assert torch.equal(s1["input_ids"], s2["input_ids"])

    def test_skip_samples_resume(self, shard_dir):
        """skip_samples correctly removes the first N entries."""
        paths = _create_shards(shard_dir, [101])
        ds_full = StreamingTokenDataset(paths, seq_length=10)
        ds_skip = StreamingTokenDataset(paths, seq_length=10, skip_samples=3)

        assert len(ds_skip) == len(ds_full) - 3

        # First sample of skipped dataset == 4th sample of full dataset
        assert torch.equal(ds_skip[0]["input_ids"], ds_full[3]["input_ids"])

    def test_skip_all_samples(self, shard_dir):
        """Skipping all samples yields an empty dataset."""
        paths = _create_shards(shard_dir, [101])
        ds = StreamingTokenDataset(paths, seq_length=10)
        total = len(ds)

        ds_empty = StreamingTokenDataset(paths, seq_length=10, skip_samples=total)
        assert len(ds_empty) == 0

    def test_skip_more_than_total(self, shard_dir):
        """Skipping more than available produces empty dataset (with warning)."""
        paths = _create_shards(shard_dir, [21])
        ds = StreamingTokenDataset(paths, seq_length=10, skip_samples=9999)
        assert len(ds) == 0

    def test_uneven_shards(self, shard_dir):
        """Uneven shard sizes are handled correctly."""
        # Shard 0: 51 tokens → (51-1)//10 = 5 sequences
        # Shard 1: 11 tokens → (11-1)//10 = 1 sequence
        # Shard 2: 201 tokens → (201-1)//10 = 20 sequences
        paths = _create_shards(shard_dir, [51, 11, 201])
        ds = StreamingTokenDataset(paths, seq_length=10)
        assert len(ds) == 26

    def test_partial_shard_skipped(self, shard_dir):
        """A shard with fewer than seq_length+1 tokens produces 0 sequences."""
        paths = _create_shards(shard_dir, [10])  # 10 tokens, need 11 for 1 sequence
        ds = StreamingTokenDataset(paths, seq_length=10)
        assert len(ds) == 0

    def test_get_first_active_shard_idx(self, shard_dir):
        """get_first_active_shard_idx returns correct shard after skip."""
        # Shard 0: 21 tokens → 2 sequences
        # Shard 1: 21 tokens → 2 sequences
        paths = _create_shards(shard_dir, [21, 21])
        ds = StreamingTokenDataset(paths, seq_length=10, skip_samples=2)
        # After skipping 2 (all of shard 0), first active should be shard 1
        assert ds.get_first_active_shard_idx() == 1

    def test_get_total_samples(self, shard_dir):
        """get_total_samples returns count before skip."""
        paths = _create_shards(shard_dir, [101])
        ds = StreamingTokenDataset(paths, seq_length=10, skip_samples=3)
        assert ds.get_total_samples() == 10
        assert len(ds) == 7

    def test_gpu_count_agnostic_resume(self, shard_dir):
        """Simulates resume with different GPU count — same global skip works."""
        paths = _create_shards(shard_dir, [101, 101])  # 20 total sequences

        # "8 GPU run": consumed 8 samples (1 step × batch_size=1 × 8 GPUs)
        ds_8gpu = StreamingTokenDataset(paths, seq_length=10, skip_samples=8)

        # "5 GPU run" with same skip: should see the same remaining data
        ds_5gpu = StreamingTokenDataset(paths, seq_length=10, skip_samples=8)

        assert len(ds_8gpu) == len(ds_5gpu) == 12
        assert torch.equal(ds_8gpu[0]["input_ids"], ds_5gpu[0]["input_ids"])
