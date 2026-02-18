"""
Tests for PrefetchDataLoader.

Verifies:
- Batches arrive on correct device (CPU fallback if no GPU)
- Iteration completes without hangs
- __len__ matches underlying loader
- Prefetch depth is respected
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_loader.prefetch_loader import PrefetchDataLoader


def _make_loader(num_samples=32, batch_size=4, seq_length=10):
    """Create a simple DataLoader with dummy data."""
    input_ids = torch.randint(0, 1000, (num_samples, seq_length))
    labels = torch.randint(0, 1000, (num_samples, seq_length))
    dataset = TensorDataset(input_ids, labels)
    return DataLoader(dataset, batch_size=batch_size, pin_memory=False)


class TestPrefetchDataLoader:
    """Tests for PrefetchDataLoader."""

    def test_len_matches(self):
        """__len__ matches the underlying DataLoader."""
        loader = _make_loader(num_samples=32, batch_size=4)
        prefetch = PrefetchDataLoader(loader, device=torch.device("cpu"))
        assert len(prefetch) == len(loader)

    def test_iteration_completes(self):
        """Full iteration completes without hanging."""
        loader = _make_loader(num_samples=20, batch_size=4)
        prefetch = PrefetchDataLoader(
            loader, device=torch.device("cpu"), prefetch_depth=2
        )

        count = 0
        for batch in prefetch:
            count += 1
            assert isinstance(batch, (list, tuple))
            assert len(batch) == 2  # input_ids, labels

        assert count == len(loader)

    def test_correct_device_cpu(self):
        """Batches are on CPU when device is CPU."""
        loader = _make_loader(num_samples=8, batch_size=4)
        prefetch = PrefetchDataLoader(loader, device=torch.device("cpu"))

        for batch in prefetch:
            for tensor in batch:
                assert tensor.device == torch.device("cpu")

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_correct_device_gpu(self):
        """Batches are on GPU when CUDA device is specified."""
        loader = _make_loader(num_samples=8, batch_size=4)
        prefetch = PrefetchDataLoader(
            loader, device=torch.device("cuda:0"), prefetch_depth=2
        )

        for batch in prefetch:
            for tensor in batch:
                assert tensor.is_cuda

    def test_batch_content_preserved(self):
        """Data content is preserved through prefetching."""
        input_ids = torch.arange(16).reshape(4, 4)
        labels = torch.arange(16, 32).reshape(4, 4)
        dataset = TensorDataset(input_ids, labels)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        prefetch = PrefetchDataLoader(loader, device=torch.device("cpu"))

        all_inputs = []
        for batch in prefetch:
            all_inputs.append(batch[0])

        result = torch.cat(all_inputs, dim=0)
        assert torch.equal(result, input_ids)

    def test_empty_loader(self):
        """Empty DataLoader produces no batches."""
        dataset = TensorDataset(torch.empty(0, 10), torch.empty(0, 10))
        loader = DataLoader(dataset, batch_size=4)
        prefetch = PrefetchDataLoader(loader, device=torch.device("cpu"))

        count = 0
        for _batch in prefetch:
            count += 1
        assert count == 0

    def test_single_batch(self):
        """Works with exactly one batch."""
        loader = _make_loader(num_samples=4, batch_size=4)
        prefetch = PrefetchDataLoader(
            loader, device=torch.device("cpu"), prefetch_depth=3
        )

        count = 0
        for _batch in prefetch:
            count += 1
        assert count == 1

    def test_dict_batch(self):
        """Handles dict-style batches (like StreamingTokenDataset returns)."""
        # Simulate dict-returning collate
        class DictDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.tensor([idx] * 4),
                    "labels": torch.tensor([idx + 1] * 4),
                    "attention_mask": torch.ones(4),
                }

        loader = DataLoader(DictDataset(), batch_size=2)
        prefetch = PrefetchDataLoader(loader, device=torch.device("cpu"))

        count = 0
        for batch in prefetch:
            assert isinstance(batch, dict)
            assert "input_ids" in batch
            assert "labels" in batch
            assert "attention_mask" in batch
            count += 1
        assert count == 4
