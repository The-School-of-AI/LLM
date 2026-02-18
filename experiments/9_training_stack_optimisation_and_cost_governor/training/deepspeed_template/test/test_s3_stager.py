"""
Tests for S3Stager.

Uses unittest.mock to mock boto3 S3 calls.  Verifies:
- discover_shards() returns deterministic sorted keys
- stage_initial() downloads correct number of shards
- _download_shard() writes to correct local path
- Idempotent download (skips existing files)
- stage_initial_async returns a joinable thread
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from data_loader.s3_stager import S3Stager


@pytest.fixture
def local_dir():
    """Temporary directory for staged shards."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client."""
    client = MagicMock()
    return client


def _make_paginator_response(keys):
    """Build a mock paginator response for list_objects_v2."""
    return [
        {"Contents": [{"Key": k} for k in keys]}
    ]


class TestS3Stager:
    """Tests for S3Stager."""

    @patch("data_loader.s3_stager.boto3")
    def test_discover_shards_sorted(self, mock_boto3, local_dir):
        """discover_shards returns keys in deterministic sorted order."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        # Return keys in unsorted order
        keys = [
            "prefix/shard-00002.npy",
            "prefix/shard-00000.npy",
            "prefix/shard-00001.npy",
            "prefix/README.md",  # Non-.npy file, should be excluded
        ]

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = _make_paginator_response(keys)
        mock_client.get_paginator.return_value = mock_paginator

        stager = S3Stager(
            bucket="test-bucket",
            prefix="prefix",
            local_dir=local_dir,
        )

        result = stager.discover_shards()

        assert result == [
            "prefix/shard-00000.npy",
            "prefix/shard-00001.npy",
            "prefix/shard-00002.npy",
        ]

    @patch("data_loader.s3_stager.boto3")
    def test_stage_initial_downloads_correct_count(self, mock_boto3, local_dir):
        """stage_initial downloads the requested number of shards."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        all_keys = [
            f"prefix/shard-{i:05d}.npy" for i in range(10)
        ]

        # Mock download_file to create actual files
        def fake_download(bucket, key, local_path):
            # Write raw bytes (not np.save which appends .npy extension)
            tokens = np.arange(100, dtype=np.int64)
            with open(local_path, "wb") as f:
                f.write(tokens.tobytes())

        mock_client.download_file.side_effect = fake_download

        # Mock head_object for integrity check (multipart ETag to skip MD5)
        mock_client.head_object.return_value = {"ETag": '"abc-1"'}

        stager = S3Stager(
            bucket="test-bucket",
            prefix="prefix",
            local_dir=local_dir,
        )

        staged = stager.stage_initial(all_keys, start_idx=0, num_shards=3)

        assert len(staged) == 3
        assert mock_client.download_file.call_count == 3

    @patch("data_loader.s3_stager.boto3")
    def test_stage_initial_with_offset(self, mock_boto3, local_dir):
        """stage_initial respects start_idx offset."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        all_keys = [f"prefix/shard-{i:05d}.npy" for i in range(10)]

        def fake_download(bucket, key, local_path):
            # Write raw bytes (not np.save which appends .npy extension)
            tokens = np.arange(100, dtype=np.int64)
            with open(local_path, "wb") as f:
                f.write(tokens.tobytes())

        mock_client.download_file.side_effect = fake_download
        mock_client.head_object.return_value = {"ETag": '"abc-1"'}

        stager = S3Stager(
            bucket="test-bucket",
            prefix="prefix",
            local_dir=local_dir,
        )

        staged = stager.stage_initial(all_keys, start_idx=5, num_shards=3)

        assert len(staged) == 3
        # Verify the correct keys were downloaded (shards 5, 6, 7)
        downloaded_keys = [
            call[0][1] for call in mock_client.download_file.call_args_list
        ]
        for key in downloaded_keys:
            idx = int(key.split("-")[-1].replace(".npy", ""))
            assert 5 <= idx <= 7

    @patch("data_loader.s3_stager.boto3")
    def test_idempotent_download(self, mock_boto3, local_dir):
        """_download_shard skips files that already exist."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        stager = S3Stager(
            bucket="test-bucket",
            prefix="prefix",
            local_dir=local_dir,
        )

        # Pre-create the file
        existing_path = os.path.join(local_dir, "shard-00000.npy")
        np.save(existing_path, np.arange(10, dtype=np.int64))

        result = stager._download_shard("prefix/shard-00000.npy")
        assert result == existing_path
        mock_client.download_file.assert_not_called()

    @patch("data_loader.s3_stager.boto3")
    def test_stage_initial_async_joinable(self, mock_boto3, local_dir):
        """stage_initial_async returns a thread that can be joined."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        def fake_download(bucket, key, local_path):
            # Write raw bytes (not np.save which appends .npy extension)
            tokens = np.arange(100, dtype=np.int64)
            with open(local_path, "wb") as f:
                f.write(tokens.tobytes())

        mock_client.download_file.side_effect = fake_download
        mock_client.head_object.return_value = {"ETag": '"abc-1"'}

        stager = S3Stager(
            bucket="test-bucket",
            prefix="prefix",
            local_dir=local_dir,
        )

        all_keys = [f"prefix/shard-{i:05d}.npy" for i in range(3)]
        thread = stager.stage_initial_async(all_keys, start_idx=0, num_shards=3)

        thread.join(timeout=10)
        assert not thread.is_alive()
        assert len(stager.get_staged_shards()) == 3

    @patch("data_loader.s3_stager.boto3")
    def test_s3_key_to_local_path(self, mock_boto3, local_dir):
        """s3_key_to_local_path returns correct path."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        stager = S3Stager(
            bucket="test-bucket",
            prefix="prefix",
            local_dir=local_dir,
        )

        result = stager.s3_key_to_local_path("prefix/shard-00042.npy")
        expected = os.path.join(local_dir, "shard-00042.npy")
        assert result == expected
