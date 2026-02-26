from pathlib import Path
from unittest.mock import MagicMock

import boto3
import pytest
from repro.s3 import ImmutableS3Writer

# Test: S3 client is created on init


def test_s3_client_created(monkeypatch):
    mock_s3 = MagicMock()

    monkeypatch.setattr(boto3, "client", lambda service: mock_s3)

    writer = ImmutableS3Writer(bucket="my-bucket", prefix="runs")

    assert writer.s3 is mock_s3


# Test: upload_file() builds correct S3 key and calls API


def test_upload_file_calls_s3_with_correct_arguments(monkeypatch, tmp_path: Path):
    mock_s3 = MagicMock()
    monkeypatch.setattr(boto3, "client", lambda service: mock_s3)

    writer = ImmutableS3Writer(bucket="test-bucket", prefix="artifacts")

    local_file = tmp_path / "data.txt"
    local_file.write_text("hello")

    writer.upload_file(local_file, key="config.yaml")

    mock_s3.upload_file.assert_called_once_with(
        str(local_file),
        "test-bucket",
        "artifacts/config.yaml",
        ExtraArgs={"ACL": "bucket-owner-full-control"},
    )


# Test: prefix handling (no accidental mutation)


def test_prefix_is_not_modified(monkeypatch, tmp_path: Path):
    mock_s3 = MagicMock()
    monkeypatch.setattr(boto3, "client", lambda service: mock_s3)

    writer = ImmutableS3Writer(bucket="b", prefix="immutable")

    local_file = tmp_path / "f.txt"
    local_file.write_text("x")

    writer.upload_file(local_file, key="k.txt")

    assert writer.prefix == "immutable"


# Test: upload propagates S3 errors (fail-fast)


def test_upload_file_propagates_s3_error(monkeypatch, tmp_path: Path):
    mock_s3 = MagicMock()
    mock_s3.upload_file.side_effect = RuntimeError("S3 failure")

    monkeypatch.setattr(boto3, "client", lambda service: mock_s3)

    writer = ImmutableS3Writer(bucket="b", prefix="p")

    local_file = tmp_path / "file.txt"
    local_file.write_text("data")

    with pytest.raises(RuntimeError):
        writer.upload_file(local_file, key="file.txt")
