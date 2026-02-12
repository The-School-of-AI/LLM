import hashlib
from pathlib import Path

import pytest
from repro.checksum import checksum_path

# Test: returns valid SHA-256 hash


def test_checksum_returns_sha256(tmp_path: Path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("hello world")

    digest = checksum_path(file_path)

    assert isinstance(digest, str)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


# Test: checksum matches hashlib reference


def test_checksum_matches_hashlib(tmp_path: Path):
    content = b"some binary data\nwith multiple lines"
    file_path = tmp_path / "data.bin"
    file_path.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    actual = checksum_path(file_path)

    assert actual == expected


# Test: same file → same checksum (deterministic)


def test_checksum_is_deterministic(tmp_path: Path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("repeatable content")

    h1 = checksum_path(file_path)
    h2 = checksum_path(file_path)

    assert h1 == h2


# Test: different content → different checksum


def test_checksum_changes_when_file_changes(tmp_path: Path):
    file_path = tmp_path / "file.txt"

    file_path.write_text("version one")
    h1 = checksum_path(file_path)

    file_path.write_text("version two")
    h2 = checksum_path(file_path)

    assert h1 != h2


# Test: empty file has known SHA-256


def test_checksum_empty_file(tmp_path: Path):
    file_path = tmp_path / "empty.txt"
    file_path.write_bytes(b"")

    digest = checksum_path(file_path)

    assert digest == hashlib.sha256(b"").hexdigest()


# (Optional) Test: missing file raises error


def test_checksum_missing_file_raises(tmp_path: Path):
    missing = tmp_path / "does_not_exist.txt"

    with pytest.raises(FileNotFoundError):
        checksum_path(missing)
