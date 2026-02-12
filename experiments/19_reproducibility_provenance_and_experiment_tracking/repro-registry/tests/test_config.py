from pathlib import Path

import yaml
from repro.config import freeze_config, hash_file

# Test hash_file() produces a valid SHA-256 hash


def test_hash_file_returns_sha256(tmp_path: Path):
    file_path = tmp_path / "data.txt"
    file_path.write_text("hello world")

    digest = hash_file(file_path)

    assert isinstance(digest, str)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


# Test freeze_config() creates parent directories


def test_freeze_config_creates_directories(tmp_path: Path):
    output_path = tmp_path / "a/b/c/config.yaml"

    freeze_config({"x": 1}, output_path)

    assert output_path.exists()


# Test YAML is written correctly


def test_freeze_config_writes_yaml(tmp_path: Path):
    config = {"b": 2, "a": 1}
    output_path = tmp_path / "config.yaml"

    freeze_config(config, output_path)

    loaded = yaml.safe_load(output_path.read_text())
    assert loaded == config


# Deterministic hash for same config (important)


def test_freeze_config_is_deterministic(tmp_path: Path):
    config = {"b": 2, "a": 1}

    hash1 = freeze_config(config, tmp_path / "c1.yaml")
    hash2 = freeze_config(config, tmp_path / "c2.yaml")

    assert hash1 == hash2


# Hash changes when config changes


def test_freeze_config_hash_changes_on_config_change(tmp_path: Path):
    hash1 = freeze_config({"a": 1}, tmp_path / "c1.yaml")
    hash2 = freeze_config({"a": 2}, tmp_path / "c2.yaml")

    assert hash1 != hash2


# Hash changes when structure changes


def test_freeze_config_hash_changes_on_structure_change(tmp_path: Path):
    hash1 = freeze_config({"a": {"b": 1}}, tmp_path / "c1.yaml")
    hash2 = freeze_config({"a": {"b": 1, "c": 2}}, tmp_path / "c2.yaml")

    assert hash1 != hash2


# Optional: verify returned hash matches file hash


def test_freeze_config_returns_correct_hash(tmp_path: Path):
    output_path = tmp_path / "config.yaml"

    returned_hash = freeze_config({"a": 1}, output_path)
    computed_hash = hash_file(output_path)

    assert returned_hash == computed_hash
