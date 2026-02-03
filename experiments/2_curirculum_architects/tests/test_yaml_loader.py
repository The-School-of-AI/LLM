"""Tests for YAML configuration loader."""

import tempfile
from pathlib import Path

import pytest
import yaml
from curriculum_tags.utils.curriculum_loader import CurriculumConfig


@pytest.fixture
def sample_curriculum_v1():
    """Sample curriculum YAML (version 1 structure)."""
    return {
        "version": "0.1",
        "difficulty_system": {
            "bands": {
                "B0": {
                    "name": "Nursery",
                    "intent": "Surface language",
                },
                "B1": {
                    "name": "Primary",
                    "intent": "Basic text",
                },
            }
        },
        "languages": {
            "primary": ["en"],
        },
    }


@pytest.fixture
def sample_curriculum_v2():
    """Sample curriculum YAML (version 2 structure)."""
    return {
        "version": "0.2",
        "difficulty_bands": {
            "bands": [
                {"id": "B0", "name": "Nursery"},
                {"id": "B1", "name": "Primary"},
            ]
        },
        "languages": {
            "primary": ["en"],
        },
    }


@pytest.fixture
def temp_yaml_file(sample_curriculum_v1):
    """Create temporary YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_curriculum_v1, f)
        path = Path(f.name)

    yield path
    path.unlink()


def test_load_yaml(temp_yaml_file):
    """Test loading YAML file."""
    config = CurriculumConfig(temp_yaml_file)
    assert config.version == "0.1"


def test_get_simple_key(temp_yaml_file):
    """Test getting simple key."""
    config = CurriculumConfig(temp_yaml_file)
    assert config.get("version") == "0.1"


def test_get_nested_key(temp_yaml_file):
    """Test getting nested key with dot notation."""
    config = CurriculumConfig(temp_yaml_file)
    result = config.get("difficulty_system.bands.B0.name")
    assert result == "Nursery"


def test_get_missing_key_with_default(temp_yaml_file):
    """Test getting missing key returns default."""
    config = CurriculumConfig(temp_yaml_file)
    result = config.get("missing.key", "default_value")
    assert result == "default_value"


def test_get_required_existing_key(temp_yaml_file):
    """Test get_required with existing key."""
    config = CurriculumConfig(temp_yaml_file)
    result = config.get_required("version")
    assert result == "0.1"


def test_get_required_missing_key(temp_yaml_file):
    """Test get_required raises error for missing key."""
    config = CurriculumConfig(temp_yaml_file)
    with pytest.raises(KeyError):
        config.get_required("missing.key")


def test_file_not_found():
    """Test error when file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        CurriculumConfig("nonexistent.yaml")


def test_raw_property(temp_yaml_file):
    """Test accessing raw YAML data."""
    config = CurriculumConfig(temp_yaml_file)
    assert isinstance(config.raw, dict)
    assert "version" in config.raw
