"""Tests for plugin system."""

import tempfile
from pathlib import Path

import pytest
import yaml
from curriculum_tags.core.plugin import MetricPlugin
from curriculum_tags.utils.curriculum_loader import CurriculumConfig


class DummyMetric(MetricPlugin):
    """Test plugin."""

    name = "dummy"

    def compute(self, sample):
        return {"result": len(sample.get("text", ""))}


@pytest.fixture
def temp_config():
    """Create temporary config file."""
    config_data = {"version": "0.1", "test": {"key": "value"}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        path = Path(f.name)

    yield CurriculumConfig(path)
    path.unlink()


def test_plugin_initialization(temp_config):
    """Test initializing a plugin."""
    plugin = DummyMetric(temp_config)

    assert plugin.name == "dummy"
    assert plugin.config is not None


def test_plugin_compute(temp_config):
    """Test plugin computation."""
    plugin = DummyMetric(temp_config)
    result = plugin.compute({"text": "hello"})

    assert result["result"] == 5


def test_plugin_with_empty_text(temp_config):
    """Test plugin with empty text."""
    plugin = DummyMetric(temp_config)
    result = plugin.compute({"text": ""})

    assert result["result"] == 0


def test_plugin_with_missing_text(temp_config):
    """Test plugin with missing text field."""
    plugin = DummyMetric(temp_config)
    result = plugin.compute({})

    assert result["result"] == 0


def test_plugin_access_curriculum_config(temp_config):
    """Test that plugin can access curriculum config."""

    class ConfigAwareMetric(MetricPlugin):
        name = "config_aware"

        def compute(self, sample):
            # Access config
            test_value = self.config.get("test.key")
            return {"config_value": test_value}

    plugin = ConfigAwareMetric(temp_config)
    result = plugin.compute({"text": "test"})

    assert result["config_value"] == "value"
