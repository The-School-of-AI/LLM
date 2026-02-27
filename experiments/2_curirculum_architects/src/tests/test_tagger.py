"""Tests for curriculum tagger."""

import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from curriculum_tags.core.plugin import MetricPlugin
from curriculum_tags.core.tagger import CurriculumTagger
from curriculum_tags.utils.curriculum_loader import CurriculumConfig


class SimpleMetric(MetricPlugin):
    """Simple test metric."""

    name = "simple"

    def compute(self, sample):
        text = sample.get("text", "")
        return {
            "length": len(text),
            "word_count": len(text.split()),
        }


@pytest.fixture
def temp_curriculum():
    """Create temporary curriculum config."""
    config_data = {
        "version": "0.1",
        "difficulty_bands": {"bands": [{"id": "B0", "name": "Nursery"}]},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        path = Path(f.name)

    yield path
    path.unlink()


@pytest.fixture
def temp_parquet():
    """Create temporary parquet file with test data."""
    data = [
        {"id": "1", "text": "Hello world", "source": "test"},
        {"id": "2", "text": "Python programming", "source": "test"},
    ]
    table = pa.Table.from_pylist(data)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = Path(f.name)

    pq.write_table(table, path)

    yield path
    path.unlink()


def test_tagger_initialization(temp_curriculum):
    """Test tagger initialization."""
    config = CurriculumConfig(temp_curriculum)
    metrics = [SimpleMetric(config)]

    tagger = CurriculumTagger(temp_curriculum, metrics=metrics)

    assert tagger.config.version == "0.1"
    assert len(tagger.plugins) == 1


def test_tagger_default_metrics(temp_curriculum):
    """Test tagger with default metrics from config."""
    # Should use default metrics if none specified
    tagger = CurriculumTagger(temp_curriculum)

    # Should have loaded default metrics
    assert len(tagger.plugins) > 0
    # Check they are actual metric instances
    assert all(isinstance(p, MetricPlugin) for p in tagger.plugins)


def test_tag_sample(temp_curriculum):
    """Test tagging a single sample."""
    config = CurriculumConfig(temp_curriculum)
    plugins = [SimpleMetric(config)]

    tagger = CurriculumTagger(temp_curriculum, metrics=plugins)

    sample = {"id": "1", "text": "Hello world"}
    tagged = tagger.tag_sample(sample)

    assert "curriculum_tags" in tagged
    assert tagged["curriculum_tags"]["version"] == "0.1"
    assert "simple" in tagged["curriculum_tags"]
    assert tagged["curriculum_tags"]["simple"]["length"] == 11


def test_process_batch(temp_curriculum):
    """Test processing a batch of samples."""
    config = CurriculumConfig(temp_curriculum)
    plugins = [SimpleMetric(config)]

    tagger = CurriculumTagger(temp_curriculum, metrics=plugins)

    samples = [
        {"id": "1", "text": "Hello"},
        {"id": "2", "text": "World"},
    ]

    tagged_samples = tagger.process_batch(samples)

    assert len(tagged_samples) == 2
    assert all("curriculum_tags" in s for s in tagged_samples)


def test_plugin_chaining(temp_curriculum):
    """Test that plugins can see results from previous plugins."""

    class FirstMetric(MetricPlugin):
        name = "first"

        def compute(self, sample):
            return {"value": 42}

    class SecondMetric(MetricPlugin):
        name = "second"

        def compute(self, sample):
            # Access first metric's result
            first_value = (
                sample.get("curriculum_tags", {}).get("first", {}).get("value")
            )
            return {"doubled": first_value * 2 if first_value else 0}

    config = CurriculumConfig(temp_curriculum)
    plugins = [FirstMetric(config), SecondMetric(config)]

    tagger = CurriculumTagger(temp_curriculum, metrics=plugins)

    sample = {"id": "1", "text": "Test"}
    tagged = tagger.tag_sample(sample)

    assert tagged["curriculum_tags"]["first"]["value"] == 42
    assert tagged["curriculum_tags"]["second"]["doubled"] == 84


def test_process_parquet(temp_curriculum, temp_parquet):
    """Test processing parquet file."""
    config = CurriculumConfig(temp_curriculum)
    plugins = [SimpleMetric(config)]

    tagger = CurriculumTagger(temp_curriculum, metrics=plugins)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        output_path = Path(f.name)

    try:
        stats = tagger.process_parquet(temp_parquet, output_path)

        assert stats["total_rows"] == 2
        assert stats["error_count"] == 0
        assert Path(stats["output_file"]).exists()

        # Verify output
        result_table = pq.read_table(output_path)
        result_data = result_table.to_pylist()

        assert len(result_data) == 2
        assert all("curriculum_tags" in row for row in result_data)

    finally:
        output_path.unlink()


def test_process_parquet_with_errors(temp_curriculum, temp_parquet):
    """Test handling errors during processing."""

    class ErrorMetric(MetricPlugin):
        name = "error_metric"

        def compute(self, sample):
            raise ValueError("Test error")

    config = CurriculumConfig(temp_curriculum)
    plugins = [ErrorMetric(config)]

    tagger = CurriculumTagger(temp_curriculum, metrics=plugins)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        output_path = Path(f.name)

    try:
        stats = tagger.process_parquet(temp_parquet, output_path)

        # Should process but record errors
        assert stats["total_rows"] == 2
        # Errors are caught and stored in tags
        result_table = pq.read_table(output_path)
        result_data = result_table.to_pylist()
        assert all(
            "error" in row["curriculum_tags"]["error_metric"] for row in result_data
        )

    finally:
        output_path.unlink()


def test_process_nonexistent_file(temp_curriculum):
    """Test error when input file doesn't exist."""
    tagger = CurriculumTagger(temp_curriculum, metrics=[])

    with pytest.raises(FileNotFoundError):
        tagger.process_parquet("nonexistent.parquet", "output.parquet")
