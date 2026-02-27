"""Tests for curriculum extractor."""

import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from curriculum_extractor.core.extractor import CurriculumExtractor
from curriculum_extractor.core.plugin import ExtractionResult, MetricPlugin
from curriculum_extractor.utils.curriculum_loader import CurriculumConfig


class SimpleMetric(MetricPlugin):
    """Simple test metric."""

    name = "simple"

    def compute(self, sample):
        text = sample.get("text", "")
        return {
            "length": len(text),
            "word_count": len(text.split()),
        }


class RejectingMetric(MetricPlugin):
    """Metric that rejects short texts."""

    name = "rejector"

    def compute(self, sample):
        return {"checked": True}

    def extract(self, sample):
        text = sample.get("text", "")
        if len(text) < 10:
            return ExtractionResult(
                metrics={"checked": True},
                rejected=True,
                rejection_reason="Text too short",
            )
        return ExtractionResult(metrics={"checked": True, "length": len(text)})


@pytest.fixture
def temp_curriculum():
    """Create temporary curriculum config."""
    config_data = {
        "version": "0.2",
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
        {"id": "1", "text": "Hello world this is a test", "source": "test"},
        {"id": "2", "text": "Short", "source": "test"},
        {
            "id": "3",
            "text": "Python programming language is great for data science",
            "source": "test",
        },
    ]
    table = pa.Table.from_pylist(data)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = Path(f.name)

    pq.write_table(table, path)
    yield path
    path.unlink()


class TestExtractorInitialization:
    """Tests for extractor initialization."""

    def test_init_with_custom_metrics(self, temp_curriculum):
        """Test initialization with custom metrics."""
        config = CurriculumConfig(temp_curriculum)
        metrics = [SimpleMetric(config)]

        extractor = CurriculumExtractor(temp_curriculum, metrics=metrics)

        assert extractor.config.version == "0.2"
        assert len(extractor.plugins) == 1
        assert extractor.plugins[0].name == "simple"

    def test_init_with_default_metrics(self, temp_curriculum):
        """Test initialization with default metrics."""
        extractor = CurriculumExtractor(temp_curriculum)

        assert len(extractor.plugins) > 0
        assert all(isinstance(p, MetricPlugin) for p in extractor.plugins)


class TestExtractRecord:
    """Tests for single record extraction."""

    def test_extract_record_success(self, temp_curriculum):
        """Test successful extraction."""
        config = CurriculumConfig(temp_curriculum)
        metrics = [SimpleMetric(config)]
        extractor = CurriculumExtractor(temp_curriculum, metrics=metrics)

        sample = {"id": "1", "text": "Hello world"}
        metadata, rejection = extractor.extract_record(sample)

        assert rejection is None
        assert metadata is not None
        assert "simple_length" in metadata
        assert metadata["simple_length"] == 11
        assert metadata["simple_word_count"] == 2

    def test_extract_record_rejection(self, temp_curriculum):
        """Test extraction with rejection."""
        config = CurriculumConfig(temp_curriculum)
        metrics = [RejectingMetric(config)]
        extractor = CurriculumExtractor(temp_curriculum, metrics=metrics)

        sample = {"id": "1", "text": "Short"}
        metadata, rejection = extractor.extract_record(sample)

        assert metadata is None
        assert rejection is not None
        assert rejection.rejected_reason == "Text too short"
        assert rejection.rejected_at == "rejector"

    def test_flattened_output(self, temp_curriculum):
        """Test that output is properly flattened."""
        config = CurriculumConfig(temp_curriculum)
        metrics = [SimpleMetric(config)]
        extractor = CurriculumExtractor(temp_curriculum, metrics=metrics)

        sample = {"id": "1", "text": "Hello world"}
        metadata, _ = extractor.extract_record(sample)

        # All keys should be flat (no nested dicts)
        for key, value in metadata.items():
            assert not isinstance(value, dict), f"Key {key} has nested dict"


class TestProcessBatch:
    """Tests for batch processing."""

    def test_process_batch(self, temp_curriculum):
        """Test batch processing."""
        config = CurriculumConfig(temp_curriculum)
        metrics = [SimpleMetric(config)]
        extractor = CurriculumExtractor(temp_curriculum, metrics=metrics)

        samples = [
            {"id": "1", "text": "Hello world"},
            {"id": "2", "text": "Goodbye world"},
        ]

        processed, rejected = extractor.process_batch(samples)

        assert len(processed) == 2
        assert len(rejected) == 0
        assert processed[0]["id"] == "1"
        assert "simple_length" in processed[0]

    def test_process_batch_with_rejections(self, temp_curriculum):
        """Test batch processing with rejections."""
        config = CurriculumConfig(temp_curriculum)
        metrics = [RejectingMetric(config)]
        extractor = CurriculumExtractor(temp_curriculum, metrics=metrics)

        samples = [
            {"id": "1", "text": "Short"},
            {"id": "2", "text": "This is a longer text that passes"},
        ]

        processed, rejected = extractor.process_batch(samples)

        assert len(processed) == 1
        assert len(rejected) == 1
        assert rejected[0]["id"] == "1"


class TestProcessParquet:
    """Tests for parquet processing."""

    def test_process_parquet(self, temp_curriculum, temp_parquet):
        """Test parquet file processing."""
        config = CurriculumConfig(temp_curriculum)
        metrics = [SimpleMetric(config)]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            extractor = CurriculumExtractor(
                temp_curriculum,
                metrics=metrics,
                metadata_output_path=str(tmpdir / "metadata"),
                rejection_output_path=str(tmpdir / "rejections"),
            )

            result = extractor.process_parquet(temp_parquet)

            assert result["status"] == "completed"
            assert result["total_rows"] == 3
            assert result["processed_rows"] == 3

            # Check metadata output exists
            metadata_dir = tmpdir / "metadata"
            assert metadata_dir.exists()


class TestMetricPlugin:
    """Tests for MetricPlugin base class."""

    def test_flatten_metrics(self, temp_curriculum):
        """Test metric flattening."""
        config = CurriculumConfig(temp_curriculum)
        plugin = SimpleMetric(config)

        nested = {
            "score": 0.5,
            "features": {
                "count": 10,
                "ratio": 0.3,
            },
        }

        flat = plugin.flatten_metrics(nested)

        assert "simple_score" in flat
        assert "simple_features_count" in flat
        assert "simple_features_ratio" in flat
        assert flat["simple_score"] == 0.5
        assert flat["simple_features_count"] == 10
