"""Tests for curriculum reader and batch creator."""

import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from curriculum_reader.core.batch_creator import BatchConfig, BatchCreator
from curriculum_reader.core.reader import MetadataReader


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create partitioned structure
        partition_dir = tmpdir / "file_name=test_file"
        partition_dir.mkdir(parents=True)

        # Create sample records
        records = []
        for i in range(100):
            records.append(
                {
                    "uuid": f"uuid-{i:04d}",
                    "id": f"record-{i}",
                    "file_path": "test_file.parquet",
                    "difficulty_score": (i % 10) / 10.0,
                    "band_assignment_band": f"B{i % 6}",
                    "modality_primary_modality": (
                        "general_text" if i % 3 == 0 else "code"
                    ),
                    "opt_metric_1": None,
                    "opt_metric_2": None,
                    "opt_metric_3": None,
                    "opt_metric_4": None,
                    "opt_metric_5": None,
                }
            )

        table = pa.Table.from_pylist(records)
        pq.write_table(table, partition_dir / "metadata.parquet")

        yield tmpdir


class TestMetadataReader:
    """Tests for MetadataReader."""

    def test_count_rows(self, sample_metadata):
        """Test counting rows."""
        reader = MetadataReader(sample_metadata)
        assert reader.count_rows() == 100

    def test_get_schema(self, sample_metadata):
        """Test getting schema."""
        reader = MetadataReader(sample_metadata)
        schema = reader.get_schema()

        assert "uuid" in schema.names
        assert "id" in schema.names
        assert "difficulty_score" in schema.names

    def test_get_column_names(self, sample_metadata):
        """Test getting column names."""
        reader = MetadataReader(sample_metadata)
        columns = reader.get_column_names()

        assert "uuid" in columns
        assert "band_assignment_band" in columns

    def test_read_all(self, sample_metadata):
        """Test reading all data."""
        reader = MetadataReader(sample_metadata)
        table = reader.read_all()

        assert len(table) == 100

    def test_read_with_columns(self, sample_metadata):
        """Test reading specific columns."""
        reader = MetadataReader(sample_metadata)
        table = reader.read_all(columns=["uuid", "id"])

        assert len(table) == 100
        assert len(table.schema) == 2

    def test_sample(self, sample_metadata):
        """Test random sampling."""
        reader = MetadataReader(sample_metadata)

        sample1 = reader.sample(n=10, seed=42)
        sample2 = reader.sample(n=10, seed=42)

        assert len(sample1) == 10
        # Same seed should give same results
        assert sample1.column("uuid").to_pylist() == sample2.column("uuid").to_pylist()

    def test_get_record_by_id(self, sample_metadata):
        """Test getting record by ID."""
        reader = MetadataReader(sample_metadata)

        record = reader.get_record_by_id("record-5")

        assert record is not None
        assert record["id"] == "record-5"

    def test_get_record_not_found(self, sample_metadata):
        """Test getting non-existent record."""
        reader = MetadataReader(sample_metadata)

        record = reader.get_record_by_id("nonexistent")

        assert record is None


class TestBatchCreator:
    """Tests for BatchCreator."""

    def test_init(self, sample_metadata):
        """Test initialization."""
        reader = MetadataReader(sample_metadata)
        config = BatchConfig(batch_size=10, seed=42)

        creator = BatchCreator(reader, config)

        assert creator.state.total_records == 100
        assert creator.state.total_batches == 10

    def test_get_batch_deterministic(self, sample_metadata):
        """Test that batches are deterministic."""
        reader = MetadataReader(sample_metadata)
        config = BatchConfig(batch_size=10, seed=42)

        creator1 = BatchCreator(reader, config)
        creator2 = BatchCreator(reader, config)

        batch1 = creator1.get_batch(0)
        batch2 = creator2.get_batch(0)

        # Same seed, same batch number -> same records
        assert batch1.column("uuid").to_pylist() == batch2.column("uuid").to_pylist()

    def test_different_seeds_different_order(self, sample_metadata):
        """Test that different seeds give different orders."""
        reader = MetadataReader(sample_metadata)

        config1 = BatchConfig(batch_size=10, seed=42)
        config2 = BatchConfig(batch_size=10, seed=99)

        creator1 = BatchCreator(reader, config1)
        creator2 = BatchCreator(reader, config2)

        batch1 = creator1.get_batch(0)
        batch2 = creator2.get_batch(0)

        # Different seeds -> different records
        assert batch1.column("uuid").to_pylist() != batch2.column("uuid").to_pylist()

    def test_auto_increment(self, sample_metadata):
        """Test auto-incrementing batch number."""
        reader = MetadataReader(sample_metadata)
        config = BatchConfig(batch_size=10, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            creator = BatchCreator(reader, config, state_path=Path(tmpdir))

            assert creator.get_current_batch_number() == 0

            creator.get_batch()  # Auto-increment
            assert creator.get_current_batch_number() == 1

            creator.get_batch()  # Auto-increment
            assert creator.get_current_batch_number() == 2

    def test_batch_size(self, sample_metadata):
        """Test batch size is correct."""
        reader = MetadataReader(sample_metadata)
        config = BatchConfig(batch_size=10, seed=42)

        creator = BatchCreator(reader, config)

        batch = creator.get_batch(0)
        assert len(batch) == 10

    def test_reset(self, sample_metadata):
        """Test resetting batch state."""
        reader = MetadataReader(sample_metadata)
        config = BatchConfig(batch_size=10, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            creator = BatchCreator(reader, config, state_path=Path(tmpdir))

            creator.get_batch()  # Increment to 1
            creator.get_batch()  # Increment to 2

            creator.reset()

            assert creator.get_current_batch_number() == 0

    def test_seek(self, sample_metadata):
        """Test seeking to specific batch."""
        reader = MetadataReader(sample_metadata)
        config = BatchConfig(batch_size=10, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            creator = BatchCreator(reader, config, state_path=Path(tmpdir))

            creator.seek(5)

            assert creator.get_current_batch_number() == 5

    def test_batch_includes_metadata(self, sample_metadata):
        """Test that batch includes metadata columns."""
        reader = MetadataReader(sample_metadata)
        config = BatchConfig(batch_size=10, seed=42)

        creator = BatchCreator(reader, config)

        batch = creator.get_batch(0, include_batch_info=True)

        assert "_batch_number" in batch.schema.names
        assert "_batch_seed" in batch.schema.names

        # All rows should have same batch info
        assert all(v == 0 for v in batch.column("_batch_number").to_pylist())
        assert all(v == 42 for v in batch.column("_batch_seed").to_pylist())

    def test_iter_batches(self, sample_metadata):
        """Test iterating through batches."""
        reader = MetadataReader(sample_metadata)
        config = BatchConfig(batch_size=10, seed=42)

        creator = BatchCreator(reader, config)

        batches = list(creator.iter_batches(start_batch=0, end_batch=3))

        assert len(batches) == 3
        assert batches[0][0] == 0  # First batch number
        assert batches[1][0] == 1
        assert batches[2][0] == 2
