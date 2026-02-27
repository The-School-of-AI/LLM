"""Tests for state manager."""

import tempfile
from pathlib import Path

import pytest
from curriculum_extractor.core.state_manager import StateManager


@pytest.fixture
def temp_state_dir():
    """Create temporary directory for state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestStateManager:
    """Tests for StateManager."""

    def test_init_fresh_state(self, temp_state_dir):
        """Test initialization with no existing state."""
        manager = StateManager(temp_state_dir)

        assert manager.state.total_files == 0
        assert manager.state.completed_files == 0
        assert len(manager.state.files) == 0

    def test_register_files(self, temp_state_dir):
        """Test file registration."""
        manager = StateManager(temp_state_dir)

        files = ["/path/file1.parquet", "/path/file2.parquet"]
        pending = manager.register_files(files)

        assert len(pending) == 2
        assert manager.state.total_files == 2
        assert all(f in manager.state.files for f in files)

    def test_register_files_incremental(self, temp_state_dir):
        """Test incremental file registration."""
        manager = StateManager(temp_state_dir)

        # First batch
        manager.register_files(["/path/file1.parquet"])
        assert manager.state.total_files == 1

        # Second batch (one new, one existing)
        pending = manager.register_files(["/path/file1.parquet", "/path/file2.parquet"])

        assert len(pending) == 1  # Only new file
        assert manager.state.total_files == 2

    def test_mark_in_progress(self, temp_state_dir):
        """Test marking file as in progress."""
        manager = StateManager(temp_state_dir)

        manager.mark_in_progress("/path/file1.parquet")

        assert manager.state.files["/path/file1.parquet"].status == "in_progress"
        assert manager.state.files["/path/file1.parquet"].started_at is not None

    def test_mark_completed(self, temp_state_dir):
        """Test marking file as completed."""
        manager = StateManager(temp_state_dir)

        manager.mark_in_progress("/path/file1.parquet")
        manager.mark_completed(
            "/path/file1.parquet", rows_processed=100, rows_rejected=5
        )

        state = manager.state.files["/path/file1.parquet"]
        assert state.status == "completed"
        assert state.rows_processed == 100
        assert state.rows_rejected == 5
        assert manager.state.completed_files == 1

    def test_mark_failed(self, temp_state_dir):
        """Test marking file as failed."""
        manager = StateManager(temp_state_dir)

        manager.mark_in_progress("/path/file1.parquet")
        manager.mark_failed("/path/file1.parquet", "Error message")

        state = manager.state.files["/path/file1.parquet"]
        assert state.status == "failed"
        assert state.error_message == "Error message"
        assert manager.state.failed_files == 1

    def test_is_completed(self, temp_state_dir):
        """Test checking completion status."""
        manager = StateManager(temp_state_dir)

        manager.mark_in_progress("/path/file1.parquet")
        assert not manager.is_completed("/path/file1.parquet")

        manager.mark_completed("/path/file1.parquet")
        assert manager.is_completed("/path/file1.parquet")

    def test_get_pending_files(self, temp_state_dir):
        """Test getting pending files."""
        manager = StateManager(temp_state_dir)

        manager.register_files(
            ["/path/file1.parquet", "/path/file2.parquet", "/path/file3.parquet"]
        )
        manager.mark_completed("/path/file1.parquet")
        manager.mark_failed("/path/file2.parquet", "error")

        pending = manager.get_pending_files()

        # file2 (failed) and file3 (pending) should be returned
        assert len(pending) == 2
        assert "/path/file3.parquet" in pending
        assert "/path/file2.parquet" in pending  # Failed files can be retried

    def test_persistence(self, temp_state_dir):
        """Test state persistence across instances."""
        # First instance
        manager1 = StateManager(temp_state_dir)
        manager1.register_files(["/path/file1.parquet"])
        manager1.mark_completed("/path/file1.parquet", rows_processed=100)

        # Second instance should load saved state
        manager2 = StateManager(temp_state_dir)

        assert manager2.state.total_files == 1
        assert manager2.is_completed("/path/file1.parquet")
        assert manager2.state.files["/path/file1.parquet"].rows_processed == 100

    def test_reset(self, temp_state_dir):
        """Test state reset."""
        manager = StateManager(temp_state_dir)

        manager.register_files(["/path/file1.parquet"])
        manager.mark_completed("/path/file1.parquet")

        manager.reset()

        assert manager.state.total_files == 0
        assert len(manager.state.files) == 0

    def test_get_stats(self, temp_state_dir):
        """Test statistics gathering."""
        manager = StateManager(temp_state_dir)

        manager.register_files(
            ["/path/file1.parquet", "/path/file2.parquet", "/path/file3.parquet"]
        )
        manager.mark_completed(
            "/path/file1.parquet", rows_processed=100, rows_rejected=10
        )
        manager.mark_failed("/path/file2.parquet", "error")

        stats = manager.get_stats()

        assert stats["total_files"] == 3
        assert stats["completed_files"] == 1
        assert stats["failed_files"] == 1
        assert stats["pending_files"] == 1
        assert stats["total_rows_processed"] == 100
        assert stats["total_rows_rejected"] == 10
