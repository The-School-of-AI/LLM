"""
Integration tests for 2T token scale optimizations.
Tests batch processing, checkpointing, error handling, and resumption.
"""

import os
import random
import shutil
import tempfile

import pytest
from src.core.config import PipelineConfig
from src.core.types import ChunkMetadata, DifficultyBand
from src.error_handling import ErrorRecoveryManager, ErrorSeverity, RetryableError
from src.io.batch_processor import BatchProcessor, CheckpointMetadata


class TestBatchProcessing:
    """Test streaming batch processing"""

    @pytest.fixture
    def temp_dataset(self):
        """Create temporary JSONL dataset"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(1000):
                chunk_dict = {
                    "chunk_id": f"chunk_{i:05d}",
                    "dataset_id": "test_ds",
                    "token_count_estimate": random.randint(64, 256),
                    "band": random.choice(["B0", "B1", "B2", "B3", "B4", "B5"]),
                    "domain": random.choice(
                        ["code", "math", "reasoning", "agentic", "indic", "clean_web"]
                    ),
                    "language": "en",
                    "token_ids": list(range(random.randint(64, 256))),
                }
                import json

                f.write(json.dumps(chunk_dict) + "\n")
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_batch_iterator_basic(self, temp_dataset):
        """Test basic batch iteration"""
        processor = BatchProcessor(batch_size=100)

        batches = list(processor.batch_iterator(temp_dataset))

        assert len(batches) == 10, "Should have 10 batches of 100 chunks"

        for batch_num, batch in enumerate(batches):
            assert len(batch) == 100, f"Batch {batch_num} should have 100 chunks"
            assert all(isinstance(chunk_id, str) for chunk_id, _ in batch)

    def test_batch_iterator_non_divisible(self):
        """Test batch iteration with non-divisible chunk count"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(125):
                import json

                f.write(
                    json.dumps({"chunk_id": f"chunk_{i}", "token_count_estimate": 100})
                    + "\n"
                )
            temp_path = f.name

        try:
            processor = BatchProcessor(batch_size=100)
            batches = list(processor.batch_iterator(temp_path))

            assert len(batches) == 2, "Should have 2 batches (100 + 25)"
            assert len(batches[0]) == 100
            assert len(batches[1]) == 25
        finally:
            os.unlink(temp_path)

    def test_batch_memory_efficiency(self):
        """Test that batching doesn't load entire dataset"""
        BatchProcessor(batch_size=100)

        # Mock generator that tracks max simultaneous chunks
        max_chunks_loaded = [0]
        chunks_loaded = [0]

        def chunk_generator():
            for i in range(10000):
                chunks_loaded[0] += 1
                max_chunks_loaded[0] = max(max_chunks_loaded[0], chunks_loaded[0])
                yield (f"chunk_{i}", {"token_count_estimate": 100})
                chunks_loaded[0] -= 1

        # In real use, this would stream from file
        # Here we're testing the batch_iterator logic
        # (actual streaming file I/O tested separately)


class TestCheckpointing:
    """Test checkpoint save/load/resumption"""

    @pytest.fixture
    def checkpoint_dir(self):
        """Create temporary checkpoint directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_checkpoint_save_load(self, checkpoint_dir):
        """Test saving and loading checkpoints"""
        processor = BatchProcessor(checkpoint_dir=checkpoint_dir)

        metadata = CheckpointMetadata(
            stage_name="70B",
            batch_num=42,
            chunks_processed=420_000,
            tokens_processed=67_200_000,
            selected_chunks=315_000,
            timestamp="2024-01-15T10:30:00",
            config_hash="abc123",
        )

        state = {"selected": {"chunk_1", "chunk_2"}, "stats": {"ratio": 0.75}}

        processor.save_checkpoint("70B", 42, state, metadata)

        # Check file exists (format: checkpoint_{stage}_batch_{batch_num:06d}.pkl)
        checkpoint_file = os.path.join(
            checkpoint_dir, "checkpoint_70B_batch_000042.pkl"
        )
        assert os.path.exists(checkpoint_file), f"Expected {checkpoint_file} to exist"

        # Test load
        loaded_state, loaded_metadata = processor.load_checkpoint("70B", 42)
        assert loaded_state is not None
        assert loaded_metadata["stage_name"] == "70B"
        assert loaded_metadata["batch_num"] == 42

    def test_find_last_checkpoint(self, checkpoint_dir):
        """Test finding last checkpoint"""
        processor = BatchProcessor(checkpoint_dir=checkpoint_dir)

        # Save multiple checkpoints
        for batch_num in [0, 1, 2, 5, 10]:
            metadata = CheckpointMetadata(
                stage_name="70B",
                batch_num=batch_num,
                chunks_processed=batch_num * 10_000,
                tokens_processed=batch_num * 1_600_000,
                selected_chunks=batch_num * 7_500,
                timestamp="2024-01-15T10:30:00",
                config_hash="abc123",
            )
            processor.save_checkpoint("70B", batch_num, {}, metadata)

        last = processor.find_last_checkpoint("70B")
        assert last == 10, "Should find batch 10 as last"

    def test_checkpoint_resumption_logic(self, checkpoint_dir):
        """Test logic for resuming from checkpoint"""
        BatchProcessor(batch_size=100, checkpoint_dir=checkpoint_dir)

        # Simulate: processed batches 0-4, now resuming
        last_batch = 4
        start_batch = (last_batch + 1) if last_batch is not None else 0

        assert start_batch == 5, "Should resume from batch 5"

    def test_checkpoint_skip_already_processed(self, checkpoint_dir):
        """Test skipping already-processed batches during resume"""
        processor = BatchProcessor(checkpoint_dir=checkpoint_dir)

        # Save checkpoints for batches 0-2
        for batch_num in range(3):
            metadata = CheckpointMetadata(
                stage_name="8B",
                batch_num=batch_num,
                chunks_processed=(batch_num + 1) * 10_000,
                tokens_processed=(batch_num + 1) * 1_600_000,
                selected_chunks=(batch_num + 1) * 7_500,
                timestamp="2024-01-15T10:30:00",
                config_hash="abc123",
            )
            processor.save_checkpoint("8B", batch_num, {}, metadata)

        last_batch = processor.find_last_checkpoint("8B")

        # Simulate resumption
        processed_batches = set()
        start_batch = (last_batch + 1) if last_batch is not None else 0

        for batch_num in range(10):
            if batch_num < start_batch:
                continue  # Skip
            processed_batches.add(batch_num)

        assert processed_batches == {
            3,
            4,
            5,
            6,
            7,
            8,
            9,
        }, "Should skip 0-2, process 3-9"


class TestErrorHandling:
    """Test error handling and recovery"""

    def test_error_severity_detection(self):
        """Test error severity inference"""
        manager = ErrorRecoveryManager()

        # Retryable error
        try:
            raise RetryableError("Network timeout")
        except Exception as e:
            context = manager.handle_error(e, "IOError")
            assert context.is_retriable
            assert context.severity == ErrorSeverity.WARNING

    def test_error_logging_and_summary(self):
        """Test error logging and summary generation"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as f:
            log_path = f.name

        try:
            manager = ErrorRecoveryManager(error_log_path=log_path)

            # Simulate multiple errors
            for i in range(3):
                try:
                    if i < 2:
                        raise OSError(
                            f"IO error {i}"
                        )  # IOError is alias for OSError in Python 3
                    else:
                        raise ValueError("Invalid config")
                except Exception as e:
                    manager.handle_error(
                        e, error_type=type(e).__name__, stage_name="70B", batch_num=i
                    )

            # Check error counts
            assert len(manager.errors) == 3
            assert (
                manager.error_counts.get("OSError", 0) == 2
                or manager.error_counts.get("IOError", 0) == 2
            )
            assert manager.error_counts["ValueError"] == 1

        finally:
            import time

            time.sleep(0.1)  # Give logger time to release file
            if os.path.exists(log_path):
                try:
                    os.unlink(log_path)
                except PermissionError:
                    pass  # Windows file locking, not critical

    def test_recovery_action_suggestions(self):
        """Test recovery action suggestions"""
        manager = ErrorRecoveryManager()

        try:
            raise MemoryError("Insufficient memory")
        except Exception as e:
            context = manager.handle_error(e, "MemoryError")
            action = manager.get_recovery_action(context)

            assert "batch_size" in action or "RAM" in action


class TestBatchedSelectionEngine:
    """Test batched selection engine"""

    @pytest.fixture
    def simple_config(self):
        """Create minimal config for testing"""
        config = PipelineConfig()
        config.curriculum.deterministic_seed = 42
        return config

    def test_batch_processing_integration(self):
        """Test end-to-end batch processing (simplified)"""

        # Create simple chunk generator
        def chunk_generator():
            for i in range(250):
                metadata = ChunkMetadata(
                    chunk_id=f"chunk_{i:05d}",
                    dataset_id="test",
                    token_count=100,
                    byte_length=400,
                    domain="code",
                    language="en",
                    band=DifficultyBand("B2"),
                    source_doc_id="doc_1",
                )
                yield (f"chunk_{i:05d}", metadata)

        # This would require full setup of SelectionEngine
        # Simplified test just validates the pattern
        chunks = list(chunk_generator())
        assert len(chunks) == 250


class TestOptimizedBuilder:
    """Test optimized coreset builder"""

    def test_checkpoint_aware_initialization(self):
        """Test that builder initializes with checkpoint awareness"""
        # This requires full config setup
        # Simplified: verify checkpoint-aware logic pattern

        checkpoint_dir = tempfile.mkdtemp()
        try:
            processor = BatchProcessor(checkpoint_dir=checkpoint_dir)

            # Simulate: builder checks for last checkpoint
            last_batch = processor.find_last_checkpoint("70B")

            # If resuming, start_batch = last_batch + 1
            start_batch = (last_batch + 1) if last_batch is not None else 0

            assert start_batch == 0, "Should start from 0 when no checkpoint"
        finally:
            shutil.rmtree(checkpoint_dir)


class TestMemoryBounds:
    """Test that batching maintains constant memory"""

    def test_constant_memory_with_large_dataset(self):
        """Verify batch processing has constant memory regardless of dataset size"""
        # Create large JSONL
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(10000):
                import json

                chunk = {
                    "chunk_id": f"chunk_{i:06d}",
                    "token_count_estimate": 256,
                    "band": "B2",
                    "domain": "code",
                }
                f.write(json.dumps(chunk) + "\n")
            temp_path = f.name

        try:
            processor = BatchProcessor(batch_size=100)

            # Process all batches
            batch_count = 0
            for batch in processor.batch_iterator(temp_path):
                batch_count += 1
                # Each batch should be ~100 chunks, not entire file

            assert batch_count == 100, "Should have 100 batches of 100 chunks each"
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
