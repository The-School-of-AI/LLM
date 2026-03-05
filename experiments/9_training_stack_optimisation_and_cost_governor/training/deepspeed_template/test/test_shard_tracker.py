"""
Tests for ShardTracker — processed-shard exclusion and manifest persistence.

Tests cover:
  - mark / exclude basic workflow
  - JSON round-trip persistence (save + load)
  - reset clears state
  - atomic write safety
  - integration with bin_idx_source exclusion
  - thread-safety of concurrent mark_processed calls
  - manifest portability (basenames only)
"""

import json
import os
import sys
import threading
import types

import pytest

# numpy/torch are only needed for the integration tests (bin_idx_source).
# The core ShardTracker tests are pure-Python.
try:
    import numpy as np
    import torch  # noqa: F401

    _HAS_NUMPY_TORCH = True
except ImportError:
    _HAS_NUMPY_TORCH = False


# ── Mock deepspeed so it can be imported on any machine ──────────────────────
class _DummyFlopsProfiler:
    pass


_ds = types.ModuleType("deepspeed")
_ds_prof = types.ModuleType("deepspeed.profiling")
_ds_acc = types.ModuleType("deepspeed.accelerator")
_ds_flops = types.ModuleType("deepspeed.profiling.flops_profiler")
_ds_flops.FlopsProfiler = _DummyFlopsProfiler
sys.modules.update(
    {
        "deepspeed": _ds,
        "deepspeed.profiling": _ds_prof,
        "deepspeed.accelerator": _ds_acc,
        "deepspeed.profiling.flops_profiler": _ds_flops,
    }
)

# ── Add project root so src.* sub-imports resolve ────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

# Stub src package to avoid heavy imports (same pattern as test_dataloader.py)
import importlib  # noqa: E402
import importlib.util  # noqa: E402

_src_pkg = types.ModuleType("src")
_src_pkg.__path__ = [os.path.join(_PROJECT_ROOT, "src")]
_src_pkg.__package__ = "src"
sys.modules["src"] = _src_pkg

# Load utils to get print_rank_0 available for data.py
_utils_spec = importlib.util.spec_from_file_location(
    "src.utils", os.path.join(_PROJECT_ROOT, "src", "utils.py")
)
_utils_mod = importlib.util.module_from_spec(_utils_spec)
sys.modules["src.utils"] = _utils_mod
_utils_spec.loader.exec_module(_utils_mod)

# Load shard_tracker
_tracker_spec = importlib.util.spec_from_file_location(
    "src.shard_tracker", os.path.join(_PROJECT_ROOT, "src", "shard_tracker.py")
)
_tracker_mod = importlib.util.module_from_spec(_tracker_spec)
sys.modules["src.shard_tracker"] = _tracker_mod
_tracker_spec.loader.exec_module(_tracker_mod)

ShardTracker = _tracker_mod.ShardTracker


# ============================================================================
# Helper: create mock .bin/.idx shard files (requires numpy)
# ============================================================================

if _HAS_NUMPY_TORCH:

    def _create_mock_shard(shard_dir: str, name: str, num_tokens: int, dtype=np.uint32):
        """Create a .bin/.idx pair with sequential token IDs."""
        tokens = np.arange(num_tokens, dtype=dtype)
        bin_path = os.path.join(shard_dir, name)
        tokens.tofile(bin_path)

        # The .idx file has an 8-byte header then uint64 offsets marking regions.
        # For simplicity, one region spanning the entire file.
        idx_path = bin_path.replace(".bin", ".idx")
        with open(idx_path, "wb") as f:
            f.write(b"\x00" * 8)  # 8-byte header
            start = np.array([0], dtype=np.uint64)
            end = np.array([num_tokens * dtype(0).itemsize], dtype=np.uint64)
            f.write(start.tobytes())
            f.write(end.tobytes())


# ============================================================================
# Tests: ShardTracker core functionality
# ============================================================================


class TestShardTrackerBasic:
    """Tests for ShardTracker mark / exclude / persistence."""

    def test_mark_and_exclude(self, tmp_path):
        manifest = tmp_path / "consumed_shards.json"
        tracker = ShardTracker(manifest)

        files = ["shard_000.bin", "shard_001.bin", "shard_002.bin"]

        # Nothing excluded yet
        assert tracker.exclude(files) == files
        assert tracker.num_processed == 0

        # Mark first two
        tracker.mark_processed("shard_000.bin", rank=0)
        tracker.mark_processed("shard_001.bin", rank=0)

        remaining = tracker.exclude(files)
        assert remaining == ["shard_002.bin"]
        assert tracker.num_processed == 2

    def test_exclude_with_full_paths(self, tmp_path):
        """Paths with directories are reduced to basenames for matching."""
        manifest = tmp_path / "consumed_shards.json"
        tracker = ShardTracker(manifest)

        tracker.mark_processed("/mnt/nvme/data/shard_000.bin", rank=0)

        # Both basename and full path should be excluded
        full = ["/other/path/shard_000.bin", "/other/path/shard_001.bin"]
        result = tracker.exclude(full)
        assert result == ["/other/path/shard_001.bin"]

    def test_is_processed(self, tmp_path):
        manifest = tmp_path / "consumed_shards.json"
        tracker = ShardTracker(manifest)

        assert not tracker.is_processed("shard_000.bin")
        tracker.mark_processed("shard_000.bin")
        assert tracker.is_processed("shard_000.bin")
        assert "shard_000.bin" in tracker

    def test_persistence_round_trip(self, tmp_path):
        manifest = tmp_path / "consumed_shards.json"

        # Create tracker, mark some shards, save
        t1 = ShardTracker(manifest, auto_save=False)
        t1.mark_processed("shard_000.bin", rank=0, source_dir="/data/train")
        t1.mark_processed("shard_001.bin", rank=1, source_dir="/data/train")
        t1.save()

        assert manifest.exists()

        # Load in new tracker instance
        t2 = ShardTracker(manifest)
        assert t2.num_processed == 2
        assert t2.is_processed("shard_000.bin")
        assert t2.is_processed("shard_001.bin")
        assert not t2.is_processed("shard_002.bin")

        # Verify JSON structure
        with open(manifest) as f:
            data = json.load(f)
        assert data["version"] == 1
        assert "shard_000.bin" in data["consumed_shards"]
        assert "shard_001.bin" in data["consumed_shards"]
        assert data["consumed_shards"]["shard_000.bin"]["rank"] == 0

    def test_reset(self, tmp_path):
        manifest = tmp_path / "consumed_shards.json"
        tracker = ShardTracker(manifest)

        tracker.mark_processed("shard_000.bin")
        assert tracker.num_processed == 1

        tracker.reset()
        assert tracker.num_processed == 0
        assert not tracker.is_processed("shard_000.bin")

        # Manifest on disk should also be empty
        with open(manifest) as f:
            data = json.load(f)
        assert data["consumed_shards"] == {}

    def test_mark_many_processed(self, tmp_path):
        manifest = tmp_path / "consumed_shards.json"
        tracker = ShardTracker(manifest)

        files = ["shard_000.bin", "shard_001.bin", "shard_002.bin"]
        tracker.mark_many_processed(files, rank=0, source_dir="/data")

        assert tracker.num_processed == 3
        for f in files:
            assert tracker.is_processed(f)

    def test_auto_save_creates_manifest(self, tmp_path):
        manifest = tmp_path / "consumed_shards.json"
        tracker = ShardTracker(manifest, auto_save=True)

        assert not manifest.exists()

        tracker.mark_processed("shard_000.bin")
        assert manifest.exists()

    def test_no_auto_save(self, tmp_path):
        manifest = tmp_path / "consumed_shards.json"
        tracker = ShardTracker(manifest, auto_save=False)

        tracker.mark_processed("shard_000.bin")
        assert not manifest.exists()

        tracker.save()
        assert manifest.exists()

    def test_get_manifest(self, tmp_path):
        manifest = tmp_path / "consumed_shards.json"
        tracker = ShardTracker(manifest)
        tracker.mark_processed("shard_000.bin", rank=0)

        m = tracker.get_manifest()
        assert m["version"] == 1
        assert "shard_000.bin" in m["consumed_shards"]

    def test_repr_and_len(self, tmp_path):
        manifest = tmp_path / "consumed_shards.json"
        tracker = ShardTracker(manifest)
        tracker.mark_processed("shard_000.bin")

        assert len(tracker) == 1
        assert "num_processed=1" in repr(tracker)

    def test_idempotent_mark(self, tmp_path):
        """Marking the same file twice overwrites metadata but doesn't duplicate."""
        manifest = tmp_path / "consumed_shards.json"
        tracker = ShardTracker(manifest)

        tracker.mark_processed("shard_000.bin", rank=0)
        tracker.mark_processed("shard_000.bin", rank=1)

        assert tracker.num_processed == 1
        m = tracker.get_manifest()
        # Last write wins
        assert m["consumed_shards"]["shard_000.bin"]["rank"] == 1


class TestShardTrackerConcurrency:
    """thread-safety of concurrent writes."""

    def test_concurrent_marks(self, tmp_path):
        manifest = tmp_path / "consumed_shards.json"
        tracker = ShardTracker(manifest, auto_save=False)

        errors = []

        def mark_range(start, count):
            try:
                for i in range(start, start + count):
                    tracker.mark_processed(f"shard_{i:04d}.bin", rank=0)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=mark_range, args=(0, 50)),
            threading.Thread(target=mark_range, args=(50, 50)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert tracker.num_processed == 100
        tracker.save()

        # Reload and verify
        t2 = ShardTracker(manifest)
        assert t2.num_processed == 100


@pytest.mark.skipif(not _HAS_NUMPY_TORCH, reason="numpy/torch not installed")
class TestShardTrackerWithBinIdxSource:
    """Integration tests: ShardTracker + bin_idx_source."""

    def _setup_shards(self, tmp_path, num_shards=5, tokens_per_shard=128):
        """Create mock shard files and return the directory path."""
        shard_dir = str(tmp_path / "shards")
        os.makedirs(shard_dir, exist_ok=True)
        for i in range(num_shards):
            _create_mock_shard(shard_dir, f"shard_{i:03d}.bin", tokens_per_shard)
        return shard_dir

    def test_exclude_shards_from_bin_idx_source(self, tmp_path):
        """bin_idx_source should skip shards listed in exclude_files."""
        shard_dir = self._setup_shards(tmp_path, num_shards=5, tokens_per_shard=128)
        seq_len = 32

        # Stub SPDL import for bin_idx_source
        try:
            _spdl = types.ModuleType("spdl")
            _spdl_pipeline = types.ModuleType("spdl.pipeline")
            _spdl_pipeline.PipelineBuilder = type("PipelineBuilder", (), {})
            sys.modules["spdl"] = _spdl
            sys.modules["spdl.pipeline"] = _spdl_pipeline

            _data_spec = importlib.util.spec_from_file_location(
                "src.data", os.path.join(_PROJECT_ROOT, "src", "data.py")
            )
            _data_mod = importlib.util.module_from_spec(_data_spec)
            sys.modules["src.data"] = _data_mod
            _data_spec.loader.exec_module(_data_mod)

            bin_idx_source = _data_mod.bin_idx_source

            # Without exclusion: count sequences from all 5 shards
            all_seqs = list(bin_idx_source(shard_dir, seq_len=seq_len, dtype=np.uint32))
            total_all = len(all_seqs)
            assert total_all > 0

            # Per shard: 128 tokens / 32 seq_len = 4 sequences each, 5 shards = 20
            assert total_all == 20

            # Exclude 2 shards
            exclude = {"shard_000.bin", "shard_001.bin"}
            partial_seqs = list(
                bin_idx_source(
                    shard_dir, seq_len=seq_len, dtype=np.uint32, exclude_files=exclude
                )
            )
            assert len(partial_seqs) == 12  # 3 shards * 4 sequences

        finally:
            # Clean up stubbed modules
            for key in ["spdl", "spdl.pipeline", "src.data"]:
                sys.modules.pop(key, None)

    def test_on_shard_complete_callback(self, tmp_path):
        """on_shard_complete should fire once per fully-consumed shard."""
        shard_dir = self._setup_shards(tmp_path, num_shards=3, tokens_per_shard=64)
        seq_len = 32

        try:
            _spdl = types.ModuleType("spdl")
            _spdl_pipeline = types.ModuleType("spdl.pipeline")
            _spdl_pipeline.PipelineBuilder = type("PipelineBuilder", (), {})
            sys.modules["spdl"] = _spdl
            sys.modules["spdl.pipeline"] = _spdl_pipeline

            _data_spec = importlib.util.spec_from_file_location(
                "src.data", os.path.join(_PROJECT_ROOT, "src", "data.py")
            )
            _data_mod = importlib.util.module_from_spec(_data_spec)
            sys.modules["src.data"] = _data_mod
            _data_spec.loader.exec_module(_data_mod)

            bin_idx_source = _data_mod.bin_idx_source

            completed = []

            def on_complete(name):
                completed.append(name)

            seqs = list(
                bin_idx_source(
                    shard_dir,
                    seq_len=seq_len,
                    dtype=np.uint32,
                    on_shard_complete=on_complete,
                )
            )
            assert len(seqs) == 6  # 3 shards * 2 sequences
            assert completed == ["shard_000.bin", "shard_001.bin", "shard_002.bin"]

        finally:
            for key in ["spdl", "spdl.pipeline", "src.data"]:
                sys.modules.pop(key, None)

    def test_full_workflow_exclude_resume(self, tmp_path):
        """Simulate: process 3 shards → save manifest → resume → only 2 remain."""
        shard_dir = self._setup_shards(tmp_path, num_shards=5, tokens_per_shard=64)
        manifest_path = tmp_path / "consumed_shards.json"
        seq_len = 32

        try:
            _spdl = types.ModuleType("spdl")
            _spdl_pipeline = types.ModuleType("spdl.pipeline")
            _spdl_pipeline.PipelineBuilder = type("PipelineBuilder", (), {})
            sys.modules["spdl"] = _spdl
            sys.modules["spdl.pipeline"] = _spdl_pipeline

            _data_spec = importlib.util.spec_from_file_location(
                "src.data", os.path.join(_PROJECT_ROOT, "src", "data.py")
            )
            _data_mod = importlib.util.module_from_spec(_data_spec)
            sys.modules["src.data"] = _data_mod
            _data_spec.loader.exec_module(_data_mod)

            bin_idx_source = _data_mod.bin_idx_source

            # Phase 1: process first 3 shards, track them
            tracker = ShardTracker(manifest_path)

            completed_phase1 = []
            seqs1 = list(
                bin_idx_source(
                    shard_dir,
                    seq_len=seq_len,
                    dtype=np.uint32,
                    exclude_files=tracker.get_processed_files(),
                    on_shard_complete=lambda name: (
                        completed_phase1.append(name),
                        tracker.mark_processed(name, rank=0, source_dir=shard_dir),
                    ),
                )
            )
            assert len(seqs1) == 10  # 5 shards * 2 sequences

            # Simulate crash after processing 3 shards — only mark first 3
            tracker_partial = ShardTracker(manifest_path)
            tracker_partial.reset()
            for name in completed_phase1[:3]:
                tracker_partial.mark_processed(name, rank=0)
            tracker_partial.save()

            # Phase 2: resume — only 2 shards should be read
            tracker_resume = ShardTracker(manifest_path)
            assert tracker_resume.num_processed == 3

            seqs2 = list(
                bin_idx_source(
                    shard_dir,
                    seq_len=seq_len,
                    dtype=np.uint32,
                    exclude_files=tracker_resume.get_processed_files(),
                )
            )
            assert len(seqs2) == 4  # 2 remaining shards * 2 sequences

        finally:
            for key in ["spdl", "spdl.pipeline", "src.data"]:
                sys.modules.pop(key, None)


# ============================================================================
# Run with pytest
# ============================================================================

if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "--tb=short"])
