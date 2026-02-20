import importlib.util
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

# Allow running this validation script in environments without SPDL installed.
try:
    from spdl.pipeline import PipelineBuilder as _PipelineBuilder  # noqa: F401
except ModuleNotFoundError:
    spdl_mod = types.ModuleType("spdl")
    pipeline_mod = types.ModuleType("spdl.pipeline")

    class _DummyPipeline:
        def __init__(self, source, batch_size):
            self._source = source
            self._batch_size = int(batch_size)

        def start(self):
            return None

        def stop(self):
            return None

        def __iter__(self):
            bucket = []
            for item in self._source:
                bucket.append(item)
                if len(bucket) == self._batch_size:
                    yield bucket
                    bucket = []

    class PipelineBuilder:
        def __init__(self):
            self._source = None
            self._batch_size = 1

        def add_source(self, source):
            self._source = source
            return self

        def aggregate(self, batch_size):
            self._batch_size = batch_size
            return self

        def add_sink(self, _prefetch_buffer):
            return self

        def build(self, num_threads=1):  # noqa: ARG002
            return _DummyPipeline(self._source, self._batch_size)

    pipeline_mod.PipelineBuilder = PipelineBuilder
    sys.modules["spdl"] = spdl_mod
    sys.modules["spdl.pipeline"] = pipeline_mod

# Load only the required src modules to avoid importing src.__init__.
src_pkg = types.ModuleType("src")
src_pkg.__path__ = [str(SRC_DIR)]
src_pkg.__package__ = "src"
sys.modules["src"] = src_pkg


def _load_src_module(name: str):
    module_name = f"src.{name}"
    module_path = SRC_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_load_src_module("utils")
shard_tracker_mod = _load_src_module("shard_tracker")
data_mod = _load_src_module("data")

get_dataloaders = data_mod.get_dataloaders
build_spdl_pipeline = data_mod.build_spdl_pipeline
SPDLIterableDataset = data_mod.SPDLIterableDataset
ShardTracker = shard_tracker_mod.ShardTracker


def test_pipeline():
    td_path = Path(__file__).resolve().parent / "_tmp_validation_run"
    if td_path.exists():
        shutil.rmtree(td_path, ignore_errors=True)
    td_path.mkdir(parents=True, exist_ok=True)
    td = str(td_path)
    try:
        # create mock shards
        # 4 shards, each with 10 tokens
        seq_len = 2
        for i in range(4):
            bin_path = os.path.join(td, f"shard_00{i}.bin")
            idx_path = os.path.join(td, f"shard_00{i}.idx")
            tokens = np.arange(i * 10, (i + 1) * 10, dtype=np.uint32)

            with open(bin_path, "wb") as f:
                f.write(tokens.tobytes())

            with open(idx_path, "wb") as f:
                f.write(b"MMEDPKT1")  # header length 8
                # offsets
                offsets = np.array([0, 40], dtype=np.uint64)
                f.write(offsets.tobytes())

        # create tracker and mark shard 1 and 2 as processed
        manifest = os.path.join(td, "consumed.json")
        tracker = ShardTracker(manifest)
        tracker.mark_processed("shard_001.bin")
        tracker.mark_processed("shard_002.bin")

        # Test 1: use pipeline directly
        bin_idx_source = data_mod.bin_idx_source

        source = bin_idx_source(
            td,
            seq_len=seq_len,
            dtype=np.uint32,
            rank=0,
            world_size=1,
            exclude_files=tracker.get_processed_files(),
            on_shard_complete=tracker.mark_processed,
        )
        yielded_tensors = list(source)
        # Should only get tensors from shard 0 and 3
        # shard 0: [0,1], [2,3], [4,5], [6,7], [8,9]
        # shard 3: [30,31], ...

        assert len(yielded_tensors) == 10
        assert yielded_tensors[0][0].item() == 0
        assert yielded_tensors[-1][0].item() == 38

        # test dataset wrapper
        dataset = SPDLIterableDataset(
            shard_dir=td,
            seq_len=seq_len,
            batch_size=2,
            dtype=np.uint32,
            shard_tracker=tracker,
        )
        dataset_iter = iter(dataset)
        batches = list(dataset_iter)

        # At this point, shard 0 and 3 were marked processed by the source test,
        # so everything is processed if we use the same tracker.
        # Wait, the source test called tracker.mark_processed, so tracker now has 0 and 3 as well!
        assert len(batches) == 0, f"Expected 0 batches, got {len(batches)}"
    finally:
        shutil.rmtree(td_path, ignore_errors=True)


if __name__ == "__main__":
    test_pipeline()
    print("ALL TESTS PASSED")
