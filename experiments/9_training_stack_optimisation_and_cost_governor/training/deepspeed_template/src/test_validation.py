import os
import tempfile
import numpy as np
import torch
from pathlib import Path
from data import get_dataloaders, build_spdl_pipeline, SPDLIterableDataset
from shard_tracker import ShardTracker

def test_pipeline():
    with tempfile.TemporaryDirectory() as td:
        # create mock shards
        # 4 shards, each with 10 tokens
        seq_len = 2
        for i in range(4):
            bin_path = os.path.join(td, f"shard_00{i}.bin")
            idx_path = os.path.join(td, f"shard_00{i}.idx")
            tokens = np.arange(i*10, (i+1)*10, dtype=np.uint32)
            
            with open(bin_path, "wb") as f:
                f.write(tokens.tobytes())
                
            with open(idx_path, "wb") as f:
                f.write(b"MMEDPKT1") # header length 8
                # offsets
                offsets = np.array([0, 40], dtype=np.uint64)
                f.write(offsets.tobytes())
                
        # create tracker and mark shard 1 and 2 as processed
        manifest = os.path.join(td, "consumed.json")
        tracker = ShardTracker(manifest)
        tracker.mark_processed("shard_001.bin")
        tracker.mark_processed("shard_002.bin")
        
        # Test 1: use pipeline directly
        from data import bin_idx_source
        
        source = bin_idx_source(td, seq_len=seq_len, dtype=np.uint32, rank=0, world_size=1, exclude_files=tracker.get_processed_files(), on_shard_complete=tracker.mark_processed)
        yielded_tensors = list(source)
        # Should only get tensors from shard 0 and 3
        # shard 0: [0,1], [2,3], [4,5], [6,7], [8,9]
        # shard 3: [30,31], ...
        
        assert len(yielded_tensors) == 10
        assert yielded_tensors[0][0].item() == 0
        assert yielded_tensors[-1][0].item() == 38
        
        # test dataset wrapper
        dataset = SPDLIterableDataset(shard_dir=td, seq_len=seq_len, batch_size=2, dtype=np.uint32, shard_tracker=tracker)
        dataset_iter = iter(dataset)
        batches = list(dataset_iter)
        
        # At this point, shard 0 and 3 were marked processed by the source test, 
        # so everything is processed if we use the same tracker. 
        # Wait, the source test called tracker.mark_processed, so tracker now has 0 and 3 as well!
        assert len(batches) == 0, f"Expected 0 batches, got {len(batches)}"

if __name__ == '__main__':
    test_pipeline()
    print("ALL TESTS PASSED")
