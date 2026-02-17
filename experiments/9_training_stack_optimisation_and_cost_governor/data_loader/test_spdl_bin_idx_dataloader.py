import os
import torch
from spdl_dataloader import build_pipeline, DummyModel


def test_spdl_bin_idx_dataloader(token_folder=None):
    import sys
    # Allow token folder override from command line
    if token_folder is None and len(sys.argv) > 1:
        token_folder = sys.argv[1]
    if token_folder is None:
        token_folder = os.path.join(os.path.dirname(__file__), "Test_data")
    # Use config for seq_len/dtype if available
    from common import SEQUENCE_LENGTH, DTYPE
    import numpy as np
    seq_len = SEQUENCE_LENGTH
    dtype = np.dtype(DTYPE)
    batch_count = 0
    total_tokens = 0
    model = DummyModel()

    pipeline = build_pipeline(token_folder, seq_len=seq_len, dtype=dtype)
    print(f"Testing SPDL bin/idx dataloader in: {token_folder}")
    with pipeline.auto_stop():
        for step, batch in enumerate(pipeline):
            # batch is a list of tensors, stack to get [batch_size, seq_len]
            if isinstance(batch, list):
                batch = torch.stack(batch)
            assert batch.shape[1] == seq_len, f"Batch sequence length mismatch: {batch.shape[1]} != {seq_len}"
            assert batch.dtype in (torch.uint16, torch.int32), f"Batch dtype mismatch: {batch.dtype}"
            outputs = model(batch)
            assert outputs.shape[0] == batch.shape[0], "Model output batch size mismatch"
            batch_count += 1
            total_tokens += batch.numel()
            if step < 3:
                print(f"Step {step}: batch shape {batch.shape}, output shape {outputs.shape}")
            if step >= 9:
                break  # Only test 10 batches for speed
    print(f"Test completed: {batch_count} batches, {total_tokens} tokens processed.")
    assert batch_count > 0, "No batches processed!"
    print("SPDL bin/idx dataloader test PASSED.")

if __name__ == "__main__":
    import sys
    token_folder = sys.argv[1] if len(sys.argv) > 1 else None
    test_spdl_bin_idx_dataloader(token_folder)
