import os
import shutil
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import time
from spdl_dataloader import build_pipeline, DummyModel
from common import TOKENS_COLUMN

def generate_parquet_files(num_files=2, records_per_file=500000, out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    file_paths = []
    for i in range(num_files):
        tokens = [np.random.randint(0, 50000, 128).tolist() for _ in range(records_per_file)]
        table = pa.Table.from_arrays([pa.array(tokens, type=pa.list_(pa.int64()))], names=[TOKENS_COLUMN])
        file_path = os.path.join(out_dir, f"test_shard_{i}.parquet")
        pq.write_table(table, file_path)
        file_paths.append(file_path)
    return file_paths

def cleanup_parquet_files(out_dir="data"):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

def test_spdl_dataloader():
    # Generate test data: 1 million records
    start_time = time.time()
    file_paths = generate_parquet_files(num_files=2, records_per_file=500000, out_dir="data")
    data_gen_time = time.time() - start_time
    print(f"Data generation time: {data_gen_time:.2f} seconds")
    assert all(os.path.exists(fp) for fp in file_paths), "Parquet files not created!"

    # Build pipeline and model
    pipeline = build_pipeline(file_paths)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DummyModel().to(device)

    # Test pipeline
    batch_count = 0
    processing_start = time.time()
    with pipeline.auto_stop():
        for step, batch in enumerate(pipeline):
            input_ids = torch.stack(batch).to(device)
            outputs = model(input_ids)
            batch_count += 1
            if step >= 9:  # Process 10 batches for better measurement
                break
    processing_time = time.time() - processing_start
    total_records = 1000000
    throughput = total_records / processing_time if processing_time > 0 else 0

    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Batches processed: {batch_count}")
    print(f"Total records: {total_records}")
    print(f"Throughput: {throughput:.2f} records/second")
    assert batch_count > 0, "No batches processed!"

    # Cleanup
    cleanup_parquet_files("data")
    assert not os.path.exists("data"), "Test data directory not cleaned up!"

if __name__ == "__main__":
    test_spdl_dataloader()
    print("Test completed successfully.")
