import torch
import sys
from spdl_dataloader import build_pipeline, DummyModel

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run SPDL dataloader with specified parquet files.")
    parser.add_argument('parquet_files', nargs='+', help='Paths to parquet files to load')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = get_device()
    model = DummyModel().to(device)
    print(f"Using device: {device}")
    print(f"Loading parquet files: {args.parquet_files}")
    pipeline = build_pipeline(args.parquet_files)
    with pipeline.auto_stop():
        for step, batch in enumerate(pipeline):
            input_ids = torch.stack(batch).to(device, non_blocking=(device=="cuda"))
            outputs = model(input_ids)
            print(f"Step {step}, Output: {outputs}")
            if step >= 2:
                break  # Only run a few steps for local testing
