import argparse
import logging
from spdl_dataloader import build_pipeline, DummyModel
from common import SEQUENCE_LENGTH, DTYPE
import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="SPDL bin/idx dataloader runner")
    parser.add_argument("--token-folder", required=True, help="Path to folder containing .bin/.idx files")
    parser.add_argument("--batches", type=int, default=10, help="Number of batches to process (default: 10)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("spdl.main")

    seq_len = SEQUENCE_LENGTH
    dtype = np.dtype(DTYPE)
    model = DummyModel()
    pipeline = build_pipeline(args.token_folder, seq_len=seq_len, dtype=dtype)
    logger.info(f"Running dataloader on: {args.token_folder} | seq_len={seq_len} | dtype={dtype}")
    batch_count = 0
    total_tokens = 0
    with pipeline.auto_stop():
        for step, batch in enumerate(pipeline):
            if isinstance(batch, list):
                batch = torch.stack(batch)
            logger.info(f"Step {step}: batch shape {batch.shape}")
            outputs = model(batch)
            logger.info(f"Output shape: {outputs.shape}")
            batch_count += 1
            total_tokens += batch.numel()
            if step >= args.batches - 1:
                break
    logger.info(f"Completed: {batch_count} batches, {total_tokens} tokens processed.")

if __name__ == "__main__":
    main()
