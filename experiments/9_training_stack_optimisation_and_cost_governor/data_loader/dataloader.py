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

    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logger = logging.getLogger("spdl.main")
    logger.setLevel(args.log_level.upper())
    # Remove all handlers if already present (avoid duplicate logs)
    if logger.hasHandlers():
        logger.handlers.clear()
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(log_format))
    logger.addHandler(ch)
    # File handler
    fh = logging.FileHandler("run.log", mode="a")
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)

    seq_len = SEQUENCE_LENGTH
    dtype = np.dtype(DTYPE)
    model = DummyModel()
    pipeline = build_pipeline(args.token_folder, seq_len=seq_len, dtype=dtype)
    logger.info(f"Running dataloader on: {args.token_folder} | seq_len={seq_len} | dtype={dtype}")
    import time
    batch_count = 0
    total_tokens = 0
    start_time = time.time()
    with pipeline.auto_stop():
        for step, batch in enumerate(pipeline):
            import time as _time
            batch_start_time = _time.time()
            # Try to extract offset info if available (assume batch is a list of tensors with .offset_start/.offset_end attributes if present)
            offset_start, offset_end = None, None
            if isinstance(batch, list) and len(batch) > 0:
                # If batch elements have offset info, extract min/max
                try:
                    offset_start = min(getattr(x, 'offset_start', None) for x in batch if hasattr(x, 'offset_start'))
                    offset_end = max(getattr(x, 'offset_end', None) for x in batch if hasattr(x, 'offset_end'))
                except Exception:
                    offset_start, offset_end = None, None
                batch = torch.stack(batch)
            logger.info(f"Step {step}: batch shape {batch.shape}")
            if offset_start is not None and offset_end is not None:
                logger.info(f"Batch offsets: start={offset_start}, end={offset_end}")
            outputs = model(batch)
            logger.info(f"Output shape: {outputs.shape}")
            batch_time = _time.time() - batch_start_time
            logger.info(f"Batch {step} processing time: {batch_time:.4f} seconds")
            batch_count += 1
            total_tokens += batch.numel()
            if step >= args.batches - 1:
                break
    elapsed = time.time() - start_time
    throughput = total_tokens / elapsed if elapsed > 0 else 0
    logger.info(f"Completed: {batch_count} batches, {total_tokens} tokens processed in {elapsed:.2f} seconds.")
    logger.info(f"Throughput: {throughput:.2f} tokens/sec")

if __name__ == "__main__":
    main()
