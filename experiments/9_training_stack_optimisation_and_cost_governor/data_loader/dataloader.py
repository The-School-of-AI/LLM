import argparse
import logging
from logging.handlers import RotatingFileHandler

import numpy as np
import torch
from common import DTYPE, LOG_BACKUP_COUNT, LOG_FILE_SIZE, SEQUENCE_LENGTH
from spdl_dataloader import DummyModel, build_pipeline


# --- Argument Parsing ---
def parse_args():
    """
    Parse command-line arguments for the dataloader script.
    Returns:
        argparse.Namespace: Parsed arguments with token_folder, batches, and log_level.
    """
    parser = argparse.ArgumentParser(description="SPDL bin/idx dataloader runner")
    parser.add_argument(
        "--token-folder",
        required=True,
        help="Path to folder containing .bin/.idx files",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=10,
        help="Number of batches to process (default: 10)",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (default: INFO)"
    )
    return parser.parse_args()


# --- Logger Setup ---
def setup_logger(log_level):
    """
    Set up and return a logger for the dataloader script with file rotation.
    Args:
        log_level (str): Logging level (e.g., 'INFO', 'DEBUG').
    Returns:
        logging.Logger: Configured logger instance.
    """
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logger = logging.getLogger("spdl.main")
    logger.setLevel(log_level.upper())
    if logger.hasHandlers():
        logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(log_format))
    logger.addHandler(ch)
    # Rotating file handler: 100MB per file, 20 backups
    fh = RotatingFileHandler(
        "dataloader.log", maxBytes=LOG_FILE_SIZE, backupCount=LOG_BACKUP_COUNT
    )
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)
    return logger


# --- Batch Processing ---
def process_batch(step, batch, model, logger):
    """
    Process a single batch: log shape, offsets, run model, and log timing.
    Args:
        step (int): Batch index.
        batch (list[Tensor] or Tensor): Batch data.
        model (nn.Module): Model to run on batch.
        logger (logging.Logger): Logger instance.
    Returns:
        tuple: (batch, outputs, batch_time, offset_start, offset_end)
    """
    import time as _time

    batch_start_time = _time.time()
    offset_start, offset_end = None, None
    if isinstance(batch, list) and len(batch) > 0:
        try:
            offset_start = min(
                getattr(x, "offset_start", None)
                for x in batch
                if hasattr(x, "offset_start")
            )
            offset_end = max(
                getattr(x, "offset_end", None)
                for x in batch
                if hasattr(x, "offset_end")
            )
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
    return batch, outputs, batch_time, offset_start, offset_end


# --- Pipeline Runner ---
def run_pipeline(token_folder, batches, logger):
    """
    Run the SPDL dataloader pipeline for a given number of batches.
    Args:
        token_folder (str): Path to .bin/.idx files.
        batches (int): Number of batches to process.
        logger (logging.Logger): Logger instance.
    """
    import time

    seq_len = SEQUENCE_LENGTH
    dtype = np.dtype(DTYPE)
    model = DummyModel()
    pipeline = build_pipeline(token_folder, seq_len=seq_len, dtype=dtype)
    logger.info(
        f"Running dataloader on: {token_folder} | seq_len={seq_len} | dtype={dtype}"
    )
    batch_count = 0
    total_tokens = 0
    start_time = time.time()
    with pipeline.auto_stop():
        for step, batch in enumerate(pipeline):
            batch, outputs, batch_time, offset_start, offset_end = process_batch(
                step, batch, model, logger
            )
            batch_count += 1
            total_tokens += batch.numel()
            if step >= batches - 1:
                break
    elapsed = time.time() - start_time
    throughput = total_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Completed: {batch_count} batches, {total_tokens} tokens processed in {elapsed:.2f} seconds."
    )
    logger.info(f"Throughput: {throughput:.2f} tokens/sec")


# --- Main Entrypoint ---
def main():
    """
    Main entrypoint for the dataloader script. Parses arguments, sets up logging, and runs pipeline.
    """
    args = parse_args()
    logger = setup_logger(args.log_level)
    run_pipeline(args.token_folder, args.batches, logger)


if __name__ == "__main__":
    main()
