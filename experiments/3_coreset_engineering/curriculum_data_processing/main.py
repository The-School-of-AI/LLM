import yaml
import argparse
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from src import (
    setup_logger,
    discover_dataset_structure,
    get_pyarrow_filesystem,
    DataProcessor,
)

# Load environment variables from .env file
load_dotenv()


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Professional Data Processing & Dedup Pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        help="Specific bands to process (e.g. B0 B1). If omitted, finds all.",
    )
    parser.add_argument("--output-dir", type=str, help="Directory to save output files")
    parser.add_argument(
        "--output-mode",
        type=str,
        choices=["single", "per_band", "sharded"],
        help="Output mode",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        help="Record count per shard (for sharded/per_band modes)",
    )
    parser.add_argument("--workers", type=int, help="Number of parallel workers")
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )

    args = parser.parse_args()

    # Load Config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    # CLI Overrides config
    output_dir = args.output_dir if args.output_dir else config["output"]["dir"]
    output_mode = args.output_mode if args.output_mode else config["output"]["mode"]
    shard_size = (
        args.shard_size if args.shard_size else config["output"].get("shard_size")
    )

    import os

    max_workers = (
        args.workers
        if args.workers
        else config["processing"].get("max_workers", os.cpu_count())
    )

    # Setup Logging
    log_file = config.get("logging", {}).get("log_file", "pipeline.log")
    logger = setup_logger(
        name="data_pipeline", log_file=log_file, level=getattr(logging, args.log_level)
    )

    logger.info("Initializing Pipeline")

    # Discover Filesystem
    fs = get_pyarrow_filesystem(region=config["s3"]["region"])

    logger.info("Discovering dataset structure in S3...")
    structure = discover_dataset_structure(
        config["s3"]["bucket"], config["s3"]["base_prefix"]
    )

    available_bands = sorted(
        list(set(b for b_list in structure.values() for b in b_list))
    )
    target_bands = args.bands if args.bands else available_bands

    if not target_bands:
        logger.error("❌ No bands found or specified. Exiting.")
        sys.exit(1)

    logger.info(f"Target bands: {target_bands} | Mode: {output_mode}")

    # Initialize Processor and Writer
    from src.data_processor import RecordWriter  # Importing here to ensure visibility

    writer = RecordWriter(
        output_dir=output_dir,
        mode=output_mode,
        shard_size=shard_size,
        base_name="consolidated",
    )
    processor = DataProcessor(config, fs)

    # Process all selected bands
    processor.process_all(target_bands, structure, writer, max_workers=max_workers)

    logger.info("Pipeline execution complete.")


if __name__ == "__main__":
    main()
