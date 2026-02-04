#!/usr/bin/env python3
"""Pipeline runner script for curriculum metadata extraction.

This script provides a CLI for running the extraction pipeline on S3 or local data.

Usage:
    # Local processing
    uv run python -m curriculum_extractor.scripts.run_pipeline \
        --input /Users/hemanthk/Downloads/Chrome/sample_dataset/ \
        --output ./downloads/pipeline/ \
        --curriculum ./curriculum.yaml

    # S3 processing with distributed workers
    uv run python -m curriculum_extractor.scripts.run_pipeline \
        --input s3://bucket/input-data \
        --output s3://bucket/output \
        --curriculum ./curriculum.yaml \
        --s3 \
        --workers 2
        
    # Resume from previous run (incremental)
    uv run python -m curriculum_extractor.scripts.run_pipeline \
        --input s3://bucket/input-data \
        --output s3://bucket/output \
        --curriculum ./curriculum.yaml \
        --s3 \
        --resume

    # Full refresh (reprocess everything)
    uv run python -m curriculum_extractor.scripts.run_pipeline \
        --input s3://bucket/input-data \
        --output s3://bucket/output \
        --curriculum ./curriculum.yaml \
        --s3 \
        --full-refresh
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run curriculum metadata extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input path (directory containing parquet files)",
    )

    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output base path for metadata and rejection layers",
    )

    parser.add_argument(
        "--curriculum",
        "-c",
        required=True,
        help="Path to curriculum YAML configuration",
    )

    parser.add_argument(
        "--metrics-config",
        help="Path to metrics configuration YAML (optional)",
    )

    parser.add_argument(
        "--s3",
        action="store_true",
        help="Enable S3 mode for input/output",
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)",
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=10000,
        help="Batch size for processing (default: 10000)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip already processed files)",
    )

    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Full refresh - reprocess all files",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files to process without actually processing",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def run_local_single_file(
    input_path: str,
    output_path: str,
    curriculum_path: str,
    metrics_config_path: Optional[str] = None,
    batch_size: int = 10000,
):
    """Run extraction on a single local file."""
    from curriculum_extractor import CurriculumExtractor
    from curriculum_extractor.core.state_manager import StateManager

    # Setup paths
    metadata_path = Path(output_path) / "metadata"
    rejection_path = Path(output_path) / "rejections"
    state_path = Path(output_path) / "state"

    # Create directories
    metadata_path.mkdir(parents=True, exist_ok=True)
    rejection_path.mkdir(parents=True, exist_ok=True)
    state_path.mkdir(parents=True, exist_ok=True)

    # Initialize
    state_manager = StateManager(state_path)

    extractor = CurriculumExtractor(
        curriculum_path=curriculum_path,
        metrics_config_path=metrics_config_path,
        state_manager=state_manager,
        metadata_output_path=str(metadata_path),
        rejection_output_path=str(rejection_path),
    )

    # Process
    def progress(total):
        print(f"  Processed {total} rows...", end="\r")

    result = extractor.process_parquet(
        input_path=input_path,
        batch_size=batch_size,
        progress_callback=progress,
    )

    print(f"\nResult: {result}")
    return result


def run_local_directory(
    input_path: str,
    output_path: str,
    curriculum_path: str,
    metrics_config_path: Optional[str] = None,
    batch_size: int = 10000,
    full_refresh: bool = False,
):
    """Run extraction on a local directory."""
    from curriculum_extractor import CurriculumExtractor
    from curriculum_extractor.core.state_manager import StateManager

    # Setup paths
    input_dir = Path(input_path)
    metadata_path = Path(output_path) / "metadata"
    rejection_path = Path(output_path) / "rejections"
    state_path = Path(output_path) / "state"

    # Create directories
    metadata_path.mkdir(parents=True, exist_ok=True)
    rejection_path.mkdir(parents=True, exist_ok=True)
    state_path.mkdir(parents=True, exist_ok=True)

    # Initialize state
    state_manager = StateManager(state_path)

    if full_refresh:
        state_manager.reset()

    # Find input files
    input_files = list(input_dir.glob("*.parquet"))
    print(f"Found {len(input_files)} parquet files")

    # Register and filter
    pending_files = state_manager.register_files([str(f) for f in input_files])
    print(f"Files to process: {len(pending_files)}")

    if not pending_files:
        print("All files already processed. Use --full-refresh to reprocess.")
        return {"status": "completed", "message": "All files already processed"}

    # Process each file
    total_rows = 0
    total_rejected = 0

    for i, file_path in enumerate(pending_files):
        print(f"\n[{i+1}/{len(pending_files)}] Processing: {Path(file_path).name}")

        extractor = CurriculumExtractor(
            curriculum_path=curriculum_path,
            metrics_config_path=metrics_config_path,
            state_manager=state_manager,
            metadata_output_path=str(metadata_path),
            rejection_output_path=str(rejection_path),
        )

        try:
            result = extractor.process_parquet(
                input_path=file_path,
                batch_size=batch_size,
            )

            total_rows += result.get("processed_rows", 0)
            total_rejected += result.get("rejected_rows", 0)

            print(
                f"  Processed: {result.get('processed_rows', 0)}, Rejected: {result.get('rejected_rows', 0)}"
            )

        except Exception as e:
            print(f"  ERROR: {e}")
            state_manager.mark_failed(file_path, str(e))

    stats = state_manager.get_stats()
    print(f"\n{'='*60}")
    print("COMPLETED")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Completed: {stats['completed_files']}")
    print(f"  Failed: {stats['failed_files']}")
    print(f"  Total rows processed: {total_rows}")
    print(f"  Total rows rejected: {total_rejected}")
    print(f"{'='*60}")

    return stats


def run_distributed(
    input_path: str,
    output_path: str,
    curriculum_path: str,
    metrics_config_path: Optional[str] = None,
    s3_mode: bool = False,
    num_workers: int = 2,
    batch_size: int = 10000,
    full_refresh: bool = False,
):
    """Run distributed extraction with Ray."""
    from curriculum_extractor.core.distributed import DistributedExtractor

    metadata_path = f"{output_path}/metadata"
    rejection_path = f"{output_path}/rejections"
    state_path = f"{output_path}/state"

    extractor = DistributedExtractor(
        curriculum_path=curriculum_path,
        metadata_output_path=metadata_path,
        rejection_output_path=rejection_path,
        state_path=state_path,
        s3_bucket="enabled" if s3_mode else None,
        num_workers=num_workers,
        metrics_config_path=metrics_config_path,
    )

    if full_refresh:
        print("Full refresh: resetting state...")
        extractor.reset_state()

    # List input files
    input_files = extractor.list_input_files(input_path)
    print(f"Found {len(input_files)} input files")

    # Process
    start_time = time.time()

    def progress(file: str, rows: int):
        print(f"  Completed: {Path(file).name} ({rows} rows)")

    result = extractor.process_files(
        input_files=input_files,
        batch_size=batch_size,
        progress_callback=progress,
    )

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"COMPLETED in {elapsed:.1f}s")
    print(f"  Total files: {result['total_files']}")
    print(f"  Completed: {result['completed_files']}")
    print(f"  Failed: {result['failed_files']}")
    print(f"  Total rows processed: {result['total_rows_processed']}")
    print(f"  Total rows rejected: {result['total_rows_rejected']}")
    print(f"{'='*60}")

    return result


def main():
    args = parse_args()

    print("=" * 60)
    print("CURRICULUM METADATA EXTRACTION PIPELINE")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Curriculum: {args.curriculum}")
    print(f"S3 Mode: {args.s3}")
    print(f"Workers: {args.workers}")
    print(f"Batch Size: {args.batch_size}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Would process files from:", args.input)
        # Just list files
        if args.s3:
            import s3fs

            fs = s3fs.S3FileSystem()
            files = fs.glob(f"{args.input.rstrip('/')}/*.parquet")
            for f in files[:10]:
                print(f"  s3://{f}")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more")
        else:
            files = list(Path(args.input).glob("*.parquet"))
            for f in files[:10]:
                print(f"  {f}")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more")
        return 0

    # Check curriculum exists
    if not Path(args.curriculum).exists():
        print(f"ERROR: Curriculum file not found: {args.curriculum}")
        return 1

    try:
        if args.workers > 1 or args.s3:
            # Distributed processing
            _ = run_distributed(
                input_path=args.input,
                output_path=args.output,
                curriculum_path=args.curriculum,
                metrics_config_path=args.metrics_config,
                s3_mode=args.s3,
                num_workers=args.workers,
                batch_size=args.batch_size,
                full_refresh=args.full_refresh,
            )
        else:
            # Local processing
            input_path = Path(args.input)
            if input_path.is_file():
                _ = run_local_single_file(
                    input_path=str(input_path),
                    output_path=args.output,
                    curriculum_path=args.curriculum,
                    metrics_config_path=args.metrics_config,
                    batch_size=args.batch_size,
                )
            else:
                _ = run_local_directory(
                    input_path=str(input_path),
                    output_path=args.output,
                    curriculum_path=args.curriculum,
                    metrics_config_path=args.metrics_config,
                    batch_size=args.batch_size,
                    full_refresh=args.full_refresh,
                )

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
