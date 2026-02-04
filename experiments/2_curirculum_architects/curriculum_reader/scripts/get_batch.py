#!/usr/bin/env python3
"""Deterministic batch retrieval script.

This script provides a CLI for retrieving deterministic batches from the metadata layer.

Usage:
    # Get specific batch
    python -m curriculum_reader.scripts.get_batch \
        --metadata ./output/metadata \
        --batch 0 \
        --seed 42

    # Get next batch (auto-increment)
    python -m curriculum_reader.scripts.get_batch \
        --metadata ./output/metadata \
        --next \
        --seed 42

    # Get batch with filtering
    python -m curriculum_reader.scripts.get_batch \
        --metadata ./output/metadata \
        --batch 0 \
        --filter "band_assignment_band == 'B3'" \
        --seed 42

    # Stratified batch by band
    python -m curriculum_reader.scripts.get_batch \
        --metadata ./output/metadata \
        --batch 0 \
        --stratify band_assignment_band \
        --seed 42
        
    # Export batch to parquet
    python -m curriculum_reader.scripts.get_batch \
        --metadata ./output/metadata \
        --batch 0 \
        --output batch_0.parquet
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrieve deterministic batches from metadata layer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--metadata",
        "-m",
        required=True,
        help="Path to metadata layer",
    )

    parser.add_argument(
        "--batch",
        "-b",
        type=int,
        help="Batch number to retrieve (0-indexed)",
    )

    parser.add_argument(
        "--next",
        "-n",
        action="store_true",
        help="Get next batch (auto-increment)",
    )

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for deterministic ordering (default: 42)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size (default: 1024)",
    )

    parser.add_argument(
        "--stratify",
        help="Column to stratify by (e.g., band_assignment_band)",
    )

    parser.add_argument(
        "--filter",
        help="Filter expression (e.g., \"band_assignment_band == 'B3'\")",
    )

    parser.add_argument(
        "--columns",
        help="Comma-separated list of columns to include",
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output file (parquet or jsonl)",
    )

    parser.add_argument(
        "--state-path",
        help="Path to store batch state for auto-increment",
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Show batch info without retrieving data",
    )

    parser.add_argument(
        "--s3",
        action="store_true",
        help="Enable S3 mode",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format (to stdout)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    from curriculum_reader import BatchConfig, BatchCreator, MetadataReader
    from curriculum_reader.core.batch_creator import StratifiedBatchCreator

    # Initialize reader
    fs = None
    if args.s3:
        import s3fs

        fs = s3fs.S3FileSystem()

    metadata_reader = MetadataReader(args.metadata, filesystem=fs)

    # Parse columns
    columns = None
    if args.columns:
        columns = [c.strip() for c in args.columns.split(",")]

    # Parse filter (simplified - real implementation would need proper parsing)
    filter_expr = None
    if args.filter:
        # Simple equality filter parsing
        # e.g., "band_assignment_band == 'B3'"
        # This is a simplified implementation
        try:
            parts = args.filter.split("==")
            if len(parts) == 2:
                col = parts[0].strip()
                val = parts[1].strip().strip("'\"")
                filter_expr = ds.field(col) == val
        except Exception as e:
            print(f"Warning: Could not parse filter: {e}")

    # Create config
    config = BatchConfig(
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=True,
        columns=columns,
        filter_expr=filter_expr,
        stratify_by=args.stratify,
    )

    # Create batch creator
    state_path = Path(args.state_path) if args.state_path else None

    if args.stratify:
        creator = StratifiedBatchCreator(
            reader=metadata_reader,
            config=config,
            state_path=state_path,
        )
    else:
        creator = BatchCreator(
            reader=metadata_reader,
            config=config,
            state_path=state_path,
        )

    # Show info
    if args.info:
        print(f"\n{'='*60}")
        print("BATCH INFO")
        print(f"{'='*60}")
        print(f"Total records: {creator.state.total_records:,}")
        print(f"Total batches: {creator.state.total_batches:,}")
        print(f"Batch size: {config.batch_size}")
        print(f"Seed: {config.seed}")
        print(f"Current batch: {creator.state.current_batch}")
        if args.stratify:
            print(f"Stratified by: {args.stratify}")
        print(f"{'='*60}")
        return 0

    # Get batch
    batch_number = None
    if args.batch is not None:
        batch_number = args.batch
    elif args.next:
        batch_number = None  # Will use auto-increment
    else:
        print("Error: Specify --batch <number> or --next")
        return 1

    print(
        f"Retrieving batch {batch_number if batch_number is not None else '(next)'}..."
    )
    print(f"  Seed: {config.seed}")
    print(f"  Batch size: {config.batch_size}")

    table = creator.get_batch(batch_number)

    print(f"  Retrieved {len(table)} records")

    # Get actual batch number (in case of auto-increment)
    actual_batch = (
        batch_number if batch_number is not None else (creator.state.current_batch - 1)
    )

    # Output
    if args.output:
        output_path = Path(args.output)

        if output_path.suffix == ".parquet":
            pq.write_table(table, output_path)
            print(f"Wrote to {output_path}")

        elif output_path.suffix in (".jsonl", ".json"):
            records = table.to_pylist()
            with open(output_path, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
            print(f"Wrote to {output_path}")

    elif args.json:
        records = table.to_pylist()
        print(
            json.dumps(
                {
                    "batch_number": actual_batch,
                    "seed": config.seed,
                    "count": len(records),
                    "records": records[:10] if len(records) > 10 else records,
                    "truncated": len(records) > 10,
                },
                indent=2,
            )
        )

    else:
        # Print sample
        print("\nSample records (first 5):")
        print("-" * 60)
        records = table.to_pylist()
        for i, record in enumerate(records[:5]):
            print(f"\n[{i}] uuid: {record.get('uuid', 'N/A')[:16]}...")
            print(f"    id: {record.get('id', 'N/A')}")
            if "band_assignment_band" in record:
                print(f"    band: {record.get('band_assignment_band')}")
            if "difficulty_score" in record:
                print(f"    difficulty: {record.get('difficulty_score')}")

    # Show next batch info
    print(f"\nNext batch number: {creator.state.current_batch}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
