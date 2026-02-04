"""Example: Parquet file processing with state management.

This example demonstrates:
- Processing parquet files with metadata output
- Incremental processing with StateManager
- Resuming failed/interrupted jobs
- Rejection layer output
"""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from curriculum_extractor import CurriculumExtractor
from curriculum_extractor.core.state_manager import StateManager


def create_sample_parquet(output_path: Path, num_records: int = 100) -> Path:
    """Create a sample parquet file for testing."""
    records = []
    for i in range(num_records):
        # Vary text complexity
        if i % 10 == 0:
            text = "Short."  # Will likely be rejected
        elif i % 3 == 0:
            text = f"""
def function_{i}():
    '''Function {i} docstring.'''
    result = {i} * 2
    return result
            """
        else:
            text = f"""
This is sample document number {i}. It contains enough text to be 
processed by the curriculum extraction pipeline. The content varies 
to demonstrate different difficulty levels and modalities. Some 
documents contain technical terms while others are more casual.
            """

        records.append(
            {
                "id": f"record_{i:04d}",
                "text": text,
                "source": "test_data",
                "lang": "en",
            }
        )

    table = pa.Table.from_pylist(records)
    parquet_path = output_path / "sample_data.parquet"
    pq.write_table(table, parquet_path)

    return parquet_path


def main():
    """Demonstrate parquet processing with state management, with user input for parquet file and output dir."""

    parser = argparse.ArgumentParser(
        description="Parquet file processing with state management."
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default=None,
        help="Path to input parquet file. If not provided, a sample will be created.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for metadata, rejections, and state. Defaults to ./downloads/02_parquet_processing/",
    )
    args = parser.parse_args()

    curriculum_path = Path(__file__).parent.parent / "curriculum.yaml"

    print("=" * 80)
    print("CURRICULUM EXTRACTOR - Parquet Processing")
    print("=" * 80)

    script_dir = Path(__file__).parent
    default_output_dir = script_dir / "downloads" / "02_parquet_processing"
    output_dir = Path(args.output) if args.output else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data directory for sample parquet (if needed)
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    if args.parquet:
        parquet_path = Path(args.parquet)
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Provided parquet file does not exist: {parquet_path}"
            )
        print(f"\n[OK] Using user-provided parquet file: {parquet_path}")
    else:
        parquet_path = create_sample_parquet(data_dir, num_records=100)
        print(f"\n[OK] Created sample data: {parquet_path}")

    # Setup output directories
    metadata_dir = output_dir / "metadata"
    rejection_dir = output_dir / "rejections"
    state_dir = output_dir / "state"
    metadata_dir.mkdir(exist_ok=True)
    rejection_dir.mkdir(exist_ok=True)
    state_dir.mkdir(exist_ok=True)

    # Create state manager for incremental processing
    state_manager = StateManager(state_dir)

    # Initialize extractor with all components
    extractor = CurriculumExtractor(
        curriculum_path,
        state_manager=state_manager,
        metadata_output_path=str(metadata_dir),
        rejection_output_path=str(rejection_dir),
        track_timing=True,
    )

    print("\n[OK] Initialized extractor with:")
    print(f"     - {len(extractor.plugins)} metrics")
    print(f"     - State management: {state_dir}")
    print(f"     - Metadata output: {metadata_dir}")
    print(f"     - Rejection output: {rejection_dir}")

    # Process parquet file
    print("\n" + "-" * 40)
    print("FIRST RUN: Processing parquet file...")
    print("-" * 40)

    result = extractor.process_parquet(parquet_path, batch_size=50)

    print(f"\nResult: {result['status']}")
    print(f"  Total rows: {result.get('total_rows', 'n/a')}")
    print(f"  Processed: {result.get('processed_rows', 'n/a')}")
    print(f"  Rejected: {result.get('rejected_rows', 'n/a')}")

    if result.get("timing"):
        print("\nTiming:")
        for name, stats in result["timing"].items():
            print(
                f"  {name}: {stats['mean_ms']:.2f}ms avg, {stats['total_seconds']:.2f}s total"
            )

    # Check state
    stats = state_manager.get_stats()
    print("\nState Manager Stats:")
    print(f"  Completed files: {stats['completed_files']}")
    print(f"  Total rows processed: {stats['total_rows_processed']}")
    print(f"  Total rows rejected: {stats['total_rows_rejected']}")

    # Try to reprocess - should be skipped
    print("\n" + "-" * 40)
    print("SECOND RUN: Attempting reprocess...")
    print("-" * 40)

    result2 = extractor.process_parquet(parquet_path)
    print(f"Result: {result2['status']} (reason: {result2.get('reason', 'n/a')})")

    # Check output files
    print("\n" + "-" * 40)
    print("OUTPUT FILES:")
    print("-" * 40)

    if metadata_dir.exists():
        for p in metadata_dir.rglob("*.parquet"):
            table = pq.read_table(p)
            print(f"\nMetadata: {p.relative_to(output_dir)}")
            print(f"  Rows: {len(table)}")
            print(f"  Columns: {table.schema.names[:5]}...")

    if rejection_dir.exists():
        for p in rejection_dir.rglob("*.parquet"):
            table = pq.read_table(p)
            print(f"\nRejections: {p.relative_to(output_dir)}")
            print(f"  Rows: {len(table)}")
            if len(table) > 0:
                df = table.to_pandas()
                print(f"  Reasons: {df['rejected_reason'].value_counts().to_dict()}")

    # Demonstrate reset and reprocess
    print("\n" + "-" * 40)
    print("RESET AND REPROCESS:")
    print("-" * 40)

    state_manager.reset()
    print("State reset - file will be reprocessed")

    result3 = extractor.process_parquet(parquet_path)
    print(f"Result: {result3['status']}")
    print(f"  Processed: {result3['processed_rows']}")

    print("\n[OK] Parquet processing example complete.")


if __name__ == "__main__":
    main()
