"""Example: Deterministic batch creation for training.

This example demonstrates:
- BatchCreator for reproducible training data loading
- Deterministic ordering with xxhash
- Auto-increment and seek operations
- Stratified sampling
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from curriculum_reader import BatchConfig, BatchCreator, MetadataReader
from curriculum_reader.core.batch_creator import StratifiedBatchCreator


def create_sample_metadata(base_dir: Path, num_records: int = 500) -> Path:
    """Create sample metadata for batch creation demo."""
    records = []

    for i in range(num_records):
        band = i % 6
        records.append(
            {
                "uuid": f"uuid-{i:05d}",
                "id": f"record-{i:05d}",
                "file_path": f"source_{i // 100}.parquet",
                "difficulty_score": round(band / 6 + 0.05, 3),
                "band_id": f"B{band}",
                "modality_primary": ["text", "code", "math"][i % 3],
                "curriculum_version": "0.2",
            }
        )

    metadata_dir = base_dir / "metadata"
    partition_dir = metadata_dir / "file_name=all"
    partition_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(records), partition_dir / "data.parquet")

    return metadata_dir


def main():
    """Demonstrate deterministic batch creation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Deterministic batch creation example."
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to existing metadata directory. If not provided, sample data will be created.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for sample data and state. Defaults to ./downloads/06_batch_creation/",
    )
    parser.add_argument(
        "--num-records",
        type=int,
        default=500,
        help="Number of records to create in sample data (default: 500).",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("BATCH CREATOR - Deterministic Training Batches")
    print("=" * 80)

    script_dir = Path(__file__).parent
    default_output_dir = script_dir / "downloads" / "06_batch_creation"
    output_dir = Path(args.output) if args.output else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tmpdir = output_dir

    # Use user-provided metadata or create sample data
    if args.metadata:
        metadata_dir = Path(args.metadata)
        if not metadata_dir.exists():
            raise FileNotFoundError(
                f"Metadata directory does not exist: {metadata_dir}"
            )
        print(f"\n[OK] Using user-provided metadata: {metadata_dir}")
    else:
        # Create sample data
        metadata_dir = create_sample_metadata(tmpdir, num_records=args.num_records)
        print(f"\n[OK] Created {args.num_records} sample records")

    reader = MetadataReader(metadata_dir)

    # ============================================================
    # PART 1: Basic Batch Creation
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 1: Basic Batch Creation")
    print("=" * 60)

    config = BatchConfig(batch_size=50, seed=42)
    creator = BatchCreator(reader, config)

    print("\nBatch configuration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Seed: {config.seed}")
    print(f"  Total records: {creator.state.total_records}")
    print(f"  Total batches: {creator.state.total_batches}")

    # Get first batch
    batch_0 = creator.get_batch(0)
    print(f"\nBatch 0: {len(batch_0)} records")
    print(f"  First ID: {batch_0.column('id')[0].as_py()}")
    print(f"  Last ID: {batch_0.column('id')[-1].as_py()}")

    # ============================================================
    # PART 2: Determinism Verification
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 2: Determinism Verification")
    print("=" * 60)

    # Create two independent creators with same seed
    creator_a = BatchCreator(reader, BatchConfig(batch_size=50, seed=42))
    creator_b = BatchCreator(reader, BatchConfig(batch_size=50, seed=42))

    batch_a = creator_a.get_batch(0)
    batch_b = creator_b.get_batch(0)

    ids_a = batch_a.column("id").to_pylist()
    ids_b = batch_b.column("id").to_pylist()

    print("\nSame seed (42):")
    print(f"  Batch A first 5 IDs: {ids_a[:5]}")
    print(f"  Batch B first 5 IDs: {ids_b[:5]}")
    print(f"  Identical: {ids_a == ids_b} ✓")

    # Different seed gives different order
    creator_c = BatchCreator(reader, BatchConfig(batch_size=50, seed=999))
    batch_c = creator_c.get_batch(0)
    ids_c = batch_c.column("id").to_pylist()

    print("\nDifferent seed (999):")
    print(f"  Batch C first 5 IDs: {ids_c[:5]}")
    print(f"  Different from A: {ids_a != ids_c} ✓")

    # ============================================================
    # PART 3: Auto-Increment Mode
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 3: Auto-Increment Mode")
    print("=" * 60)

    state_path = output_dir / "batch_state"
    creator = BatchCreator(reader, config, state_path=state_path)

    print(f"\nStarting position: batch {creator.get_current_batch_number()}")

    # Get batches with auto-increment
    for _ in range(3):
        batch = creator.get_batch()  # No batch number = auto-increment
        print(
            f"Got batch {creator.get_current_batch_number() - 1}, "
            f"next will be {creator.get_current_batch_number()}"
        )

    # Seek to specific position
    creator.seek(5)
    print(f"\nSeeked to batch 5, current: {creator.get_current_batch_number()}")

    # Reset to beginning
    creator.reset()
    print(f"Reset, current: {creator.get_current_batch_number()}")

    # ============================================================
    # PART 4: Iteration Pattern
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 4: Iteration Pattern")
    print("=" * 60)

    # Iterate through a range of batches
    print("\nIterating through batches 0-4:")
    for batch_num, batch in creator.iter_batches(start_batch=0, end_batch=5):
        band_dist = {}
        for band in batch.column("band_id").to_pylist():
            band_dist[band] = band_dist.get(band, 0) + 1
        print(
            f"  Batch {batch_num}: {len(batch)} records, bands: {dict(sorted(band_dist.items()))}"
        )

    # ============================================================
    # PART 5: Batch Metadata
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 5: Batch Metadata in Records")
    print("=" * 60)

    batch_with_info = creator.get_batch(0, include_batch_info=True)
    print("\nBatch with metadata columns:")
    print(f"  Columns: {batch_with_info.schema.names}")
    print(f"  _batch_number: {batch_with_info.column('_batch_number')[0].as_py()}")
    print(f"  _batch_seed: {batch_with_info.column('_batch_seed')[0].as_py()}")

    # ============================================================
    # PART 6: Stratified Sampling
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 6: Stratified Sampling")
    print("=" * 60)

    # Regular batch - may have uneven band distribution
    regular_creator = BatchCreator(reader, BatchConfig(batch_size=60, seed=42))
    regular_batch = regular_creator.get_batch(0)

    regular_dist = {}
    for band in regular_batch.column("band_id").to_pylist():
        regular_dist[band] = regular_dist.get(band, 0) + 1

    print("\nRegular batch band distribution:")
    for band, count in sorted(regular_dist.items()):
        print(f"  {band}: {count}")

    # Stratified batch - maintains proportions
    strat_creator = StratifiedBatchCreator(
        reader,
        BatchConfig(batch_size=60, seed=42),
        stratify_column="band_id",
    )
    strat_batch = strat_creator.get_batch(0)

    strat_dist = {}
    for band in strat_batch.column("band_id").to_pylist():
        strat_dist[band] = strat_dist.get(band, 0) + 1

    print("\nStratified batch band distribution:")
    for band, count in sorted(strat_dist.items()):
        print(f"  {band}: {count}")

    # ============================================================
    # PART 7: Training Loop Pattern
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 7: Training Loop Pattern")
    print("=" * 60)

    print(
        """
# Example training loop with deterministic batches:

config = BatchConfig(batch_size=1000, seed=42)
creator = BatchCreator(reader, config, state_path="./training_state")

for epoch in range(num_epochs):
    # Reset to beginning of dataset for each epoch
    creator.reset()
    
    for batch_num, batch in creator.iter_batches():
        # batch is always the same for the same batch_num and seed
        train_step(batch)
        
        # Can checkpoint at any batch number
        if batch_num % 100 == 0:
            checkpoint(batch_num)
    
    # For next epoch with different order, change seed:
    creator = BatchCreator(reader, BatchConfig(batch_size=1000, seed=42+epoch))
"""
    )

    print("\n[OK] Batch creation example complete.")


if __name__ == "__main__":
    main()
