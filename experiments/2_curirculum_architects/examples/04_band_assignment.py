"""Example: Band assignment post-processing.

This example demonstrates:
- Running extraction WITHOUT band assignment
- Assigning bands after extraction by reading metadata layer
- Using different band configurations
- Updating metadata layer with band information
"""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from curriculum_extractor.scripts.assign_bands import BandAssigner


def create_sample_metadata(output_dir: Path) -> None:
    """Create sample metadata layer (simulating extraction output)."""
    records = []

    # Create records with varying difficulty scores
    for i in range(50):
        # Spread scores across difficulty range
        score = i / 50  # 0.0 to 0.98

        records.append(
            {
                "uuid": f"uuid-{i:04d}",
                "id": f"record-{i}",
                "file_path": "sample.parquet",
                "difficulty_score": round(score, 3),
                "difficulty_level": f"L{min(int(score * 6), 5)}",
                "readability_score": round(1 - score, 3),  # Inverse
                "modality_primary": "code" if i % 3 == 0 else "text",
                "curriculum_version": "0.2",
            }
        )

    # Create partitioned output
    partition_dir = output_dir / "file_name=sample"
    partition_dir.mkdir(parents=True)

    table = pa.Table.from_pylist(records)
    pq.write_table(table, partition_dir / "metadata.parquet")


def main():
    """Demonstrate band assignment workflow."""
    parser = argparse.ArgumentParser(
        description="Band assignment post-processing example."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for metadata. Defaults to ./downloads/04_band_assignment/",
    )
    args = parser.parse_args()

    curriculum_path = Path(__file__).parent.parent / "curriculum.yaml"

    print("=" * 80)
    print("BAND ASSIGNMENT - Post-Processing Example")
    print("=" * 80)

    script_dir = Path(__file__).parent
    default_output_dir = script_dir / "downloads" / "04_band_assignment"
    output_dir = Path(args.output) if args.output else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tmpdir = output_dir

    # Step 1: Create sample metadata (simulating extraction output)
    metadata_dir = tmpdir / "metadata"
    create_sample_metadata(metadata_dir)
    print(f"\n[1] Created sample metadata layer at: {metadata_dir}")

    # Read and display before band assignment
    table_before = pq.read_table(metadata_dir)
    print(f"    Records: {len(table_before)}")
    print(f"    Columns: {table_before.schema.names}")

    # Step 2: Initialize band assigner with default config
    print("\n" + "-" * 40)
    print("[2] Assigning bands with DEFAULT config")
    print("-" * 40)

    assigner = BandAssigner(curriculum_path)

    print("Band definitions:")
    for band in assigner.bands:
        print(
            f"  {band['id']}: {band['name']} (max_score: {band.get('max_score', 1.0)})"
        )

    # Process and get stats
    output_with_bands = tmpdir / "metadata_with_bands"
    stats = assigner.process_metadata_directory(metadata_dir, output_with_bands)

    print("\nResults:")
    print(f"  Total records: {stats['total_records']}")
    print("  Band distribution:")
    for band_id, count in sorted(stats["band_distribution"].items()):
        pct = count / stats["total_records"] * 100
        print(f"    {band_id}: {count} ({pct:.1f}%)")

    # Read and display after band assignment
    table_after = pq.read_table(output_with_bands)
    print("\nNew columns added: band_id, band_name, band_score")

    # Show sample records
    df = table_after.to_pandas()
    print("\nSample records:")
    sample = df[["id", "difficulty_score", "band_id", "band_name", "band_score"]].head(
        10
    )
    print(sample.to_string(index=False))

    # Step 3: Custom band configuration
    print("\n" + "-" * 40)
    print("[3] Assigning bands with CUSTOM config")
    print("-" * 40)

    custom_config = {
        "score_column": "difficulty_score",
        "weights": {
            "difficulty_score": 0.7,
            "readability_score": 0.3,  # Weighted blend
        },
    }

    assigner_custom = BandAssigner(curriculum_path, band_config=custom_config)

    output_custom = tmpdir / "metadata_custom_bands"
    stats_custom = assigner_custom.process_metadata_directory(
        metadata_dir, output_custom
    )

    print("Custom weighted band distribution:")
    for band_id, count in sorted(stats_custom["band_distribution"].items()):
        pct = count / stats_custom["total_records"] * 100
        print(f"  {band_id}: {count} ({pct:.1f}%)")

    # Step 4: Single record band assignment
    print("\n" + "-" * 40)
    print("[4] Single record band assignment")
    print("-" * 40)

    test_records = [
        {"difficulty_score": 0.05, "id": "easy"},
        {"difficulty_score": 0.35, "id": "medium"},
        {"difficulty_score": 0.75, "id": "hard"},
        {"difficulty_score": 0.95, "id": "expert"},
    ]

    for record in test_records:
        band_info = assigner.assign_band(record)
        print(
            f"  {record['id']} (score={record['difficulty_score']}) -> "
            f"{band_info['band_id']} ({band_info['band_name']})"
        )

    print("\n[OK] Band assignment example complete.")
    print("\nWorkflow Summary:")
    print("  1. Run extraction pipeline (bands NOT assigned)")
    print("  2. Read metadata layer")
    print("  3. Assign bands based on extracted metrics")
    print("  4. Write updated metadata with band columns")


if __name__ == "__main__":
    main()
