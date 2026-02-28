"""Example: Reading and analyzing metadata layers.

This example demonstrates:
- MetadataReader for accessing metadata
- MetadataAnalyzer for statistics
- RejectionReader for quality analysis
- Various query patterns
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from curriculum_reader import MetadataReader, RejectionReader
from curriculum_reader.core.analyzer import MetadataAnalyzer


def create_sample_data(base_dir: Path) -> tuple[Path, Path]:
    """Create sample metadata and rejection layers."""

    # Metadata layer
    metadata_records = []
    for i in range(200):
        band = i % 6
        metadata_records.append(
            {
                "uuid": f"uuid-{i:04d}",
                "id": f"record-{i}",
                "file_path": f"file_{i // 50}.parquet",
                "difficulty_score": round(band / 6 + 0.05, 3),
                "difficulty_level": f"L{band}",
                "readability_score": round(0.9 - band * 0.1, 3),
                "modality_primary": ["text", "code", "math"][i % 3],
                "band_id": f"B{band}",
                "band_name": [
                    "Nursery",
                    "Elementary",
                    "Middle",
                    "High",
                    "Undergrad",
                    "Graduate",
                ][band],
                "curriculum_version": "0.2",
            }
        )

    metadata_dir = base_dir / "metadata"
    partition_dir = metadata_dir / "file_name=combined"
    partition_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(metadata_records), partition_dir / "data.parquet"
    )

    # Rejection layer
    rejection_records = [
        {
            "uuid": "rej-001",
            "id": "bad-1",
            "file_path": "file_0.parquet",
            "rejected_reason": "Text too short",
            "rejected_at": "min_length",
        },
        {
            "uuid": "rej-002",
            "id": "bad-2",
            "file_path": "file_0.parquet",
            "rejected_reason": "Text too short",
            "rejected_at": "min_length",
        },
        {
            "uuid": "rej-003",
            "id": "bad-3",
            "file_path": "file_1.parquet",
            "rejected_reason": "Non-English content",
            "rejected_at": "language",
        },
        {
            "uuid": "rej-004",
            "id": "bad-4",
            "file_path": "file_2.parquet",
            "rejected_reason": "Low quality score",
            "rejected_at": "quality",
        },
    ]

    rejection_dir = base_dir / "rejections"
    partition_dir = rejection_dir / "file_name=combined"
    partition_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(rejection_records), partition_dir / "data.parquet"
    )

    return metadata_dir, rejection_dir


def main():
    """Demonstrate metadata reading and analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Metadata reading and analysis example."
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to existing metadata directory. If not provided, sample data will be created.",
    )
    parser.add_argument(
        "--rejections",
        type=str,
        default=None,
        help="Path to existing rejections directory. If not provided, sample data will be created.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for sample data. Defaults to ./downloads/05_metadata_analysis/",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("CURRICULUM READER - Metadata Analysis")
    print("=" * 80)

    script_dir = Path(__file__).parent
    default_output_dir = script_dir / "downloads" / "05_metadata_analysis"
    output_dir = Path(args.output) if args.output else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tmpdir = output_dir

    # Use user-provided paths or create sample data
    if args.metadata and args.rejections:
        metadata_dir = Path(args.metadata)
        rejection_dir = Path(args.rejections)
        if not metadata_dir.exists():
            raise FileNotFoundError(
                f"Metadata directory does not exist: {metadata_dir}"
            )
        if not rejection_dir.exists():
            raise FileNotFoundError(
                f"Rejections directory does not exist: {rejection_dir}"
            )
        print("\n[OK] Using user-provided data")
    else:
        # Create sample data
        metadata_dir, rejection_dir = create_sample_data(tmpdir)
        print("\n[OK] Created sample data")

    # ============================================================
    # PART 1: MetadataReader basics
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 1: MetadataReader Basics")
    print("=" * 60)

    reader = MetadataReader(metadata_dir)

    # Basic info
    print(f"\nTotal records: {reader.count_rows()}")
    print(f"Columns: {reader.get_column_names()}")

    # Schema
    schema = reader.get_schema()
    print("\nSchema:")
    for field in schema[:5]:
        print(f"  {field.name}: {field.type}")
    print(f"  ... ({len(schema) - 5} more)")

    # Read specific columns
    subset = reader.read_all(columns=["id", "difficulty_score", "band_id"])
    print(f"\nSubset read: {len(subset)} rows x {len(subset.schema)} cols")

    # Random sample
    sample = reader.sample(n=5, seed=42)
    print("\nRandom sample (seed=42):")
    df = sample.to_pandas()
    print(df[["id", "difficulty_score", "band_id"]].to_string(index=False))

    # Find by ID
    record = reader.get_record_by_id("record-42")
    if record:
        print("\nFound record-42:")
        print(f"  Band: {record['band_id']} ({record['band_name']})")
        print(f"  Difficulty: {record['difficulty_score']}")

    # ============================================================
    # PART 2: MetadataAnalyzer
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 2: MetadataAnalyzer Statistics")
    print("=" * 60)

    analyzer = MetadataAnalyzer(reader)

    # Summary
    summary = analyzer.get_summary()
    print("\nDataset Summary:")
    print(f"  Total records: {summary.total_records}")
    print(f"  Unique sources: {summary.unique_sources}")

    # Band distribution
    band_dist = analyzer.get_band_distribution()
    print("\nBand Distribution:")
    for band, count in sorted(band_dist.items()):
        pct = count / summary.total_records * 100
        bar = "█" * int(pct / 5)
        print(f"  {band}: {count:4d} ({pct:5.1f}%) {bar}")

    # Modality distribution
    print("\nModality Distribution:")
    modality_dist = analyzer.get_column_distribution("modality_primary")
    for mod, count in sorted(modality_dist.items(), key=lambda x: -x[1]):
        pct = count / summary.total_records * 100
        print(f"  {mod}: {count} ({pct:.1f}%)")

    # Numeric column stats
    print("\nDifficulty Score Statistics:")
    stats = analyzer.get_numeric_stats("difficulty_score")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Min:  {stats['min']:.3f}")
    print(f"  Max:  {stats['max']:.3f}")
    print(f"  Std:  {stats['std']:.3f}")

    # ============================================================
    # PART 3: RejectionReader
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 3: RejectionReader Analysis")
    print("=" * 60)

    rej_reader = RejectionReader(rejection_dir)

    print(f"\nTotal rejections: {rej_reader.count_rows()}")

    rejections = rej_reader.read_all().to_pandas()

    print("\nRejections by reason:")
    for reason, count in rejections.groupby("rejected_reason").size().items():
        print(f"  {reason}: {count}")

    print("\nRejections by metric:")
    for metric, count in rejections.groupby("rejected_at").size().items():
        print(f"  {metric}: {count}")

    print("\nRejections by source file:")
    for fp, count in rejections.groupby("file_path").size().items():
        print(f"  {fp}: {count}")

    # ============================================================
    # PART 4: Advanced Queries
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 4: Advanced Queries")
    print("=" * 60)

    # Filter to specific band
    table = reader.read_all()
    df = table.to_pandas()

    hard_content = df[df["band_id"].isin(["B4", "B5"])]
    print(f"\nHard content (B4+B5): {len(hard_content)} records")

    code_content = df[df["modality_primary"] == "code"]
    print(f"Code content: {len(code_content)} records")

    # Cross-tabulation
    print("\nBand x Modality cross-tab:")
    crosstab = df.groupby(["band_id", "modality_primary"]).size().unstack(fill_value=0)
    print(crosstab.to_string())

    # Export summary
    report_path = output_dir / "analysis_report.json"
    analyzer.export_summary_report(str(report_path))
    print(f"\nExported summary report to: {report_path}")

    print("\n[OK] Metadata analysis example complete.")


if __name__ == "__main__":
    main()
