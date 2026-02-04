#!/usr/bin/env python3
"""Analysis script for curriculum metadata layer.

This script provides a CLI for analyzing the extracted metadata.

Usage:
    # Basic summary
    python -m curriculum_reader.scripts.analyze \
        --metadata ./output/metadata \
        --summary

    # Full report
    python -m curriculum_reader.scripts.analyze \
        --metadata ./output/metadata \
        --rejection ./output/rejections \
        --report

    # Export to CSV
    python -m curriculum_reader.scripts.analyze \
        --metadata ./output/metadata \
        --export stats.csv

    # Band comparison
    python -m curriculum_reader.scripts.analyze \
        --metadata ./output/metadata \
        --compare-bands difficulty_score
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze curriculum metadata layer",
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
        "--rejection",
        "-r",
        help="Path to rejection layer (optional)",
    )

    parser.add_argument(
        "--summary",
        "-s",
        action="store_true",
        help="Show summary statistics",
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate full text report",
    )

    parser.add_argument(
        "--export",
        help="Export statistics to file (CSV/JSON based on extension)",
    )

    parser.add_argument(
        "--compare-bands",
        help="Compare a metric across bands",
    )

    parser.add_argument(
        "--file-stats",
        action="store_true",
        help="Show per-file statistics",
    )

    parser.add_argument(
        "--s3",
        action="store_true",
        help="Enable S3 mode",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    from curriculum_reader import MetadataAnalyzer, MetadataReader
    from curriculum_reader.core.reader import RejectionReader

    # Initialize readers
    fs = None
    if args.s3:
        import s3fs

        fs = s3fs.S3FileSystem()

    metadata_reader = MetadataReader(args.metadata, filesystem=fs)

    rejection_reader = None
    if args.rejection:
        rejection_reader = RejectionReader(args.rejection, filesystem=fs)

    analyzer = MetadataAnalyzer(metadata_reader, rejection_reader)

    # Execute requested analysis
    if args.report:
        report = analyzer.export_summary_report()
        print(report)
        return 0

    if args.summary:
        summary = analyzer.get_summary()

        if args.json:
            output = {
                "total_records": summary.total_records,
                "total_partitions": summary.total_partitions,
                "columns": summary.columns,
            }
            if summary.band_distribution:
                output["band_distribution"] = {
                    "counts": summary.band_distribution.counts,
                    "percentages": summary.band_distribution.percentages,
                }
            if summary.modality_distribution:
                output["modality_distribution"] = summary.modality_distribution
            if summary.difficulty_stats:
                output["difficulty_stats"] = {
                    "mean": summary.difficulty_stats.mean,
                    "std": summary.difficulty_stats.std,
                    "min": summary.difficulty_stats.min,
                    "max": summary.difficulty_stats.max,
                }
            print(json.dumps(output, indent=2))
        else:
            print(f"\n{'='*60}")
            print("METADATA SUMMARY")
            print(f"{'='*60}")
            print(f"Total Records: {summary.total_records:,}")
            print(f"Total Partitions: {summary.total_partitions}")
            print(f"Columns: {len(summary.columns)}")

            if summary.band_distribution:
                print("\nBand Distribution:")
                for band in sorted(summary.band_distribution.counts.keys()):
                    count = summary.band_distribution.counts[band]
                    pct = summary.band_distribution.percentages[band]
                    print(f"  {band}: {count:,} ({pct}%)")

            if summary.difficulty_stats:
                print("\nDifficulty Score:")
                print(f"  Mean: {summary.difficulty_stats.mean}")
                print(f"  Std:  {summary.difficulty_stats.std}")

        return 0

    if args.compare_bands:
        results = analyzer.compare_bands(args.compare_bands)

        if args.json:
            output = {}
            for band, stats in results.items():
                output[band] = {
                    "mean": stats.mean,
                    "std": stats.std,
                    "min": stats.min,
                    "max": stats.max,
                    "p50": stats.p50,
                    "count": stats.count,
                }
            print(json.dumps(output, indent=2))
        else:
            print(f"\n{'='*60}")
            print(f"BAND COMPARISON: {args.compare_bands}")
            print(f"{'='*60}")
            print(
                f"{'Band':<8} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Count':>10}"
            )
            print("-" * 60)
            for band in sorted(results.keys()):
                stats = results[band]
                print(
                    f"{band:<8} {stats.mean:>10.4f} {stats.std:>10.4f} {stats.min:>10.4f} {stats.max:>10.4f} {stats.count:>10,}"
                )

        return 0

    if args.file_stats:
        stats = analyzer.get_file_stats()

        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print(f"\n{'='*60}")
            print("PER-FILE STATISTICS")
            print(f"{'='*60}")
            print(f"{'File':<40} {'Records':>10}")
            print("-" * 60)
            for stat in stats:
                name = (
                    stat["file_name"][:37] + "..."
                    if len(stat["file_name"]) > 40
                    else stat["file_name"]
                )
                print(f"{name:<40} {stat['record_count']:>10,}")

        return 0

    if args.export:
        export_path = Path(args.export)
        summary = analyzer.get_summary()

        if export_path.suffix == ".json":
            output = {
                "total_records": summary.total_records,
                "total_partitions": summary.total_partitions,
                "band_distribution": (
                    summary.band_distribution.counts
                    if summary.band_distribution
                    else {}
                ),
                "modality_distribution": summary.modality_distribution or {},
            }
            with open(export_path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"Exported to {export_path}")

        elif export_path.suffix == ".csv":
            import csv

            # Export band distribution
            if summary.band_distribution:
                with open(export_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["band", "count", "percentage"])
                    for band in sorted(summary.band_distribution.counts.keys()):
                        writer.writerow(
                            [
                                band,
                                summary.band_distribution.counts[band],
                                summary.band_distribution.percentages[band],
                            ]
                        )
                print(f"Exported band distribution to {export_path}")

        return 0

    # Default: show help
    print("No analysis option specified. Use --summary, --report, or --compare-bands")
    return 1


if __name__ == "__main__":
    sys.exit(main())
