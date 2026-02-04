"""Band assignment script for post-processing metadata layer.

Band assignment is performed AFTER extraction to enable:
- Batch-level calibration based on full dataset statistics
- Easy experimentation with different band thresholds
- Separation of concerns (extraction vs classification)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BandAssigner:
    """Assign curriculum bands to records based on metadata metrics."""

    def __init__(
        self,
        curriculum_path: str | Path,
        band_config: Optional[Dict] = None,
    ):
        """Initialize band assigner.

        Args:
            curriculum_path: Path to curriculum YAML for band definitions
            band_config: Optional override for band configuration
        """
        self.curriculum_path = Path(curriculum_path)

        with open(self.curriculum_path) as f:
            self.curriculum = yaml.safe_load(f)

        self.band_config = band_config or self._get_default_band_config()
        self.bands = self._load_bands()

    def _load_bands(self) -> List[Dict]:
        """Load band definitions from curriculum."""
        bands_config = self.curriculum.get("difficulty_bands", {})
        return bands_config.get(
            "bands",
            [
                {"id": "B0", "name": "Nursery", "max_score": 0.1},
                {"id": "B1", "name": "Elementary", "max_score": 0.25},
                {"id": "B2", "name": "Middle School", "max_score": 0.45},
                {"id": "B3", "name": "High School", "max_score": 0.65},
                {"id": "B4", "name": "Undergraduate", "max_score": 0.85},
                {"id": "B5", "name": "Graduate", "max_score": 1.0},
            ],
        )

    def _get_default_band_config(self) -> Dict:
        """Get default band assignment configuration."""
        return {
            "score_column": "difficulty_score",  # Primary scoring column
            "secondary_columns": [  # Additional columns to consider
                "readability_score",
                "entropy_score",
            ],
            "weights": {
                "difficulty_score": 0.6,
                "readability_score": 0.3,
                "entropy_score": 0.1,
            },
        }

    def assign_band(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Assign band to a single record.

        Args:
            record: Metadata record with metric columns

        Returns:
            Dictionary with band assignment fields
        """
        # Get primary score
        score_col = self.band_config.get("score_column", "difficulty_score")
        primary_score = record.get(score_col, 0.5)

        # Calculate weighted score if secondary columns available
        weights = self.band_config.get("weights", {})
        if weights:
            total_weight = 0.0
            weighted_sum = 0.0

            for col, weight in weights.items():
                if col in record and record[col] is not None:
                    weighted_sum += record[col] * weight
                    total_weight += weight

            if total_weight > 0:
                final_score = weighted_sum / total_weight
            else:
                final_score = primary_score
        else:
            final_score = primary_score

        # Assign band based on score
        assigned_band = self.bands[-1]["id"]  # Default to highest band
        assigned_name = self.bands[-1]["name"]

        for band in self.bands:
            if final_score <= band.get("max_score", 1.0):
                assigned_band = band["id"]
                assigned_name = band["name"]
                break

        return {
            "band_id": assigned_band,
            "band_name": assigned_name,
            "band_score": round(final_score, 4),
        }

    def process_metadata_file(
        self,
        input_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Process a metadata parquet file and add band assignments.

        Args:
            input_path: Path to metadata parquet file
            output_path: Optional output path (defaults to updating in place)

        Returns:
            Processing statistics
        """
        input_path = Path(input_path)
        output_path = Path(output_path) if output_path else input_path

        logger.info(f"Processing: {input_path}")

        # Read metadata
        table = pq.read_table(input_path)
        records = table.to_pylist()

        # Assign bands
        updated_records = []
        band_counts: Dict[str, int] = {}

        for record in records:
            band_info = self.assign_band(record)
            record.update(band_info)
            updated_records.append(record)

            band_id = band_info["band_id"]
            band_counts[band_id] = band_counts.get(band_id, 0) + 1

        # Write output
        output_table = pa.Table.from_pylist(updated_records)

        if output_path != input_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        pq.write_table(output_table, output_path)

        logger.info(f"Wrote {len(records)} records to {output_path}")
        logger.info(f"Band distribution: {band_counts}")

        return {
            "total_records": len(records),
            "band_distribution": band_counts,
            "output_path": str(output_path),
        }

    def process_metadata_directory(
        self,
        input_dir: str | Path,
        output_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Process all metadata files in a directory.

        Handles Hive-style partitioning (file_name=xxx/).

        Args:
            input_dir: Input directory with metadata files
            output_dir: Optional output directory

        Returns:
            Aggregate statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) if output_dir else input_dir

        total_records = 0
        total_files = 0
        aggregate_bands: Dict[str, int] = {}

        # Find all parquet files (including in partitions)
        for parquet_file in input_dir.rglob("*.parquet"):
            relative = parquet_file.relative_to(input_dir)
            output_file = output_dir / relative

            stats = self.process_metadata_file(parquet_file, output_file)

            total_records += stats["total_records"]
            total_files += 1

            for band_id, count in stats["band_distribution"].items():
                aggregate_bands[band_id] = aggregate_bands.get(band_id, 0) + count

        logger.info(f"Processed {total_files} files, {total_records} total records")

        return {
            "total_files": total_files,
            "total_records": total_records,
            "band_distribution": aggregate_bands,
        }

    def process_s3(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        filesystem: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Process metadata from S3.

        Args:
            input_path: S3 path to metadata directory
            output_path: Optional S3 output path (bucket or prefix)
            filesystem: s3fs filesystem instance

        Returns:
            Processing statistics
        """
        if filesystem is None:
            import s3fs

            filesystem = s3fs.S3FileSystem()

        output_path = output_path or input_path

        total_records = 0
        total_files = 0
        aggregate_bands: Dict[str, int] = {}

        # Normalize paths
        input_path = input_path.rstrip("/")
        output_path = output_path.rstrip("/")

        # Extract bucket + prefix
        # s3://bucket/prefix -> bucket, prefix
        def split_s3_path(path: str):
            assert path.startswith("s3://")
            bucket, _, key = path[5:].partition("/")
            return bucket, key

        in_bucket, in_prefix = split_s3_path(input_path)
        out_bucket, out_prefix = split_s3_path(output_path)

        # List all parquet files
        files = filesystem.glob(f"{input_path}/**/*.parquet")

        for file_path in files:
            logger.info(f"Processing S3: {file_path}")

            # Read from S3
            table = pq.read_table(file_path, filesystem=filesystem)
            records = table.to_pylist()

            # Assign bands
            updated_records = []
            for record in records:
                band_info = self.assign_band(record)
                record.update(band_info)
                updated_records.append(record)

                band_id = band_info["band_id"]
                aggregate_bands[band_id] = aggregate_bands.get(band_id, 0) + 1

            # Write back
            output_table = pa.Table.from_pylist(updated_records)

            # Compute relative key *within the input prefix*
            # s3://bucket/input_prefix/xxx -> xxx
            full_key = file_path[5 + len(in_bucket) + 1 :]  # strip s3://bucket/
            rel_key = full_key[len(in_prefix) :].lstrip("/")
            out_file = f"s3://{out_bucket}/{out_prefix}/{rel_key}"

            tmp_file = out_file + ".tmp"
            with filesystem.open(tmp_file, "wb") as f:
                pq.write_table(output_table, f)
            filesystem.mv(tmp_file, out_file)

            total_records += len(records)
            total_files += 1

        logger.info(f"Processed {total_files} S3 files, {total_records} total records")

        return {
            "total_files": total_files,
            "total_records": total_records,
            "band_distribution": aggregate_bands,
        }


def main():
    """CLI for band assignment."""
    parser = argparse.ArgumentParser(
        description="Assign curriculum bands to metadata layer"
    )
    parser.add_argument(
        "--curriculum",
        type=str,
        required=True,
        help="Path to curriculum YAML file",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input metadata path (local or S3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path (defaults to in-place update)",
    )
    parser.add_argument(
        "--score-column",
        type=str,
        default="difficulty_score",
        help="Column to use for band scoring",
    )

    args = parser.parse_args()

    # Create band config
    band_config = {
        "score_column": args.score_column,
    }

    assigner = BandAssigner(args.curriculum, band_config)

    # Determine if S3 or local
    if args.input.startswith("s3://"):
        stats = assigner.process_s3(args.input, args.output)
    else:
        input_path = Path(args.input)
        if input_path.is_file():
            stats = assigner.process_metadata_file(input_path, args.output)
        else:
            stats = assigner.process_metadata_directory(input_path, args.output)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
