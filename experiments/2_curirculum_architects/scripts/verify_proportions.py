"""Verify calculate_proportions script functionality."""

import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def create_mock_metadata(path: Path):
    """Create a mock metadata.parquet file with diverse bands."""

    # Create distribution: mostly lower bands, some higher
    bands = (
        ["B0"] * 40 + ["B1"] * 30 + ["B2"] * 15 + ["B3"] * 10 + ["B4"] * 4 + ["B5"] * 1
    )  # 100 samples

    data = []
    for i, band in enumerate(bands):
        data.append(
            {
                "id": f"id_{i}",
                "curriculum_tags": {
                    "difficulty": {"score": 0.5, "level": "L2"},  # Mock legacy
                    "band_assignment": {"band": band, "reason": "Mocked"},  # New Source
                    "version": "0.3",
                },
            }
        )

    table = pa.Table.from_pylist(data)
    pq.write_table(table, path)
    print(f"Created mock metadata at {path} with 100 samples.")


def run_verification():
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = Path(tmpdir) / "output.metadata.parquet"
        create_mock_metadata(meta_path)

        # Run calculation script
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/calculate_proportions.py",
            str(meta_path),
            "--sampling-rate",
            "1.0",  # Sample everything for deterministic test
            "--curriculum-path",
            "curriculum.yaml",
        ]

        print(f"\nRunning command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        print("\n=== Output ===")
        print(result.stdout)

        if result.stderr:
            print("\n=== Errors ===")
            print(result.stderr)

        if result.returncode == 0:
            print("\nSUCCESS: Script ran without error.")
        else:
            print("\nFAILURE: Script returned error code.")
            sys.exit(1)


if __name__ == "__main__":
    run_verification()
