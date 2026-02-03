"""Example of processing parquet files with curriculum tags.

By default only CSV output is written (main + rejected). Set write_parquet=True
to also write pass-through Parquet (original rows, no curriculum_tags).
"""

import csv
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from curriculum_tags import CurriculumTagger

PROCESSED_DATA_DIR = Path(__file__).parent.parent / "processed_data"
CSV_OUTPUT_PATH = PROCESSED_DATA_DIR / "output.csv"
REJECTED_CSV_OUTPUT_PATH = PROCESSED_DATA_DIR / "rejected.csv"


def create_sample_data(output_path: Path, num_samples: int = 100):
    """Create sample parquet file for demonstration."""

    samples = [
        "Hello world! This is a simple text.",
        "Python is a programming language. def hello(): print('world')",
        "Advanced mathematical concepts: ∫ f(x) dx = ∑ xᵢ",
        "Let's think step by step. First, we analyze the problem.",
    ]

    data = []
    for i in range(num_samples):
        data.append(
            {
                "id": f"sample_{i}",
                "text": samples[i % len(samples)],
                "source": "demo",
                "added": "2026-02-01T00:00:00Z",
                "metadata": {
                    "language": "en",
                    "word_count": len(samples[i % len(samples)].split()),
                },
            }
        )

    table = pa.Table.from_pylist(data)
    pq.write_table(table, output_path)
    print(f"Created sample data: {output_path} ({num_samples} rows)")


def process_parquet_demo():
    """Demonstrate parquet processing."""

    # Path relative to this example file
    curriculum_path = Path(__file__).parent.parent / "curriculum.yaml"

    # Input in temp dir; CSV output to processed_data folder
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "input.parquet"

        # Create sample data
        print("\n1. Creating sample dataset...")
        create_sample_data(input_file, num_samples=100)

        # Initialize tagger (uses default plugins from curriculum.yaml)
        print("\n2. Initializing tagger...")
        tagger = CurriculumTagger(curriculum_path)

        # Process file (CSV only by default; output to processed_data)
        print("\n3. Processing parquet file (CSV output to processed_data)...")

        processed_count = [0]

        def progress_callback(total):
            processed_count[0] = total
            if total % 50 == 0:
                print(f"   Processed {total} rows...")

        stats = tagger.process_parquet(
            input_path=input_file,
            batch_size=25,
            progress_callback=progress_callback,
            output_csv_path=CSV_OUTPUT_PATH,
            rejected_csv_path=REJECTED_CSV_OUTPUT_PATH,
            write_parquet=False,
        )
        print("\n4. Processing complete!")
        print(f"   Total rows: {stats['total_rows']}")
        print(f"   Errors: {stats['error_count']}")
        if "main_csv_path" in stats:
            print(f"   Main CSV: {stats['main_csv_path']} ({stats.get('main_csv_rows', 0)} rows)")
            print(f"   Rejected CSV: {stats['rejected_csv_path']} ({stats.get('rejected_csv_rows', 0)} rows)")
        if "output_file" in stats:
            print(f"   Parquet: {stats['output_file']}")

        # Show first 3 rows from main CSV (in processed_data)
        print("\n5. Sample tagged results (from CSV in processed_data):")
        print("=" * 80)

        main_csv = stats.get("main_csv_path")
        if main_csv and Path(main_csv).exists():
            with open(main_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            for i, row in enumerate(rows[:3]):
                print(f"\nSample {i+1}:")
                print(f"  id: {row.get('id', '—')}")
                print(f"  file_path: {row.get('file_path', '—')}")
                print(f"  band: {row.get('band', '—')}")
                print(
                    f"  difficulty_level: {row.get('difficulty_level', '—')} (score: {row.get('difficulty_score', '—')})"
                )
                print(f"  primary_modality: {row.get('primary_modality', '—')}")
                print("-" * 80)
        else:
            print("  (No main CSV path in stats or file missing)")


if __name__ == "__main__":
    process_parquet_demo()
