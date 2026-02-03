"""Example of processing parquet files with curriculum tags."""

import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from curriculum_tags import CurriculumTagger


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

    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "input.parquet"
        output_file = Path(tmpdir) / "output.parquet"

        # Create sample data
        print("\n1. Creating sample dataset...")
        create_sample_data(input_file, num_samples=100)

        # Initialize tagger (uses default plugins from curriculum.yaml)
        print("\n2. Initializing tagger...")
        tagger = CurriculumTagger(curriculum_path)

        # Process file
        print("\n3. Processing parquet file...")

        processed_count = [0]

        def progress_callback(total):
            processed_count[0] = total
            if total % 50 == 0:
                print(f"   Processed {total} rows...")

        stats = tagger.process_parquet(
            input_path=input_file,
            output_path=output_file,
            batch_size=25,
            progress_callback=progress_callback,
        )
        print("\n4. Processing complete!")
        print(f"   Total rows: {stats['total_rows']}")
        print(f"   Errors: {stats['error_count']}")
        print(f"   Output file: {stats['output_file']}")

        # Read and display sample results
        print("\n5. Sample tagged results:")
        print("=" * 80)

        result_table = pq.read_table(output_file)
        result_data = result_table.to_pylist()

        # Show first 3 samples
        for i, row in enumerate(result_data[:3]):
            print(f"\nSample {i+1}:")
            print(f"  ID: {row['id']}")
            print(f"  Text: {row['text'][:50]}...")

            tags = row["curriculum_tags"]
            print(f"  Curriculum Version: {tags['version']}")

            if "difficulty" in tags:
                diff = tags["difficulty"]
                print(f"  Difficulty Band: {diff['band']}")

            if "modality" in tags:
                mod = tags["modality"]
                print(f"  Primary Modality: {mod['primary_modality']}")

            print("-" * 80)


if __name__ == "__main__":
    process_parquet_demo()
