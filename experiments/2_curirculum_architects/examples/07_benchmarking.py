"""Example: Benchmarking the extraction pipeline.

This example demonstrates:
- Running benchmarks with timing per metric
- Memory tracking
- Interpreting benchmark results
"""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from curriculum_extractor.scripts.benchmark import PipelineBenchmarker


def create_benchmark_data(base_dir: Path, num_records: int = 1000) -> Path:
    """Create sample data for benchmarking."""
    records = []

    for i in range(num_records):
        # Vary content type and length
        if i % 5 == 0:
            text = f"""
def function_{i}(x, y):
    '''Function {i} for processing data.'''
    result = x * y + {i}
    if result > 100:
        return result // 2
    return result

# Usage example
output = function_{i}(5, 10)
print(f"Result: {{output}}")
            """
        elif i % 3 == 0:
            text = f"""
The theory of relativity, developed by Albert Einstein, revolutionized 
our understanding of space, time, and gravity. The special theory, 
published in 1905, introduced the famous equation E=mc². The general 
theory, completed in 1915, describes gravity as curvature of spacetime.
Document number {i} explores these concepts in detail.
            """
        else:
            text = f"""
This is document number {i} in the benchmark dataset. It contains 
typical prose content that might be found in web text or books. The 
content varies in length and complexity to provide realistic test 
conditions. Some documents are shorter while others are longer to 
simulate real-world distribution of text lengths. This particular 
document has moderate length and standard vocabulary.
            """

        records.append(
            {
                "id": f"bench_{i:06d}",
                "text": text,
                "source": "benchmark",
            }
        )

    parquet_path = base_dir / "benchmark_data.parquet"
    table = pa.Table.from_pylist(records)
    pq.write_table(table, parquet_path)

    return parquet_path


def main():
    """Demonstrate benchmarking functionality."""
    parser = argparse.ArgumentParser(description="Benchmark the extraction pipeline.")
    parser.add_argument(
        "--parquet",
        type=str,
        default=None,
        help="Path to input parquet file. If not provided, sample data will be created.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for sample data. Defaults to ./downloads/07_benchmarking/",
    )
    parser.add_argument(
        "--num-records",
        type=int,
        default=500,
        help="Number of records to create in sample data (default: 500).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100).",
    )
    args = parser.parse_args()

    curriculum_path = Path(__file__).parent.parent / "curriculum.yaml"

    print("=" * 80)
    print("PIPELINE BENCHMARK")
    print("=" * 80)

    script_dir = Path(__file__).parent
    default_output_dir = script_dir / "downloads" / "07_benchmarking"
    output_dir = Path(args.output) if args.output else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tmpdir = output_dir

    # Use user-provided parquet or create benchmark data
    if args.parquet:
        data_path = Path(args.parquet)
        if not data_path.exists():
            raise FileNotFoundError(f"Parquet file does not exist: {data_path}")
        print(f"\n[1] Using user-provided parquet: {data_path}")
    else:
        # Create benchmark data
        print("\n[1] Creating benchmark data...")
        data_path = create_benchmark_data(tmpdir, num_records=args.num_records)
        print(f"    Created {data_path}")

    # Initialize benchmarker
    print("\n[2] Initializing benchmarker...")
    benchmarker = PipelineBenchmarker(curriculum_path)

    # Run benchmark
    print("\n[3] Running benchmark...")
    print("-" * 60)

    results = benchmarker.benchmark_local(
        input_path=data_path,
        batch_size=args.batch_size,
        max_records=args.num_records,
    )

    # Results are already printed by the benchmarker

    # Export results
    results_dict = results.to_dict()

    print("\n" + "=" * 60)
    print("EXPORTED RESULTS (JSON format):")
    print("=" * 60)

    import json

    print(json.dumps(results_dict, indent=2))

    # Analysis tips
    print("\n" + "=" * 60)
    print("ANALYSIS TIPS:")
    print("=" * 60)
    print(
        """
1. THROUGHPUT: Aim for 1000+ records/sec for production

2. I/O vs PROCESSING:
   - High I/O time: Consider prefetching or faster storage
   - High processing time: Optimize slow metrics

3. PER-METRIC TIMING:
   - Look for metrics taking >1ms avg - may need optimization
   - Consider disabling expensive metrics if not critical
   - Use levels to fail-fast with cheap filters

4. MEMORY:
   - Peak memory should be < available RAM
   - If memory grows over time, check for leaks

5. BATCH SIZE:
   - Larger batches = better I/O efficiency
   - Smaller batches = more responsive progress
   - Sweet spot usually 1000-10000
"""
    )

    print("\n[OK] Benchmark example complete.")


if __name__ == "__main__":
    main()
