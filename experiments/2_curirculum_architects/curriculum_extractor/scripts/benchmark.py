"""Benchmark script for curriculum extraction pipeline.

Measures time, speed, and memory usage at each metric level.
Supports both local and S3 data sources.
"""

import argparse
import gc
import json
import logging
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class MetricBenchmark:
    """Benchmark results for a single metric."""

    name: str
    level: int
    total_calls: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    avg_time_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "level": self.level,
            "total_calls": self.total_calls,
            "total_time_ms": round(self.total_time_ms, 3),
            "min_time_ms": (
                round(self.min_time_ms, 3) if self.min_time_ms != float("inf") else 0
            ),
            "max_time_ms": round(self.max_time_ms, 3),
            "avg_time_ms": round(self.avg_time_ms, 3),
        }


@dataclass
class PipelineBenchmark:
    """Full pipeline benchmark results."""

    total_records: int = 0
    processed_records: int = 0
    rejected_records: int = 0
    total_time_seconds: float = 0.0
    records_per_second: float = 0.0
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0
    io_time_seconds: float = 0.0
    processing_time_seconds: float = 0.0
    write_time_seconds: float = 0.0
    metrics: Dict[str, MetricBenchmark] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "summary": {
                "total_records": self.total_records,
                "processed_records": self.processed_records,
                "rejected_records": self.rejected_records,
                "total_time_seconds": round(self.total_time_seconds, 3),
                "records_per_second": round(self.records_per_second, 2),
                "peak_memory_mb": round(self.peak_memory_mb, 2),
                "current_memory_mb": round(self.current_memory_mb, 2),
            },
            "breakdown": {
                "io_time_seconds": round(self.io_time_seconds, 3),
                "processing_time_seconds": round(self.processing_time_seconds, 3),
                "write_time_seconds": round(self.write_time_seconds, 3),
            },
            "metrics": {name: m.to_dict() for name, m in self.metrics.items()},
        }


class PipelineBenchmarker:
    """Benchmark the curriculum extraction pipeline."""

    def __init__(
        self,
        curriculum_path: str | Path,
        metrics_config_path: Optional[str | Path] = None,
    ):
        """Initialize benchmarker.

        Args:
            curriculum_path: Path to curriculum YAML
            metrics_config_path: Optional path to metrics config
        """
        self.curriculum_path = Path(curriculum_path)
        self.metrics_config_path = metrics_config_path
        self.benchmark = PipelineBenchmark()

    def _get_memory_usage(self) -> tuple[float, float]:
        """Get current and peak memory usage in MB."""
        current, peak = tracemalloc.get_traced_memory()
        return current / (1024 * 1024), peak / (1024 * 1024)

    def benchmark_local(
        self,
        input_path: str | Path,
        batch_size: int = 10000,
        max_records: Optional[int] = None,
    ) -> PipelineBenchmark:
        """Benchmark extraction on local parquet files.

        Args:
            input_path: Path to parquet file or directory
            batch_size: Records per batch
            max_records: Maximum records to process (for quick tests)

        Returns:
            Benchmark results
        """
        from curriculum_extractor import CurriculumExtractor

        input_path = Path(input_path)

        # Start memory tracking
        tracemalloc.start()
        gc.collect()

        logger.info("=" * 60)
        logger.info("BENCHMARK: Local Pipeline")
        logger.info("=" * 60)
        logger.info(f"Input: {input_path}")
        logger.info(f"Batch size: {batch_size}")

        # Initialize extractor with timing
        extractor = CurriculumExtractor(
            self.curriculum_path,
            metrics_config_path=self.metrics_config_path,
            track_timing=True,
        )

        # Initialize metric benchmarks
        for plugin in extractor.plugins:
            self.benchmark.metrics[plugin.name] = MetricBenchmark(
                name=plugin.name,
                level=plugin.level,
            )

        logger.info(f"Loaded {len(extractor.plugins)} metrics")

        # Find files
        if input_path.is_file():
            files = [input_path]
        else:
            files = list(input_path.rglob("*.parquet"))

        logger.info(f"Found {len(files)} parquet files")

        # Process
        overall_start = time.perf_counter()
        io_time = 0.0
        processing_time = 0.0
        total_records = 0

        for file_path in files:
            logger.info(f"Processing: {file_path}")

            # Read (IO time)
            io_start = time.perf_counter()
            parquet_file = pq.ParquetFile(file_path)
            io_time += time.perf_counter() - io_start

            for batch in parquet_file.iter_batches(batch_size=batch_size):
                io_start = time.perf_counter()
                records = batch.to_pylist()
                io_time += time.perf_counter() - io_start

                proc_start = time.perf_counter()

                for record in records:
                    metadata, rejection = extractor.extract_record(
                        record, str(file_path)
                    )

                    if rejection:
                        self.benchmark.rejected_records += 1
                    else:
                        self.benchmark.processed_records += 1

                    total_records += 1

                    if max_records and total_records >= max_records:
                        break

                processing_time += time.perf_counter() - proc_start

                # Progress log
                if total_records % 10000 == 0:
                    current_mem, peak_mem = self._get_memory_usage()
                    logger.info(
                        f"  Processed {total_records} records | "
                        f"Memory: {current_mem:.1f} MB (peak: {peak_mem:.1f} MB)"
                    )

                if max_records and total_records >= max_records:
                    break

            if max_records and total_records >= max_records:
                break

        # Final timing
        overall_time = time.perf_counter() - overall_start
        current_mem, peak_mem = self._get_memory_usage()

        # Update benchmark
        self.benchmark.total_records = total_records
        self.benchmark.total_time_seconds = overall_time
        self.benchmark.records_per_second = (
            total_records / overall_time if overall_time > 0 else 0
        )
        self.benchmark.io_time_seconds = io_time
        self.benchmark.processing_time_seconds = processing_time
        self.benchmark.peak_memory_mb = peak_mem
        self.benchmark.current_memory_mb = current_mem

        # Get per-metric timing
        timing_stats = extractor.get_timing_stats()
        if timing_stats:
            for name, stats in timing_stats.items():
                if name in self.benchmark.metrics:
                    m = self.benchmark.metrics[name]
                    m.total_calls = stats["count"]
                    m.total_time_ms = stats["total_seconds"] * 1000
                    m.min_time_ms = stats["min_ms"]
                    m.max_time_ms = stats["max_ms"]
                    m.avg_time_ms = stats["mean_ms"]

        tracemalloc.stop()

        self._print_results()
        return self.benchmark

    def benchmark_s3(
        self,
        input_path: str,
        batch_size: int = 10000,
        max_records: Optional[int] = None,
    ) -> PipelineBenchmark:
        """Benchmark extraction on S3 parquet files.

        Args:
            input_path: S3 path to parquet files
            batch_size: Records per batch
            max_records: Maximum records to process

        Returns:
            Benchmark results
        """
        import s3fs
        from curriculum_extractor import CurriculumExtractor

        # Start memory tracking
        tracemalloc.start()
        gc.collect()

        logger.info("=" * 60)
        logger.info("BENCHMARK: S3 Pipeline")
        logger.info("=" * 60)
        logger.info(f"Input: {input_path}")
        logger.info(f"Batch size: {batch_size}")

        # Initialize S3 filesystem
        fs = s3fs.S3FileSystem()

        # Initialize extractor with timing
        extractor = CurriculumExtractor(
            self.curriculum_path,
            metrics_config_path=self.metrics_config_path,
            filesystem=fs,
            track_timing=True,
        )

        # Initialize metric benchmarks
        for plugin in extractor.plugins:
            self.benchmark.metrics[plugin.name] = MetricBenchmark(
                name=plugin.name,
                level=plugin.level,
            )

        logger.info(f"Loaded {len(extractor.plugins)} metrics")

        # Find files
        s3_path = input_path.replace("s3://", "")
        if s3_path.endswith(".parquet"):
            files = [s3_path]
        else:
            files = fs.glob(f"{s3_path}/**/*.parquet")

        logger.info(f"Found {len(files)} S3 parquet files")

        # Process
        overall_start = time.perf_counter()
        io_time = 0.0
        processing_time = 0.0
        total_records = 0

        for file_path in files:
            logger.info(f"Processing: s3://{file_path}")

            # Read (IO time)
            io_start = time.perf_counter()
            parquet_file = pq.ParquetFile(file_path, filesystem=fs)
            io_time += time.perf_counter() - io_start

            for batch in parquet_file.iter_batches(batch_size=batch_size):
                io_start = time.perf_counter()
                records = batch.to_pylist()
                io_time += time.perf_counter() - io_start

                proc_start = time.perf_counter()

                for record in records:
                    metadata, rejection = extractor.extract_record(
                        record, f"s3://{file_path}"
                    )

                    if rejection:
                        self.benchmark.rejected_records += 1
                    else:
                        self.benchmark.processed_records += 1

                    total_records += 1

                    if max_records and total_records >= max_records:
                        break

                processing_time += time.perf_counter() - proc_start

                # Progress log
                if total_records % 10000 == 0:
                    current_mem, peak_mem = self._get_memory_usage()
                    logger.info(
                        f"  Processed {total_records} records | "
                        f"Memory: {current_mem:.1f} MB (peak: {peak_mem:.1f} MB)"
                    )

                if max_records and total_records >= max_records:
                    break

            if max_records and total_records >= max_records:
                break

        # Final timing
        overall_time = time.perf_counter() - overall_start
        current_mem, peak_mem = self._get_memory_usage()

        # Update benchmark
        self.benchmark.total_records = total_records
        self.benchmark.total_time_seconds = overall_time
        self.benchmark.records_per_second = (
            total_records / overall_time if overall_time > 0 else 0
        )
        self.benchmark.io_time_seconds = io_time
        self.benchmark.processing_time_seconds = processing_time
        self.benchmark.peak_memory_mb = peak_mem
        self.benchmark.current_memory_mb = current_mem

        # Get per-metric timing
        timing_stats = extractor.get_timing_stats()
        if timing_stats:
            for name, stats in timing_stats.items():
                if name in self.benchmark.metrics:
                    m = self.benchmark.metrics[name]
                    m.total_calls = stats["count"]
                    m.total_time_ms = stats["total_seconds"] * 1000
                    m.min_time_ms = stats["min_ms"]
                    m.max_time_ms = stats["max_ms"]
                    m.avg_time_ms = stats["mean_ms"]

        tracemalloc.stop()

        self._print_results()
        return self.benchmark

    def _print_results(self):
        """Print benchmark results to console."""
        b = self.benchmark

        logger.info("")
        logger.info("=" * 60)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 60)
        logger.info("")
        logger.info("SUMMARY")
        logger.info("-" * 40)
        logger.info(f"  Total records:      {b.total_records:,}")
        logger.info(f"  Processed:          {b.processed_records:,}")
        logger.info(f"  Rejected:           {b.rejected_records:,}")
        logger.info(f"  Total time:         {b.total_time_seconds:.2f} seconds")
        logger.info(f"  Throughput:         {b.records_per_second:.1f} records/sec")
        logger.info(f"  Peak memory:        {b.peak_memory_mb:.1f} MB")
        logger.info("")
        logger.info("TIME BREAKDOWN")
        logger.info("-" * 40)
        logger.info(
            f"  I/O (read):         {b.io_time_seconds:.2f}s ({100*b.io_time_seconds/b.total_time_seconds:.1f}%)"
        )
        logger.info(
            f"  Processing:         {b.processing_time_seconds:.2f}s ({100*b.processing_time_seconds/b.total_time_seconds:.1f}%)"
        )
        logger.info("")
        logger.info("PER-METRIC TIMING")
        logger.info("-" * 40)

        # Sort by level, then by total time
        sorted_metrics = sorted(
            b.metrics.values(),
            key=lambda m: (m.level, -m.total_time_ms),
        )

        for m in sorted_metrics:
            pct = (
                (m.total_time_ms / 1000 / b.processing_time_seconds * 100)
                if b.processing_time_seconds > 0
                else 0
            )
            logger.info(
                f"  [{m.level}] {m.name:20s} | "
                f"avg: {m.avg_time_ms:6.2f}ms | "
                f"total: {m.total_time_ms/1000:6.2f}s ({pct:4.1f}%)"
            )

        logger.info("")
        logger.info("=" * 60)


def main():
    """CLI for benchmarking."""
    parser = argparse.ArgumentParser(
        description="Benchmark curriculum extraction pipeline"
    )
    parser.add_argument(
        "--curriculum",
        type=str,
        required=True,
        help="Path to curriculum YAML",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input path (local or S3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        help="Max records to process (for quick testing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--metrics-config",
        type=str,
        help="Path to metrics config YAML",
    )

    args = parser.parse_args()

    benchmarker = PipelineBenchmarker(
        args.curriculum,
        metrics_config_path=args.metrics_config,
    )

    # Run benchmark
    if args.input.startswith("s3://"):
        results = benchmarker.benchmark_s3(
            args.input,
            batch_size=args.batch_size,
            max_records=args.max_records,
        )
    else:
        results = benchmarker.benchmark_local(
            args.input,
            batch_size=args.batch_size,
            max_records=args.max_records,
        )

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
