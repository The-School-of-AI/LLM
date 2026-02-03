import os
import time
from pathlib import Path

import pyarrow.parquet as pq
import ray
from curriculum_tags import CurriculumTagger
from tqdm import tqdm

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
INPUT_FILE = "v1_5r2_sample-0001.json.parquet"
CURRICULUM_YAML = str(Path(__file__).parent.parent / "curriculum.yaml")
METRICS_CONFIG = str(Path(__file__).parent.parent / "metrics_config.yaml")

# To simulate a distributed run, we'll process the same file multiple times in parallel
# This measures how your laptop handles high CPU saturation, similar to an S3 cluster node.
NUM_PARALLEL_TASKS = 4  # Adjust based on your laptop's core count
BATCH_SIZE = 5000


@ray.remote(num_cpus=1)
def process_file_task(task_id: int):
    """Simulates one S3 worker processing a file."""
    output_file = f"benchmark_temp_{task_id}.parquet"

    # Initialize tagger inside task (S3 implementation pattern)
    tagger = CurriculumTagger(
        curriculum_path=CURRICULUM_YAML, metrics_config_path=METRICS_CONFIG
    )

    print(f"  [Task {task_id}] STARTING: {INPUT_FILE}")

    # Get total row count for percentage calculation
    total_expected_rows = pq.ParquetFile(INPUT_FILE).metadata.num_rows

    start = time.time()

    def heartbeat_callback(rows):
        """Prints status every 10 seconds for long runs."""
        now = time.time()
        if not hasattr(heartbeat_callback, "last_log"):
            heartbeat_callback.last_log = now

        if now - heartbeat_callback.last_log > 10:
            elapsed = now - start
            pct = (rows / total_expected_rows) * 100 if total_expected_rows > 0 else 0
            print(
                f"\n  [Task {task_id}] STILL RUNNING: {rows:,}/{total_expected_rows:,} ({pct:.1f}%) | {elapsed:.0f}s elapsed"
            )
            heartbeat_callback.last_log = now

    stats = tagger.process_parquet(
        input_path=INPUT_FILE,
        output_path=output_file,
        batch_size=BATCH_SIZE,
        progress_callback=heartbeat_callback,
    )
    duration = time.time() - start

    # Clean up immediately
    if os.path.exists(output_file):
        os.remove(output_file)
    meta = output_file.replace(".parquet", ".metadata.parquet")
    if os.path.exists(meta):
        os.remove(meta)

    return {
        "rows": stats["total_rows"],
        "duration": duration,
        "bytes": os.path.getsize(INPUT_FILE),
    }


def benchmark():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print("--- 1TB Scaling Benchmark (Distributed Mode) ---")
    print(f"Parallel Tasks: {NUM_PARALLEL_TASKS}")
    print(f"File: {INPUT_FILE} ({os.path.getsize(INPUT_FILE)/1e6:.1f} MB)")

    if not ray.is_initialized():
        ray.init(num_cpus=NUM_PARALLEL_TASKS)

    print("\nLaunching parallel processing tasks...")
    start_time = time.time()

    # Launch tasks
    futures = [process_file_task.remote(i) for i in range(NUM_PARALLEL_TASKS)]

    results = []
    pbar = tqdm(total=NUM_PARALLEL_TASKS, desc="Benchmarking")

    current_rows = 0
    current_bytes = 0

    # Process futures as they complete
    remaining = futures
    while remaining:
        done, remaining = ray.wait(remaining, num_returns=1)
        res = ray.get(done[0])
        results.append(res)

        current_rows += res["rows"]
        current_bytes += res["bytes"]
        elapsed = time.time() - start_time
        agg_mb_s = (current_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0

        pbar.update(1)
        pbar.set_postfix(
            {
                "t_rows": f"{current_rows:,}",
                "agg_mb/s": f"{agg_mb_s:.1f}",
                "last": f"{res['rows']/res['duration']:.0f} r/s",
            }
        )

    pbar.close()

    total_duration = time.time() - start_time
    total_rows = sum(r["rows"] for r in results)
    total_bytes = sum(r["bytes"] for r in results)

    # Throughput calculations
    total_mb = total_bytes / (1024 * 1024)
    mb_per_sec = total_mb / total_duration
    rows_per_sec = total_rows / total_duration

    # 1 TB = 1,024 * 1,024 MB
    tb_size_mb = 1024 * 1024
    hours_per_tb_laptop = (tb_size_mb / mb_per_sec) / 3600

    print("\n--- Laptop Results (Aggregated) ---")
    print(f"Total Time: {total_duration:.2f} seconds")
    print(f"Aggregate Throughput: {rows_per_sec:.2f} rows/sec")
    print(f"Aggregate Throughput: {mb_per_sec:.2f} MB/sec")

    print("\n--- Extrapolation for 1TB ---")
    print(f"Time for 1TB on THIS Laptop: {hours_per_tb_laptop:.2f} hours")

    # Cloud cluster estimate
    cluster_nodes = 10
    cluster_cpus_per_node = 32
    total_cluster_parallelism = cluster_nodes * cluster_cpus_per_node

    # Calculate single-core throughput from our run
    avg_single_core_mb_per_sec = sum(
        r["bytes"] / (1024 * 1024) / r["duration"] for r in results
    ) / len(results)
    cluster_mb_per_sec = avg_single_core_mb_per_sec * total_cluster_parallelism
    hours_per_tb_cluster = (tb_size_mb / cluster_mb_per_sec) / 3600

    print("\n--- Cloud Estimate (10 Nodes, 320 CPUs) ---")
    print(f"Estimated Throughput: {cluster_mb_per_sec:.2f} MB/sec")
    print(f"Estimated Time: {hours_per_tb_cluster:.2f} hours")
    print(f"Estimated Time: {hours_per_tb_cluster * 60:.2f} minutes")

    ray.shutdown()


if __name__ == "__main__":
    benchmark()
