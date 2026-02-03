import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import ray
import s3fs
from tqdm import tqdm

from curriculum_tags import CurriculumTagger

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
INPUT_S3_PREFIX = "s3://smita-erav4/parquet_raw"
OUTPUT_S3_PREFIX = "s3://smita-erav4/parquet_processed"
# Local directory for CSV output (default: CSV only, no Parquet to S3)
OUTPUT_CSV_DIR = Path("./csv_output")
WRITE_PARQUET = False  # Set True to write pass-through Parquet to S3

# Absolute path to the curriculum config
CURRICULUM_YAML = str(Path(__file__).parent.parent / "curriculum.yaml")
METRICS_CONFIG = str(Path(__file__).parent.parent / "metrics_config.yaml")

# Scaling Parameters
BATCH_SIZE = 5000  # Smaller batch size for memory stability on diverse nodes
NUM_CPUS = os.cpu_count() or 4
MAX_INFLIGHT = NUM_CPUS * 2  # Pipeline depth
MAX_FILES = None  # Set to None for unlimited, or a number to limit the run

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"tagging_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# INITIALIZE
# ------------------------------------------------------------------------------
if not ray.is_initialized():
    ray.init(num_cpus=NUM_CPUS, ignore_reinit_error=True)


# ------------------------------------------------------------------------------
# PROCESS ONE FILE (Ray Task)
# ------------------------------------------------------------------------------
@ray.remote(num_cpus=1)
def process_s3_file(file_path: str) -> dict:
    """Processes a single parquet file directly from S3."""
    fs = s3fs.S3FileSystem()

    # Use print instead of logger for Ray worker visibility
    print(f"  [Worker {os.getpid()}] STARTING: {file_path}")

    try:
        tagger = CurriculumTagger(curriculum_path=CURRICULUM_YAML, metrics_config_path=METRICS_CONFIG)
    except Exception as e:
        return {"file": file_path, "status": "init_failed", "error": str(e)}

    # Map input path to output paths
    input_prefix = INPUT_S3_PREFIX.replace("s3://", "")
    relative_path = file_path.replace(input_prefix, "").lstrip("/")
    output_file = f"{OUTPUT_S3_PREFIX.rstrip('/')}/{relative_path}" if WRITE_PARQUET else None

    # Local CSV path (CSV output only by default)
    local_csv_path = OUTPUT_CSV_DIR / relative_path.replace(".parquet", ".csv")
    local_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Get total expected rows for percentage reporting
    try:
        total_expected_rows = pq.ParquetFile(f"s3://{file_path}", filesystem=fs).metadata.num_rows
    except Exception:
        total_expected_rows = 0

    start_time = time.time()

    def heartbeat_callback(rows):
        """Logs status every 10 seconds for long-running files."""
        now = time.time()
        if not hasattr(heartbeat_callback, "last_log"):
            heartbeat_callback.last_log = now
        if now - heartbeat_callback.last_log > 10:
            elapsed = now - start_time
            pct = (rows / total_expected_rows) * 100 if total_expected_rows > 0 else 0
            print(
                f"  [Worker {os.getpid()}] HEARTBEAT: {file_path} | {rows:,}/{total_expected_rows:,} ({pct:.1f}%) | {elapsed:.0f}s elapsed"
            )
            heartbeat_callback.last_log = now

    try:
        stats = tagger.process_parquet_s3(
            input_path=f"s3://{file_path}",
            output_path=output_file,
            filesystem=fs,
            batch_size=BATCH_SIZE,
            progress_callback=heartbeat_callback,
            output_csv_path=local_csv_path,
            write_parquet=WRITE_PARQUET,
        )

        duration = time.time() - start_time
        result = {
            "input_file": file_path,
            "status": "success",
            "duration_sec": round(duration, 2),
            "rows_per_sec": round(stats["total_rows"] / duration, 2) if duration > 0 else 0,
            **stats,
        }
        if output_file is not None:
            result["output_file"] = output_file
        return result

    except Exception as e:
        return {
            "input_file": file_path,
            "output_file": output_file,
            "status": "failed",
            "error": str(e),
        }


# ------------------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------------------
def process_s3_bucket():
    fs = s3fs.S3FileSystem()

    logger.info(f"Scanning S3 bucket: {INPUT_S3_PREFIX}")
    # List all parquet files recursively
    all_inputs = [f for f in fs.glob(f"{INPUT_S3_PREFIX}/**/*.parquet") if not f.endswith(".metadata.parquet")]

    if not all_inputs:
        logger.error("No parquet files found in the specified S3 path.")
        return

    # Filter out already processed files (Checkpointing)
    logger.info("Checking for already processed files (checkpointing)...")
    to_process = []
    for f in all_inputs:
        input_prefix = INPUT_S3_PREFIX.replace("s3://", "")
        rel = f.replace(input_prefix, "").lstrip("/")
        if WRITE_PARQUET:
            out = f"{OUTPUT_S3_PREFIX.rstrip('/')}/{rel}"
            if fs.exists(out):
                continue
        else:
            # CSV-only: skip if local CSV exists
            local_csv = OUTPUT_CSV_DIR / rel.replace(".parquet", ".csv")
            if local_csv.exists():
                continue
        to_process.append(f)

    logger.info(f"Total files in source: {len(all_inputs)}")
    logger.info(f"Files already processed: {len(all_inputs) - len(to_process)}")

    if MAX_FILES is not None:
        to_process = to_process[:MAX_FILES]
        logger.info(f"Capping processing to {MAX_FILES} files due to MAX_FILES setting.")

    logger.info(f"Files to process in this run: {len(to_process)}")

    if not to_process:
        logger.info("No files to process. Exiting.")
        return

    # Distributed Execution with Backpressure
    pending = []
    failures = []
    success_count = 0
    total_rows_processed = 0
    file_iter = iter(to_process)

    # Fill initial pipeline
    for _ in range(min(MAX_INFLIGHT, len(to_process))):
        try:
            f = next(file_iter)
            pending.append(process_s3_file.remote(f))
        except StopIteration:
            break

    pbar = tqdm(total=len(to_process), desc="Scaling 1TB Tagging")

    while pending:
        done_ids, pending = ray.wait(pending, num_returns=1)

        for result_id in done_ids:
            res = ray.get(result_id)

            if res["status"] == "success":
                success_count += 1
                total_rows_processed += res.get("total_rows", 0)
                # Periodic logging for big runs
                if success_count % 10 == 0:
                    logger.info(
                        f"Progress: {success_count}/{len(to_process)} files | Total Rows: {total_rows_processed}"
                    )
            else:
                failures.append(res)
                logger.error(f"Failed file: {res.get('input_file')} | Error: {res.get('error')}")

            pbar.update(1)
            pbar.set_postfix(
                {
                    "rows": f"{total_rows_processed:,}",
                    "fails": len(failures),
                    "last_speed": f"{res.get('rows_per_sec', 0):.0f} r/s" if res["status"] == "success" else "ERR",
                }
            )

        # Refill pipeline to maintain MAX_INFLIGHT
        try:
            while len(pending) < MAX_INFLIGHT:
                f = next(file_iter)
                pending.append(process_s3_file.remote(f))
        except StopIteration:
            pass

    pbar.close()

    # Final Report
    logger.info("-" * 50)
    logger.info("RUN SUMMARY")
    logger.info("-" * 50)
    logger.info(f"Successfully processed: {success_count} files")
    logger.info(f"Total rows tagged: {total_rows_processed}")
    logger.info(f"Failures encountered: {len(failures)}")

    if failures:
        # Save failures to a separate file for targeted retry
        failure_log = "failed_files.txt"
        with open(failure_log, "w") as f:
            for fail in failures:
                f.write(f"{fail['input_file']}\t{fail.get('error')}\n")
        logger.info(f"Full failure list saved to: {failure_log}")

    ray.shutdown()


if __name__ == "__main__":
    process_s3_bucket()
