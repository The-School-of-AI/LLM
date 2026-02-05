import argparse
import concurrent.futures
import glob
import os
import sys
from typing import List, Optional

import duckdb


def _process_single_file(
    file_path: str,
    output_dir: str,
    domain: str,
    version: str,
    external_source: str,
    duckdb_threads: int,
    duckdb_memory_limit: str,
    duckdb_temp_base: str,
) -> None:
    """
    Worker function: convert a single Dolma .json.gz file to Parquet using DuckDB.
    Runs in its own process to avoid GIL and share no state between workers.
    """
    file_name = os.path.basename(file_path).replace(".json.gz", ".parquet")
    output_path = os.path.join(output_dir, file_name)

    # Skip if output already exists (idempotent runs, useful for restarts)
    if os.path.exists(output_path):
        print(f"[SKIP] {file_name} already exists")
        return

    # Each worker gets its own temp directory to avoid contention
    pid = os.getpid()
    worker_temp_dir = os.path.join(duckdb_temp_base, f"duckdb_temp_{pid}")
    os.makedirs(worker_temp_dir, exist_ok=True)

    # Local DuckDB connection per worker
    con = duckdb.connect(database=":memory:")

    # Tuning: adjust based on instance size
    con.execute(f"SET threads = {duckdb_threads}")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET memory_limit = '{duckdb_memory_limit}'")
    con.execute(f"SET temp_directory = '{worker_temp_dir}'")
    con.execute("SET max_temp_directory_size = '200GB'")

    print(f"[START] {file_name}")

    query = f"""
        COPY (
            SELECT 
                id,
                sha256(text) AS hash,
                'dolma' AS dataset,
                '{domain}' AS domain,
                '{external_source}' AS source,
                text,
                'en' AS language,
                CAST(metadata AS VARCHAR) AS metadata,
                added,
                created,
                '{version}' AS version
            FROM read_json_auto('{file_path}',
                format='newline_delimited',
                compression='gzip',
                maximum_object_size=1073741824  -- 1GB per line/object
            )
        ) TO '{output_path}' (FORMAT 'parquet', COMPRESSION 'SNAPPY');
    """

    try:
        con.execute(query)
        print(f"[DONE]  {file_name}")
    except Exception as e:
        print(f"[ERROR] {file_name}: {e}", file=sys.stderr)
    finally:
        con.close()


def process_dolma_to_parquet_parallel(
    input_glob: str,
    output_dir: str,
    domain: str,
    version: str = "1.7",
    external_source: Optional[str] = None,
    workers: Optional[int] = None,
    duckdb_threads: int = 2,
    duckdb_memory_limit: str = "16GB",
    duckdb_temp_base: str = "./duckdb_temp/",
) -> None:
    """
    Parallel, multi-process converter for local Dolma .json.gz files using DuckDB.

    Strategy:
    - Fan out over files with multiple worker processes (use CPU cores efficiently).
    - Each worker uses its own DuckDB connection and temp directory.
    - Within each DuckDB connection, use a small number of threads (1–4) for stability.

    Recommended usage on a larger instance (e.g. 16 vCPUs):
    - workers: 8–12
    - duckdb_threads: 2–4
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(duckdb_temp_base, exist_ok=True)

    files: List[str] = glob.glob(input_glob)
    if not files:
        print(f"No files found matching: {input_glob}")
        return

    files.sort()
    print(f"Found {len(files)} files. Starting parallel conversion...")

    if workers is None or workers <= 0:
        # Use (CPU count - 1) workers by default, but at least 1
        cpu_count = os.cpu_count() or 1
        workers = max(1, cpu_count - 1)

    print(
        f"Using {workers} worker processes, "
        f"{duckdb_threads} DuckDB threads per worker, "
        f"memory_limit={duckdb_memory_limit}"
    )

    # Fan-out over files
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _process_single_file,
                file_path,
                output_dir,
                domain,
                version,
                external_source,
                duckdb_threads,
                duckdb_memory_limit,
                duckdb_temp_base,
            )
            for file_path in files
        ]

        # Simple progress loop
        completed = 0
        total = len(futures)
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            # Trigger any exceptions early
            try:
                future.result()
            except Exception as e:
                print(f"[WORKER ERROR] {e}", file=sys.stderr)
            if completed % 10 == 0 or completed == total:
                print(f"[PROGRESS] {completed}/{total} files processed")

    print(f"\nSuccess! All files converted to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Dolma JSONL.GZ to Parquet in parallel using DuckDB."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Glob pattern for input files (e.g. '/data/dolma/*.json.gz')",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for Parquet files",
    )
    parser.add_argument(
        "--domain",
        required=True,
        help="Domain tag (e.g. 'web', 'code', 'math')",
    )
    parser.add_argument(
        "--version",
        default="1.7",
        help="Dataset version tag",
    )
    parser.add_argument(
        "--external_source",
        required=True,
        help="Source tag to use for all records (e.g. 'books', 'web')",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count - 1)",
    )
    parser.add_argument(
        "--duckdb_threads",
        type=int,
        default=2,
        help="DuckDB threads per worker (default: 2). Keep small for stability.",
    )
    parser.add_argument(
        "--duckdb_memory_limit",
        type=str,
        default="16GB",
        help="DuckDB memory_limit setting (e.g. '16GB'). Tune to your instance.",
    )
    parser.add_argument(
        "--duckdb_temp_base",
        type=str,
        default="./duckdb_temp/",
        help="Base directory for DuckDB temp files.",
    )

    args = parser.parse_args()

    process_dolma_to_parquet_parallel(
        input_glob=args.input,
        output_dir=args.output,
        domain=args.domain,
        version=args.version,
        external_source=args.external_source,
        workers=args.workers,
        duckdb_threads=args.duckdb_threads,
        duckdb_memory_limit=args.duckdb_memory_limit,
        duckdb_temp_base=args.duckdb_temp_base,
    )


if __name__ == "__main__":
    main()

