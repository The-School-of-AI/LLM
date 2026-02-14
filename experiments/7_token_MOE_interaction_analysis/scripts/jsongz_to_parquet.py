"""
Reads a text file of URLs, downloads each .json.gz file, converts it
to parquet, and saves it to an output directory. One parquet file per URL.

Each line in the URL file should be a single URL, e.g.:
    https://olmo-data.org/dolma-v1_6-8B-sample/v1_5r2_sample-0002.json.gz
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import ray


@ray.remote
def download_and_convert(url: str, output_dir: str) -> dict:
    """
    Downloads one .json.gz file, converts it to parquet, saves to output_dir.

    The output filename mirrors the source filename, e.g.:
        v1_5r2_sample-0002.json.gz -> v1_5r2_sample-0002.parquet

    Returns:
        {
            "status":       "completed" | "skipped" | "failed",
            "url":          str,
            "output_path":  str,
            "record_count": int,
            "error":        str  (only on failure),
        }
    """
    import gzip
    import io
    import json
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq
    import requests

    url = url.strip()
    if not url:
        return {"status": "skipped", "url": url, "output_path": "", "record_count": 0}

    stem = Path(url).name.replace(".json.gz", "").replace(".jsonl.gz", "")
    output_path = Path(output_dir) / f"{stem}.parquet"

    if output_path.exists():
        return {
            "status": "skipped",
            "url": url,
            "output_path": str(output_path),
            "record_count": 0,
        }

    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()

        records = []
        with gzip.open(io.BytesIO(response.content)) as gz:
            for line in gz:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

        if not records:
            return {
                "status": "failed",
                "url": url,
                "output_path": "",
                "record_count": 0,
                "error": "No records found in file",
            }

        table = pa.Table.from_pylist(records)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path)

        return {
            "status": "completed",
            "url": url,
            "output_path": str(output_path),
            "record_count": len(records),
        }

    except Exception as e:
        return {
            "status": "failed",
            "url": url,
            "output_path": "",
            "record_count": 0,
            "error": str(e),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Dolma URLs and convert to parquet"
    )
    parser.add_argument(
        "--urls", "-u", required=True, help="Text file containing one URL per line"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output directory for parquet files"
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of Ray workers (default: 4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    urls_file = Path(args.urls)
    output_dir = str(Path(args.output).resolve())

    if not urls_file.exists():
        print(f"ERROR: URLs file not found: {urls_file}")
        sys.exit(1)

    urls = [line.strip() for line in urls_file.read_text().splitlines() if line.strip()]
    if not urls:
        print("ERROR: No URLs found in file")
        sys.exit(1)

    print(f"Found {len(urls)} URLs")
    print(f"Output directory: {output_dir}")
    print(f"Processing with {args.workers} workers...")

    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    logging.getLogger("ray").setLevel(logging.ERROR)

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        num_cpus=args.workers,
        object_store_memory=2 * 1024 * 1024 * 1024,
    )

    futures = [download_and_convert.remote(url, output_dir) for url in urls]
    completed = skipped = failed = total_records = 0
    start = time.time()
    remaining = futures

    while remaining:
        done, remaining = ray.wait(remaining, num_returns=1)
        result: dict = ray.get(done[0])
        n = completed + skipped + failed + 1

        if result["status"] == "completed":
            completed += 1
            total_records += result["record_count"]
            print(
                f"  [{n}/{len(urls)}] {Path(result['url']).name} — {result['record_count']:,} records"
            )
        elif result["status"] == "skipped":
            skipped += 1
            print(
                f"  [{n}/{len(urls)}] {Path(result['url']).name} — skipped (already exists)"
            )
        else:
            failed += 1
            print(
                f"  [{n}/{len(urls)}] FAILED {Path(result['url']).name}: {result.get('error')}"
            )

    elapsed = time.time() - start

    print(f"\n{'=' * 55}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"  Completed : {completed}")
    print(f"  Skipped   : {skipped}")
    print(f"  Failed    : {failed}")
    print(f"  Total records written: {total_records:,}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
