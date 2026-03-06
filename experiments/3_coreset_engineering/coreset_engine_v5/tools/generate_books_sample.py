"""Generate a small 'books' source sample for local pipeline testing.

Produces a JSONL file with schema compatible with the streaming coreset builder
(source, chunk_id, band, domain, token_count_estimate, source_doc_id, source_url,
t1_file_path, etc.) so you can run the full pipeline locally without S3.

source_url is T2-style (band path). t1_file_path is a local-test placeholder.
For production, use real T2 output as input; source_url will pass through.

Usage:
  python tools/generate_books_sample.py --out data/local_test/books/sample.jsonl --num-chunks 500
  # Then run pipeline:
  bash scripts/run_local_books_test.sh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a small books-source JSONL for local pipeline test"
    )
    ap.add_argument(
        "--out",
        type=str,
        default="data/local_test/books/sample.jsonl",
        help="Output JSONL path",
    )
    ap.add_argument(
        "--num-chunks",
        type=int,
        default=800,
        help="Number of chunks to generate (default 800)",
    )
    ap.add_argument(
        "--chunk-tokens",
        type=int,
        default=512,
        help="Token count per chunk (default 512)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for band distribution",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Books in T3StatsFromT2: literature domain, B0-B5. Approximate band distribution.
    bands = ["B0", "B1", "B2", "B3", "B4", "B5"]
    # Simple distribution so we have some in each band (1B stage can select)
    rng = __import__("random").Random(args.seed)
    band_weights = [0.01, 0.20, 0.35, 0.25, 0.12, 0.07]
    total_tokens = 0

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(args.num_chunks):
            band = rng.choices(bands, weights=band_weights, k=1)[0]
            tc = args.chunk_tokens
            total_tokens += tc
            # source_url = T2-style band path (passthrough in T3 output). t1_file_path = local-test only.
            part_name = "part-00000-7afc06ff-4184-4e98-bd70-9c0c0da9dfc3-c000.zstd.parquet"
            row = {
                "chunk_id": f"books_chunk_{i:07d}",
                "dataset_id": "books",
                "source": "books",
                "token_count_estimate": tc,
                "byte_length": tc * 4,
                "domain": "literature",
                "language": "en",
                "band": band,
                "source_doc_id": part_name,
                "source_url": f"s3://t2-bucket/processed_dataset/curriculum_pyspark_output/source=books/band={band}/{part_name}",
                "t1_file_path": f"local-test://books/{part_name}",
            }
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {args.num_chunks} chunks to {out_path}")
    print(f"  total_tokens: {total_tokens}")
    print(f"  Use with shard.sh: --total-input-tokens-estimate {total_tokens}")


if __name__ == "__main__":
    main()
