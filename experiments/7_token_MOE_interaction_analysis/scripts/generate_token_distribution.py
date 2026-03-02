import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from moeint.expert_analysis import ModalityDistribution
from transformers import AutoTokenizer

TokenCountDict = dict[str, np.ndarray]  # modality -> count array of shape [vocab_size]


@ray.remote
def process_file(
    file_path: str, tokenizer_path: str, vocab_size: int
) -> dict[str, Any]:
    """
    Runs on a Ray worker. Initializes the tokenizer and modality metric once,
    then processes records in batches.

    Key optimisations:
    - Batch tokenization: encodes a whole parquet batch in one tokenizer call.
    - numpy bincount: accumulates token counts without a Python loop per record.
    - Dense numpy arrays per modality: no sparse dicts, no merge overhead.

    Returns:
        {
            "status":       "completed" | "failed",
            "file_path":    str,
            "counts":       {modality: np.ndarray of shape [vocab_size]},
            "num_records":  int,
            "error":        str  (only on failure),
        }
    """
    import tempfile

    import numpy as np
    import pyarrow.parquet as pq
    import yaml
    from curriculum_tags.metrics.modality import ModalityMetric
    from curriculum_tags.utils.curriculum_loader import CurriculumConfig
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"version": "0.1"}, f)
        config = CurriculumConfig(f.name)
    metric = ModalityMetric(config)

    counts: dict[str, np.ndarray] = {}
    records_processed = 0

    try:
        pf = pq.ParquetFile(file_path)

        for batch in pf.iter_batches(batch_size=5000):
            records = batch.to_pylist()

            valid = [(r, r["text"]) for r in records if r.get("text")]
            if not valid:
                continue

            valid_records, texts = zip(*valid)

            # Batch tokenize the whole parquet batch in one call
            encoded = tokenizer(
                list(texts),
                add_special_tokens=False,
                return_attention_mask=False,
                truncation=False,
            )

            for record, token_ids in zip(valid_records, encoded["input_ids"]):
                modality = metric.compute(record)["primary_modality"]

                if modality not in counts:
                    counts[modality] = np.zeros(vocab_size, dtype=np.int64)

                counts[modality] += np.bincount(token_ids, minlength=vocab_size)
                records_processed += 1

        return dict(
            status="completed",
            file_path=file_path,
            counts=counts,
            num_records=records_processed,
        )

    except Exception as e:
        return dict(
            status="failed", file_path=file_path, counts={}, num_records=0, error=str(e)
        )


def merge_counts(results: list[dict[str, Any]], vocab_size: int) -> TokenCountDict:
    """Sum numpy count arrays from all workers into one dict."""
    merged: dict[str, np.ndarray] = {}
    for result in results:
        for modality, arr in result["counts"].items():
            if modality not in merged:
                merged[modality] = np.zeros(vocab_size, dtype=np.int64)
            merged[modality] += arr
    return merged


def build_distribution(
    token_counts: TokenCountDict,
    source_files: list[str],
) -> ModalityDistribution:
    """Convert merged count arrays into a normalised ModalityDistribution."""
    modalities = sorted(token_counts.keys())
    index_to_modality = {i: m for i, m in enumerate(modalities)}

    tensor = torch.stack(
        [torch.from_numpy(token_counts[m].astype(np.float32)) for m in modalities]
    )

    # Normalise each row independently so each modality sums to 1
    row_sums = tensor.sum(dim=1, keepdim=True)
    tensor = tensor / row_sums.clamp(min=1e-10)

    return ModalityDistribution(
        distribution=tensor,
        index_to_modality=index_to_modality,
        source_files=source_files,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Token distribution by modality")
    parser.add_argument(
        "--input", "-i", required=True, help="Directory of parquet files"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output .pt file path",
    )
    parser.add_argument(
        "--tokenizer",
        "-t",
        required=True,
        help="Path to tokenizer directory / tokenizer name",
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

    input_dir = Path(args.input)
    output_path = Path(args.output)
    tokenizer = args.tokenizer

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    files = sorted(str(p) for p in input_dir.glob("**/*.parquet"))
    if not files:
        print(f"ERROR: No parquet files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(files)} parquet files")

    vocab_size = AutoTokenizer.from_pretrained(tokenizer).vocab_size
    print(f"Vocab size: {vocab_size:,}")
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

    futures = [
        process_file.remote(str(Path(fp).resolve()), tokenizer, vocab_size)
        for fp in files
    ]
    results = []
    completed = failed = total_records = 0
    start = time.time()
    remaining = futures

    while remaining:
        done, remaining = ray.wait(remaining, num_returns=1)
        result: dict = ray.get(done[0])

        if result["status"] == "completed":
            completed += 1
            total_records += result["num_records"]
            results.append(result)
            print(
                f"  [{completed + failed}/{len(files)}] {Path(result['file_path']).name} — {result['num_records']:,} records"
            )
        else:
            failed += 1
            print(f"  [FAILED] {Path(result['file_path']).name}: {result.get('error')}")

    elapsed = time.time() - start

    print("\nMerging counts...")
    merged = merge_counts(results, vocab_size)

    print("Building distribution tensor...")
    source_files = [r["file_path"] for r in results]
    dist = build_distribution(merged, source_files)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dist.save(output_path)

    total_tokens = sum(int(arr.sum()) for arr in merged.values())
    print(f"\n{'=' * 55}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"  Files processed : {completed}/{len(files)}")
    print(f"  Files failed    : {failed}")
    print(f"  Total records   : {total_records:,}")
    print(f"  Total tokens    : {total_tokens:,}")
    print(f"  Tensor shape    : {list(dist.distribution.shape)}")
    print("Token distribution by modality (normalised):")
    for i, modality in dist.index_to_modality.items():
        raw_count = int(merged[modality].sum())
        pct = 100 * raw_count / total_tokens if total_tokens else 0
        print(f"  {modality:<25} {raw_count:>15,}  ({pct:.2f}%)")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
