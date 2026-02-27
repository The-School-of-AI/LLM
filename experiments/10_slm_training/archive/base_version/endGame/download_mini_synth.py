#!/usr/bin/env python3
"""
Download and save a local HuggingFace dataset artifact for training.

Example:
    python download_mini_synth.py --output-dir ../synth_local_en --max-samples 2000
"""

import argparse
import os
import sys
from typing import Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


def is_hf_saved_dataset_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    return os.path.isfile(os.path.join(path, "dataset_info.json")) or os.path.isfile(
        os.path.join(path, "dataset_dict.json")
    )


def load_split(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
) -> Tuple[Dataset, str]:
    """Load a non-streaming dataset split."""
    try:
        ds = load_dataset(dataset_name, dataset_config, split=split)
        return ds, split
    except Exception:
        ds_obj = load_dataset(dataset_name, dataset_config)
        if isinstance(ds_obj, Dataset):
            return ds_obj, split
        if isinstance(ds_obj, DatasetDict):
            if split in ds_obj:
                return ds_obj[split], split
            for candidate in ("train", "validation", "test"):
                if candidate in ds_obj:
                    return ds_obj[candidate], candidate
            first_split = next(iter(ds_obj.keys()))
            return ds_obj[first_split], first_split
        raise RuntimeError("Unexpected dataset object type from load_dataset().")


def load_streaming_split(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
):
    """Load a streaming dataset split (IterableDataset)."""
    try:
        ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
        return ds, split
    except Exception:
        ds_obj = load_dataset(dataset_name, dataset_config, streaming=True)
        if split in ds_obj:
            return ds_obj[split], split
        for candidate in ("train", "validation", "test"):
            if candidate in ds_obj:
                return ds_obj[candidate], candidate
        first_split = next(iter(ds_obj.keys()))
        return ds_obj[first_split], first_split


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download PleIAs/SYNTH (or another dataset) and save in save_to_disk() format."
    )
    parser.add_argument("--dataset-name", default="PleIAs/SYNTH")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--output-dir", default="../synth_local_en")
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Stream and save only requested rows (default: enabled).",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Disable streaming and load full split before trimming.",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    print(f"Target output: {output_dir}")

    if is_hf_saved_dataset_dir(output_dir):
        try:
            existing = load_from_disk(output_dir)
            print(f"Dataset already exists with {len(existing)} rows. Nothing to do.")
        except Exception:
            print("Dataset directory already exists. Nothing to do.")
        return 0

    if os.path.exists(output_dir):
        if not os.path.isdir(output_dir):
            print(
                "Output path exists and is not a directory:\n"
                f"  {output_dir}\n"
                "Use a different --output-dir."
            )
            return 1
        if os.listdir(output_dir):
            print(
                "Output path exists and is not empty:\n"
                f"  {output_dir}\n"
                "Use a different --output-dir or clean this path first."
            )
            return 1
    else:
        os.makedirs(output_dir, exist_ok=True)

    print(
        f"Downloading dataset '{args.dataset_name}'"
        + (f" (config='{args.dataset_config}')" if args.dataset_config else "")
        + f", split='{args.split}', streaming={args.streaming}..."
    )

    required_cols = ["query", "synthetic_reasoning", "synthetic_answer", "language"]

    if args.streaming:
        ds_iter, resolved_split = load_streaming_split(
            args.dataset_name, args.dataset_config, args.split
        )
        print(f"Streaming split '{resolved_split}'")

        rows = []
        inspected = 0
        target = (
            args.max_samples if args.max_samples and args.max_samples > 0 else 50000
        )
        lang_filter = args.language.lower() if args.language else None
        has_language_key = None

        for ex in ds_iter:
            inspected += 1

            if has_language_key is None:
                has_language_key = "language" in ex
                if lang_filter and not has_language_key:
                    print("No 'language' key in records. Skipping language filter.")

            if lang_filter and has_language_key:
                if str(ex.get("language", "")).lower() != lang_filter:
                    continue

            row = {col: ex.get(col, "") for col in required_cols}
            if not str(row.get("query", "")).strip():
                continue

            rows.append(row)

            if len(rows) % 1000 == 0:
                print(f"Collected {len(rows)} rows (inspected {inspected})")

            if len(rows) >= target:
                break

        if not rows:
            print(
                "No rows collected. Try --no-streaming, --language '', or a different --split."
            )
            return 1

        ds = Dataset.from_list(rows)
        print(f"Collected {len(ds)} rows total (inspected {inspected})")
    else:
        ds, resolved_split = load_split(
            args.dataset_name, args.dataset_config, args.split
        )
        print(f"Loaded split '{resolved_split}' with {len(ds)} rows")

        if args.language and "language" in ds.column_names:
            language = args.language.lower()
            before = len(ds)
            ds = ds.filter(lambda ex: str(ex.get("language", "")).lower() == language)
            print(f"Filtered language='{args.language}': {before} -> {len(ds)} rows")
        elif args.language:
            print("No 'language' column found. Skipping language filter.")

        if args.max_samples and len(ds) > args.max_samples:
            ds = ds.select(range(args.max_samples))
            print(f"Trimmed to max_samples={args.max_samples}: {len(ds)} rows")

        present_cols = [c for c in required_cols if c in ds.column_names]
        if present_cols:
            ds = ds.select_columns(present_cols)
            print(f"Keeping columns: {present_cols}")

    print(f"Saving to disk: {output_dir}")
    ds.save_to_disk(output_dir)
    print("Done.")
    print("Now run: python train_recurrence_1b.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
