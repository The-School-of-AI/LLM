import argparse
import csv
import os
from urllib.parse import urlparse

import tiktoken
from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
from transformers import AutoTokenizer


def get_dataset_name_from_url(url):
    """Extracts dataset name from Hugging Face URL."""
    # Example: https://huggingface.co/datasets/glue -> glue
    # Example: https://huggingface.co/datasets/allenai/c4 -> allenai/c4
    path = urlparse(url).path
    if path.startswith("/datasets/"):
        return path[10:]
    return path.strip("/")


def estimate_tokens(dataset_name, tokenizer_name="cl100k_base", config_name=None):
    print(f"Loading tokenizer: {tokenizer_name}...")
    try:
        encoding = tiktoken.get_encoding(tokenizer_name)
    except Exception:
        # Fallback to transformers if not a tiktoken encoding
        print(f"Tiktoken encoding {tokenizer_name} not found. Trying transformers...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        encoding = None

    target_configs = []
    if config_name:
        target_configs = [config_name]
    else:
        try:
            target_configs = get_dataset_config_names(dataset_name)
            if not target_configs:
                target_configs = [None]
            print(f"Found {len(target_configs)} configurations: {target_configs}")
        except Exception as e:
            print(f"Could not fetch configs (might be simple dataset): {e}")
            target_configs = [None]

    results = []

    for current_config in target_configs:
        print(f"Processing config: {current_config}")

        print(
            f"Fetching splits for dataset: {dataset_name}"
            + (f" (config: {current_config})" if current_config else "")
            + "..."
        )
        try:
            splits = get_dataset_split_names(dataset_name, config_name=current_config)
        except Exception as e:
            print(f"Error fetching splits for config {current_config}: {e}")
            continue

        for split in splits:
            print(f"Processing split: {split} (config: {current_config})...")
            try:
                # Load in streaming mode to handle large datasets
                ds = load_dataset(
                    dataset_name, split=split, streaming=True, name=current_config
                )

                # Peel off first item to check columns
                iterator = iter(ds)
                try:
                    first_item = next(iterator)
                except StopIteration:
                    print(f"  Split {split} is empty. Skipping.")
                    continue

                # Identify ALL string columns
                text_cols = []
                for key, val in first_item.items():
                    if isinstance(val, str):
                        text_cols.append(key)

                if not text_cols:
                    print(
                        f"  Warning: Could not identify any string columns for split {split}. Skipping."
                    )
                    continue

                print(f"  Found string columns: {text_cols}")

                # Initialize counters for each column
                col_counters = {col: 0 for col in text_cols}

                # Helper function to count tokens
                def count_text_tokens(text):
                    if not text:
                        return 0
                    if encoding:
                        return len(encoding.encode(text))
                    else:
                        return len(tokenizer(text)["input_ids"])

                # Process first item
                for col in text_cols:
                    col_counters[col] += count_text_tokens(first_item.get(col, ""))

                # Continue with rest of the items
                for i, example in enumerate(iterator):
                    for col in text_cols:
                        col_counters[col] += count_text_tokens(example.get(col, ""))

                    if (i + 1) % 1000 == 0:
                        print(f"  Processed {i + 1} rows in {split}...", end="\r")

                print(f"  Finished split {split} (config: {current_config}).")

                for col, count in col_counters.items():
                    results.append(
                        {
                            "dataset": dataset_name,
                            "config": current_config if current_config else "default",
                            "split": split,
                            "tokens": count,
                            "text_column": col,
                        }
                    )

            except Exception as e:
                print(f"Error processing split {split} (config: {current_config}): {e}")

    return results


def append_to_csv(results, filepath="token_counts.csv"):
    file_exists = os.path.isfile(filepath)
    keys = ["dataset", "config", "split", "tokens", "text_column"]

    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)
    print(f"Appended results to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Count tokens in a Hugging Face dataset."
    )
    parser.add_argument("url", help="URL of the Hugging Face dataset")
    parser.add_argument(
        "--tokenizer",
        default="cl100k_base",
        help="Tokenizer to use (default: cl100k_base)",
    )
    parser.add_argument("--config", help="Dataset configuration (optional)")
    parser.add_argument("--output", default="token_counts.csv", help="Output CSV file")

    args = parser.parse_args()

    dataset_name = get_dataset_name_from_url(args.url)
    print(f"Dataset Name: {dataset_name}")

    results = estimate_tokens(dataset_name, args.tokenizer, args.config)

    if results:
        append_to_csv(results, args.output)
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
