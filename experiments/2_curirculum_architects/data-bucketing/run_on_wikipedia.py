import json
import os

from bucketer import DataBucketer
from datasets import load_dataset
from tqdm import tqdm


def main():
    # Explicitly set cache dir outside the repo
    cache_dir = "/home/ubuntu/dataset_cache"
    output_dir = "/home/ubuntu/results"

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Using cache directory: {cache_dir}")
    print(f"Results will be saved to: {output_dir}/{{BAND}}.jsonl")
    print("Loading dataset stream...")

    # Load dataset in streaming mode
    ds = load_dataset(
        "simple-pretraining/wikipedia_chunked",
        split="train",
        streaming=True,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    bucketer = DataBucketer()
    sample_count = 10000

    print(f"Processing {sample_count} samples...")

    # Open file handles for each known band (and a catch-all if needed)
    # We'll open them lazily or all at once. All at once is safer for keeping them open.
    bands = ["B0", "B1", "B2", "B3", "B4", "B5"]
    file_handles = {}

    try:
        # Open files
        for band in bands:
            file_handles[band] = open(
                os.path.join(output_dir, f"{band}.jsonl"), "w", encoding="utf-8"
            )

        # iterate with tqdm for progress bar
        for i, item in tqdm(enumerate(ds.take(sample_count)), total=sample_count):
            text = item.get("text", "")
            if not text:
                keys = list(item.keys())
                if "content" in keys:
                    text = item["content"]
                else:
                    text = str(item)

            result = bucketer.bucket_sample(text)

            # Create result object
            output_obj = {
                "id": i,
                "text_snippet": text[:200],
                "full_text": text,
                "band": result.band,
                "rationale": result.rationale,
                "flags": result.flags,
            }

            # Write to appropriate file
            if result.band in file_handles:
                file_handles[result.band].write(json.dumps(output_obj) + "\n")
            else:
                # Fallback for unexpected bands
                if "unknown" not in file_handles:
                    file_handles["unknown"] = open(
                        os.path.join(output_dir, "unknown.jsonl"), "w", encoding="utf-8"
                    )
                file_handles["unknown"].write(json.dumps(output_obj) + "\n")

    finally:
        # Close all files
        for f in file_handles.values():
            f.close()

    print(f"\nCompleted! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
