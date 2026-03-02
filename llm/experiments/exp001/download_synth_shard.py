import argparse
import os

from datasets import load_dataset
from huggingface_hub import hf_hub_download


def main(output_dir: str, shard_number: str):
    shard_dir = os.path.join(output_dir, "shard")
    data_file_name = f"synth_{shard_number}.parquet"
    hf_hub_download(
        repo_id="PleIAs/SYNTH",
        filename=data_file_name,
        repo_type="dataset",
        local_dir=shard_dir,
    )

    ds = load_dataset(
        "parquet", data_files=os.path.join(shard_dir, data_file_name), split="train"
    )
    ds.save_to_disk(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", required=True, help="output dir")
    parser.add_argument("-s", default="047", help="dataset shard number")

    args = parser.parse_args()
    main(args.o, args.s)
