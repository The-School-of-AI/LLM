import argparse

import pyarrow.parquet as pq
import s3fs
import yaml
from shuffle import deterministic_shuffle
from stages import build_stage_manifests


def load_curriculum(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    fs = s3fs.S3FileSystem()

    print("Loading global index...")
    table = pq.read_table(args.index, filesystem=fs)

    print("Shuffling deterministically...")
    shuffled = deterministic_shuffle(table, seed=args.seed)

    shuffled_path = f"{args.out_prefix.rstrip('/')}/global_index_shuffled.parquet"
    pq.write_table(shuffled, fs.open(shuffled_path, "wb"))

    print("Loading curriculum...")
    curriculum = load_curriculum(args.curriculum)

    print("Building stage manifests...")
    build_stage_manifests(
        shuffled,
        curriculum,
        args.out_prefix,
        filesystem=fs,
    )

    print("Done.")


if __name__ == "__main__":
    main()
