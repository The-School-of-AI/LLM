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

    # ------------------------
    # Load curriculum first (needed for version)
    # ------------------------
    print("Loading curriculum...")
    curriculum = load_curriculum(args.curriculum)
    curriculum_version = curriculum.get("version", "unknown")

    # Optional: add date later if you want
    # run_date = datetime.date.today().isoformat()

    # ------------------------
    # Build canonical output prefix
    # ------------------------
    base_prefix = args.out_prefix.rstrip("/")

    out_prefix = (
        f"{base_prefix}"
        f"/curriculum_{curriculum_version}"
        f"/seed_{args.seed}"
        # Later:
        # f"/{run_date}"
    )

    print(f"Output prefix: {out_prefix}")

    # ------------------------
    # Load global index
    # ------------------------
    print("Loading global index...")
    table = pq.read_table(args.index, filesystem=fs)

    # ------------------------
    # Deterministic shuffle
    # ------------------------
    print("Shuffling deterministically...")
    shuffled = deterministic_shuffle(table, seed=args.seed)

    # Attach provenance metadata
    shuffled = shuffled.replace_schema_metadata(
        {
            b"seed": str(args.seed).encode(),
            b"curriculum_version": str(curriculum_version).encode(),
        }
    )

    # ------------------------
    # Write shuffled global index
    # ------------------------
    shuffled_path = f"{out_prefix}/global_index_shuffled.parquet"
    print(f"Writing shuffled index to {shuffled_path}")

    with fs.open(shuffled_path, "wb") as f:
        pq.write_table(shuffled, f)

    # ------------------------
    # Build stage manifests
    # ------------------------
    print("Building stage manifests...")
    build_stage_manifests(
        shuffled,
        curriculum,
        out_prefix,
        filesystem=fs,
    )

    print("Done.")


if __name__ == "__main__":
    main()
