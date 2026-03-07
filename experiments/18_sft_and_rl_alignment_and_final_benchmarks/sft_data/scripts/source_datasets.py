#!/usr/bin/env python3
"""
Source (download) SFT datasets from canonical sources (Hugging Face).
Team 18 — Final datasets to source. Writes raw data to JSONL in output-dir.
Requires: pip install datasets
"""
import json
import argparse
from pathlib import Path

# Optional: Hugging Face datasets
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# Dataset ID and config mapping (HF dataset_id, config/split, optional subsample)
DATASET_CONFIG = {
    "tulu3": {"id": "allenai/tulu-3-sft-mixture", "split": "train", "subsample": 100000},
    "openmath": {"id": "nvidia/OpenMathInstruct-2", "split": "train", "subsample": 50000},
    "magicoder": {"id": "ise-uiuc/Magicoder-OSS-Instruct-75K", "split": "train", "subsample": 50000},
    "numinamath": {"id": "AI-MO/NuminaMath-TIR", "split": "train", "subsample": None},
    "pku_saferlhf": {"id": "PKU-Alignment/PKU-SafeRLHF", "split": "train", "subsample": 20000},
    # Add more as needed: spider, bird, swe_smith, indicalign (check HF for exact IDs)
}


def export_to_jsonl(dataset, path: Path, subsample: int | None = None):
    """Write dataset to JSONL. Assumes dataset has dict-like rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for i, row in enumerate(dataset):
            if subsample and i >= subsample:
                break
            # Handle both dict and Dataset row (may have column names)
            if hasattr(row, "keys"):
                obj = dict(row)
            else:
                obj = row
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="Source SFT datasets from Hugging Face to JSONL.")
    ap.add_argument("--output-dir", type=Path, default=Path("raw_data"), help="Output directory for JSONL files")
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["tulu3", "openmath", "magicoder"],
        help="Which datasets to source: tulu3, openmath, magicoder, numinamath, pku_saferlhf, or 'all'",
    )
    ap.add_argument("--subsample", type=int, default=None, help="Override: max rows per dataset (default: use DATASET_CONFIG)")
    args = ap.parse_args()

    if not HAS_DATASETS:
        raise SystemExit("Install Hugging Face datasets: pip install datasets")

    if args.datasets == ["all"]:
        keys = list(DATASET_CONFIG.keys())
    else:
        keys = [k for k in args.datasets if k in DATASET_CONFIG]
        if len(keys) != len(args.datasets):
            unknown = set(args.datasets) - set(DATASET_CONFIG.keys())
            if unknown and "all" not in args.datasets:
                print(f"Unknown dataset(s): {unknown}; available: {list(DATASET_CONFIG.keys())}")

    for key in keys:
        cfg = DATASET_CONFIG[key]
        ds_id = cfg["id"]
        split = cfg.get("split", "train")
        subsample = args.subsample if args.subsample is not None else cfg.get("subsample")
        print(f"Loading {key}: {ds_id} (split={split}, subsample={subsample})...")
        try:
            ds = load_dataset(ds_id, split=split, trust_remote_code=True)
            out_path = args.output_dir / f"{key}.jsonl"
            n = export_to_jsonl(ds, out_path, subsample=subsample)
            print(f"  -> Wrote {n} rows to {out_path}")
        except Exception as e:
            print(f"  -> Failed: {e}")

    print("Done. Use standardize_conversation_format.py on each JSONL (or combined) next.")


if __name__ == "__main__":
    main()
