import argparse
import json
import os
import time
from itertools import islice
from pathlib import Path

import boto3
import yaml
from datasets import load_dataset

# -------------------------
# Helpers
# -------------------------


def apply_limit(dataset, limit_cfg, mode):
    """Apply test limits to dataset based on configuration"""
    if mode == "full" or limit_cfg["type"] == "none":
        return dataset
    if limit_cfg["type"] == "rows":
        return islice(dataset, limit_cfg["value"])
    if limit_cfg["type"] == "percent":
        percent = limit_cfg["value"]
        # For streaming datasets, approximate
        return islice(dataset, int(10000 * percent / 100))  # Approximate
    raise ValueError(f"Unknown limit type: {limit_cfg['type']}")


def save_jsonl_local(records, filepath):
    """Save records to local JSONL file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  💾 Saved: {filepath}")


def upload_jsonl_to_s3(records, bucket, key):
    """Upload records to S3 as JSONL"""
    s3 = boto3.client("s3")
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    s3.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"))
    print(f"  ⬆️  Uploaded: s3://{bucket}/{key}")


def save_records(records, mode, local_path, bucket, s3_key):
    """Save records based on storage mode"""
    if mode in ("local", "both"):
        save_jsonl_local(records, local_path)
    if mode in ("s3", "both"):
        upload_jsonl_to_s3(records, bucket, s3_key)


def process_stream(ds_iter, shard_size, cfg, local_subdir, s3_subdir):
    """Process streaming dataset and save in shards"""
    buffer, shard_id, total = [], 0, 0

    for r in ds_iter:
        buffer.append(r)
        total += 1

        if len(buffer) >= shard_size:
            shard = f"part-{shard_id:05d}.jsonl"
            save_records(
                buffer,
                cfg["storage"]["mode"],
                os.path.join(cfg["storage"]["local_dir"], local_subdir, shard),
                cfg["aws"]["s3_bucket"],
                f"{cfg['aws']['s3_prefix']}/{s3_subdir}/{shard}",
            )
            buffer.clear()
            shard_id += 1
            if total % 10000 == 0:
                print(f"  📊 Progress: {total:,} records")

    # Save remaining records
    if buffer:
        shard = f"part-{shard_id:05d}.jsonl"
        save_records(
            buffer,
            cfg["storage"]["mode"],
            os.path.join(cfg["storage"]["local_dir"], local_subdir, shard),
            cfg["aws"]["s3_bucket"],
            f"{cfg['aws']['s3_prefix']}/{s3_subdir}/{shard}",
        )

    return total, shard_id + 1


# -------------------------
# Dataset Handlers
# -------------------------


def process_sangraha(dcfg, cfg, mode):
    """Process Sangraha dataset with multiple languages"""
    print("  📥 Loading Sangraha dataset...")
    for lang in dcfg["languages"]:
        print(f"  🌐 Language: {lang}")
        ds = load_dataset(
            dcfg["repo"],
            dcfg["subset"],
            split=lang,
            streaming=True,
        )
        ds_iter = apply_limit(ds, dcfg["test_limit"], mode)
        total, shards = process_stream(
            ds_iter,
            10_000,
            cfg,
            f"{dcfg['local_path']}/{lang}",
            f"{dcfg['s3_path']}/{lang}",
        )
        print(f"  ✅ {lang}: {total:,} records, {shards} shards")


def process_indiccorp_v2(dcfg, cfg, mode):
    """Process IndicCorp V2 dataset"""
    print(f"  📥 Loading IndicCorp V2 (split: {dcfg['split']})...")
    ds = load_dataset(
        dcfg["repo"],
        dcfg["name"],
        split=dcfg["split"],
        streaming=True,
    )
    ds_iter = apply_limit(ds, dcfg["test_limit"], mode)
    total, shards = process_stream(
        ds_iter,
        10_000,
        cfg,
        dcfg["local_path"],
        dcfg["s3_path"],
    )
    print(f"  ✅ Total: {total:,} records, {shards} shards")


def process_dolma(dcfg, cfg, mode):
    """Process Dolma dataset"""
    print("  📥 Loading Dolma...")
    ds = load_dataset(
        dcfg["repo"],
        dcfg["name"],
        split=dcfg["split"],
        streaming=True,
    )
    ds_iter = apply_limit(ds, dcfg["test_limit"], mode)
    total, shards = process_stream(
        ds_iter,
        10_000,
        cfg,
        dcfg["local_path"],
        dcfg["s3_path"],
    )
    print(f"  ✅ Total: {total:,} records, {shards} shards")


def process_generic(name, dcfg, cfg, mode):
    """Process generic HuggingFace dataset"""
    print(f"  📥 Loading {name}...")

    load_args = {
        "path": dcfg["repo"],
        "split": dcfg.get("split", "train"),
        "streaming": True,
    }

    if "name" in dcfg:
        load_args["name"] = dcfg["name"]

    ds = load_dataset(**load_args)
    ds_iter = apply_limit(ds, dcfg["test_limit"], mode)
    total, shards = process_stream(
        ds_iter,
        10_000,
        cfg,
        dcfg["local_path"],
        dcfg["s3_path"],
    )
    print(f"  ✅ Total: {total:,} records, {shards} shards")


# -------------------------
# Main
# -------------------------

# -------------------------
# Argument Parsing
# -------------------------


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Download HuggingFace datasets to local storage and/or S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config.yml defaults
  python download.py

  # Override S3 bucket
  python download.py --s3-bucket my-custom-bucket

  # Download full datasets to S3
  python download.py --mode full --storage s3 --s3-bucket my-bucket

  # Download to both local and S3
  python download.py --storage both --s3-bucket my-bucket --region us-west-2
        """,
    )

    parser.add_argument(
        "--s3-bucket", type=str, help="S3 bucket name (overrides config.yml)"
    )

    parser.add_argument("--region", type=str, help="AWS region (overrides config.yml)")

    parser.add_argument(
        "--storage",
        type=str,
        choices=["local", "s3", "both"],
        help="Storage mode: local, s3, or both (overrides config.yml)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "full"],
        help="Download mode: test (limited) or full (complete datasets)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to config file (default: config.yml)",
    )

    return parser.parse_args()


# -------------------------
# Main
# -------------------------


def main():
    # Parse command-line arguments
    args = parse_args()

    print("=" * 60)
    print("🚀 Dataset Download Tool")
    print("=" * 60)

    # Load configuration
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override config with command-line arguments
    if args.s3_bucket:
        cfg["aws"]["s3_bucket"] = args.s3_bucket
        print(f"🔧 Using S3 bucket from command line: {args.s3_bucket}")

    if args.region:
        cfg["aws"]["region"] = args.region
        print(f"🔧 Using AWS region from command line: {args.region}")

    if args.storage:
        cfg["storage"]["mode"] = args.storage
        print(f"🔧 Using storage mode from command line: {args.storage}")

    if args.mode:
        cfg["mode"] = args.mode
        print(f"🔧 Using download mode from command line: {args.mode}")

    mode = cfg["mode"]
    print(f"\n⚙️  Mode: {mode}")
    print(f"💾 Storage: {cfg['storage']['mode']}")
    print(f"📁 Local dir: {cfg['storage']['local_dir']}")

    # Show S3 info if using S3
    if cfg["storage"]["mode"] in ["s3", "both"]:
        print(f"☁️  S3 bucket: s3://{cfg['aws']['s3_bucket']}/{cfg['aws']['s3_prefix']}")
        print(f"🌍 AWS region: {cfg['aws']['region']}")

    times = {}
    total_start = time.time()

    for name, dcfg in cfg["datasets"].items():
        print(f"\n{'='*60}")
        print(f"🚀 Processing: {name}")
        print(f"{'='*60}")
        start = time.time()

        try:
            if name == "sangraha":
                process_sangraha(dcfg, cfg, mode)
            elif name == "indiccorp_v2":
                process_indiccorp_v2(dcfg, cfg, mode)
            else:
                # Generic handler for other datasets (including dolma)
                process_generic(name, dcfg, cfg, mode)

            elapsed = time.time() - start
            times[name] = elapsed
            print(f"⏱️  Time: {elapsed:.2f}s")

        except Exception as e:
            print(f"❌ Error processing {name}: {e}")
            times[name] = -1

    # Summary
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    for k, v in times.items():
        if v >= 0:
            print(f"✅ {k}: {v:.2f}s")
        else:
            print(f"❌ {k}: Failed")
    print(f"\n⏱️  TOTAL TIME: {total_time:.2f}s")
    print(f"{'='*60}")
    print("🎉 Done!")


if __name__ == "__main__":
    main()
