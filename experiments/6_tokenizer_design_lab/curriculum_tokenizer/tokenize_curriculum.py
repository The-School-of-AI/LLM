#!/usr/bin/env python3
"""
S3 Curriculum Tokenization Pipeline
===================================

Tokenizes specific subsets of data defined by one or more "coreset" parquet files.
Each coreset file acts as a manifest, containing `source_url` and `chunk_id` (or equivalent).

Output Structure:
  s3://<dst-bucket>/<dst-prefix>/<coreset_filename>/shard_NNN/tokens.bin + tokens.idx

Features:
  - Multi-File Support: Processes a single coreset file OR a folder of parquet files.
  - Per-File Isolation: Each coreset file gets its own output subfolder (e.g. `batch001/`).
  - Manifest-Driven: Only includes specific chunks listed in the coreset.
  - Standard Output: 512MB shards with spdl-compatible binary .idx.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys
import tempfile
import time
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Tuple

import boto3
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOKEN_DTYPE = np.uint32
BYTES_PER_TOKEN = TOKEN_DTYPE().itemsize  # 4

# Binary index format (spdl-compatible):
# 8-byte header: version(uint32=1) + dtype_size(uint32=4)
# Then (N+1) uint64 byte offsets for N blocks.
SPDL_IDX_HEADER_FMT = "<II"  # version + dtype_size = 8 bytes


# ---------------------------------------------------------------------------
# S3 Helpers
# ---------------------------------------------------------------------------

def parse_s3_url(url: str) -> Tuple[str, str]:
    """Parse s3://bucket/key -> (bucket, key)."""
    parsed = urlparse(url)
    if parsed.scheme != "s3":
        raise ValueError(f"Invalid S3 URL: {url}")
    return parsed.netloc, parsed.path.lstrip("/")


def s3_download_to_temp(s3, bucket: str, key: str, tmp_dir: str) -> str:
    """Download an S3 object to a local temp file, return local path."""
    basename = os.path.basename(key)
    local_path = os.path.join(tmp_dir, basename)
    s3.download_file(bucket, key, local_path)
    return local_path


def s3_upload_file(s3, local_path: str, bucket: str, key: str) -> None:
    """Upload a local file to S3."""
    s3.upload_file(local_path, bucket, key)


def s3_key_exists(s3, bucket: str, key: str) -> bool:
    """Check if an S3 key exists."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def s3_list_parquet_files(s3, bucket: str, prefix: str) -> List[str]:
    """List all .parquet files under a prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                files.append(f"s3://{bucket}/{key}")
    return files


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def get_tokenizer(tokenizer_path: Optional[str] = None):
    """Load the tokenizer."""
    if tokenizer_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
        tokenizer_path = os.path.join(project_root, "tsai_131k_tokenizer")

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at: {tokenizer_path}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def tokenize_function(
    examples: Dict[str, List[str]],
    tokenizer: Any,
) -> Dict[str, List[List[int]]]:
    """Tokenize text examples — no truncation, no padding."""
    tokenized = tokenizer(
        examples["text"],
        truncation=False,
        padding=False,
        max_length=None,
        return_tensors=None,
    )
    tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
    return tokenized


# ---------------------------------------------------------------------------
# Shard Flusher
# ---------------------------------------------------------------------------

class ShardWriter:
    """Accumulates packed blocks and flushes 512 MB shards."""

    def __init__(
        self,
        s3_client,
        dst_bucket: str,
        domain_prefix: str,
        block_size: int,
        shard_size_mb: int,
        tmp_dir: str,
        vocab_size: int,
        pad_token_id: int,
        eos_token_id: int,
    ):
        self.s3 = s3_client
        self.dst_bucket = dst_bucket
        self.domain_prefix = domain_prefix.rstrip("/")
        self.block_size = block_size
        self.tmp_dir = tmp_dir
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        # Calculate blocks per shard
        shard_bytes = shard_size_mb * 1024 * 1024
        tokens_per_shard = shard_bytes // BYTES_PER_TOKEN
        self.blocks_per_shard = tokens_per_shard // block_size

        # State
        self.shard_idx = 0
        self.accumulated_blocks: List[List[int]] = []
        self.shard_stats: List[dict] = []

    def add_block(self, block: List[int]) -> None:
        """Add a packed block. Auto-flushes when shard is full."""
        self.accumulated_blocks.append(block)
        if len(self.accumulated_blocks) >= self.blocks_per_shard:
            self.flush_shard()

    def flush_shard(self) -> Optional[dict]:
        """Write accumulated blocks as a shard to S3."""
        if not self.accumulated_blocks:
            return None

        shard_name = f"shard_{self.shard_idx:03d}"
        s3_prefix = f"{self.domain_prefix}/{shard_name}"
        num_blocks = len(self.accumulated_blocks)

        # Skip if shard already exists (resumability)
        meta_key = f"{s3_prefix}/metadata.json"
        if s3_key_exists(self.s3, self.dst_bucket, meta_key):
            print(f"    [SKIP] {shard_name} already exists in S3")
            self.shard_idx += 1
            self.accumulated_blocks = []
            return None

        # Write .bin to temp
        bin_path = os.path.join(self.tmp_dir, "tokens.bin")
        idx_path = os.path.join(self.tmp_dir, "tokens.idx")
        meta_path = os.path.join(self.tmp_dir, "metadata.json")

        byte_offsets = []
        current_byte_offset = 0

        with open(bin_path, "wb") as f:
            for block in self.accumulated_blocks:
                byte_offsets.append(current_byte_offset)
                arr = np.array(block, dtype=TOKEN_DTYPE)
                raw_bytes = arr.tobytes()
                f.write(raw_bytes)
                current_byte_offset += len(raw_bytes)

        # Final boundary
        byte_offsets.append(current_byte_offset)

        # Write spdl-compatible binary .idx
        with open(idx_path, "wb") as f:
            f.write(struct.pack(SPDL_IDX_HEADER_FMT, 1, BYTES_PER_TOKEN))
            offset_arr = np.array(byte_offsets, dtype=np.uint64)
            f.write(offset_arr.tobytes())

        # Write metadata
        total_tokens = current_byte_offset // BYTES_PER_TOKEN
        file_size = os.path.getsize(bin_path)
        metadata = {
            "format": "megatron_bin_idx",
            "idx_format": "spdl_v1",
            "token_dtype": TOKEN_DTYPE.__name__,
            "bytes_per_token": BYTES_PER_TOKEN,
            "block_size": self.block_size,
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "num_blocks": num_blocks,
            "total_tokens": total_tokens,
            "file_size_bytes": file_size,
            "shard_name": shard_name,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Upload to S3
        s3_upload_file(self.s3, bin_path, self.dst_bucket, f"{s3_prefix}/tokens.bin")
        s3_upload_file(self.s3, idx_path, self.dst_bucket, f"{s3_prefix}/tokens.idx")
        s3_upload_file(self.s3, meta_path, self.dst_bucket, f"{s3_prefix}/metadata.json")

        # Cleanup temp files
        for p in (bin_path, idx_path, meta_path):
            if os.path.exists(p):
                os.remove(p)

        stats = {
            "shard_name": shard_name,
            "num_blocks": num_blocks,
            "total_tokens": total_tokens,
            "file_size_bytes": file_size,
        }
        self.shard_stats.append(stats)

        print(f"    [UPLOADED] {shard_name}: "
              f"{num_blocks:,} blocks, "
              f"{file_size / 1024 / 1024:.1f} MB")

        self.shard_idx += 1
        self.accumulated_blocks = []
        return stats

    def finalize(self) -> List[dict]:
        """Flush any remaining blocks and return all shard stats."""
        self.flush_shard()
        return self.shard_stats


# ---------------------------------------------------------------------------
# Curriculum Processor
# ---------------------------------------------------------------------------

def process_coreset_file(
    s3_client,
    coreset_uri: str,
    dst_bucket: str,
    dst_prefix: str,
    tokenizer: Any,
    args: argparse.Namespace,
    tmp_dir: str,
) -> dict:
    """Process a single coreset parquet file."""
    
    filename = os.path.basename(urlparse(coreset_uri).path)
    # Output folder = filename without extension (e.g. "batch001")
    coreset_name = os.path.splitext(filename)[0]
    
    print(f"\nProcessing Coreset: {filename}")
    print(f"Output folder: s3://{dst_bucket}/{dst_prefix}/{coreset_name}/")

    # Download Coreset
    bucket, key = parse_s3_url(coreset_uri)
    local_path = s3_download_to_temp(s3_client, bucket, key, tmp_dir)
    
    try:
        df = pd.read_parquet(local_path)
    except Exception as e:
        print(f"ERROR: Failed to read parquet {filename}: {e}")
        return {}
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)

    # Verify columns
    required_cols = [args.url_col, args.coreset_id_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns in {filename}: {missing}")
        print(f"Available: {df.columns.tolist()}")
        return {}

    # Initialize Writer for this coreset file
    writer = ShardWriter(
        s3_client=s3_client,
        dst_bucket=dst_bucket,
        domain_prefix=f"{dst_prefix.rstrip('/')}/{coreset_name}",
        block_size=args.block_size,
        shard_size_mb=args.shard_size_mb,
        tmp_dir=tmp_dir,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        eos_token_id=tokenizer.eos_token_id,
    )

    buffer: List[int] = []
    total_docs = 0
    t0_start = time.perf_counter()

    # Group by source URL to batch S3 downloads
    grouped_sources = df.groupby(args.url_col)
    
    print(f"  Unique source files to fetch: {len(grouped_sources)}")
    
    for src_url, group in grouped_sources:
        try:
            src_bucket, src_key = parse_s3_url(src_url)
        except ValueError:
            print(f"  [WARN] Skipping invalid source URL: {src_url}")
            continue

        valid_ids = set(group[args.coreset_id_col].unique())
        print(f"  Fetching {os.path.basename(src_key)} ({len(valid_ids)} target chunks)... ", end="", flush=True)

        # Download source parquet
        local_src = s3_download_to_temp(s3_client, src_bucket, src_key, tmp_dir)
        t0 = time.perf_counter()

        try:
            # 1. Load Parquet
            src_df = pd.read_parquet(local_src)
            
            # 2. Filter by ID
            if args.src_id_col not in src_df.columns:
                print(f"SKIP (missing ID col '{args.src_id_col}')")
                continue
            
            # Keep rows where ID is in our valid_ids set
            df_filtered = src_df[src_df[args.src_id_col].isin(valid_ids)]
            
            if df_filtered.empty:
                print(f"SKIP (0 matches)")
                continue

            # 3. Tokenize
            raw_dataset = Dataset.from_pandas(df_filtered)
            if args.text_col not in raw_dataset.column_names:
                print(f"SKIP (missing text col '{args.text_col}')")
                continue

            tokenized = raw_dataset.map(
                lambda x: tokenize_function(x, tokenizer),
                batched=True,
                num_proc=min(args.num_proc, 4),
                remove_columns=raw_dataset.column_names,
            )
            
            # 4. Pack into buffer
            doc_count = 0
            for example in tokenized:
                ids = example.get("input_ids")
                if not ids: continue
                doc_count += 1
                
                buffer.extend(ids)
                if writer.eos_token_id is not None:
                    buffer.append(writer.eos_token_id)
                
                while len(buffer) >= args.block_size:
                    writer.add_block(buffer[:args.block_size])
                    del buffer[:args.block_size]

            total_docs += doc_count
            elapsed = time.perf_counter() - t0
            print(f"{doc_count} docs [{elapsed:.1f}s]")

        except Exception as e:
            print(f"ERROR: {e}")
        finally:
            if os.path.exists(local_src):
                os.remove(local_src)

    if buffer:
        if not getattr(args, "drop_remainder", False):
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            pad_len = args.block_size - len(buffer)
            print(f"  [Info] Padded final block with {pad_len} PAD tokens (preserved {len(buffer)} tokens).")
            buffer.extend([pad_token_id] * pad_len)
            writer.add_block(buffer)
        else:
            eos_token_id = tokenizer.eos_token_id
            dropped_tokens = len(buffer)
            dropped_finished_rows = buffer.count(eos_token_id) if eos_token_id is not None else 0
            partial_row = 1 if (eos_token_id is None or buffer[-1] != eos_token_id) else 0
            dropped_rows = dropped_finished_rows + partial_row
            print(f"  [Warning] Dropped remainder intentionally: {dropped_tokens} tokens (approx {dropped_rows} rows dropped/truncated).")

    # Flush remaining blocks
    shard_stats = writer.finalize()
    total_time = time.perf_counter() - t0_start
    
    return {
        "coreset_file": filename,
        "coreset_name": coreset_name,
        "num_source_files": len(grouped_sources),
        "num_docs": total_docs,
        "num_shards": len(shard_stats),
        "total_tokens": sum(s["total_tokens"] for s in shard_stats),
        "total_size_bytes": sum(s["file_size_bytes"] for s in shard_stats),
        "elapsed_seconds": round(total_time, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="S3 Curriculum Tokenizer")
    
    # Paths
    p.add_argument("--coreset-uri", required=True, help="S3 URI to coreset parquet file OR folder prefix")
    p.add_argument("--dst-bucket", required=True, help="Destination bucket")
    p.add_argument("--dst-prefix", required=True, help="Destination prefix")
    p.add_argument("--tokenizer-path", default=None, help="Tokenizer path")
    
    # Column mappings
    p.add_argument("--url-col", default="source_url", help="Column for source parquet S3 URL")
    p.add_argument("--coreset-id-col", default="chunk_id", help="Column for ID in coreset")
    p.add_argument("--src-id-col", default="id", help="Column for ID in source parquet")
    p.add_argument("--text-col", default="text", help="Column containing text in source parquet")

    # Config
    p.add_argument("--block-size", type=int, default=4096)
    p.add_argument("--shard-size-mb", type=int, default=512)
    p.add_argument("--num-proc", type=int, default=8)
    p.add_argument("--drop-remainder", action="store_true",
                   help="Drop tail blocks shorter than --block-size instead of padding.")
    p.add_argument("--tmp-dir", default=None)
    
    return p.parse_args()


def main():
    args = parse_args()
    
    # Setup
    s3 = boto3.client("s3")
    tmp_base = args.tmp_dir or tempfile.gettempdir()
    tmp_dir = os.path.join(tmp_base, "tokenize_curriculum_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    
    print("="*70)
    print("S3 Curriculum Tokenizer (Multi-File)")
    print(f"Coreset Input: {args.coreset_uri}")
    print(f"Output Base:   s3://{args.dst_bucket}/{args.dst_prefix}")
    print("="*70)

    # Load Tokenizer
    tokenizer = get_tokenizer(args.tokenizer_path)
    print(f"Vocab size: {len(tokenizer):,}")

    # Determine Input Files (Single File vs Directory)
    bucket, key = parse_s3_url(args.coreset_uri)
    target_files = []
    
    # Check if key ends in .parquet -> single file
    if key.endswith(".parquet"):
        target_files.append(args.coreset_uri)
    else:
        # Assume prefix -> list files
        print(f"Listing parquet files under s3://{bucket}/{key} ...")
        target_files = s3_list_parquet_files(s3, bucket, key)
    
    print(f"Found {len(target_files)} coreset files to process.")
    
    all_stats = []
    
    try:
        for idx, uri in enumerate(target_files):
            print(f"\n--- File {idx+1}/{len(target_files)} ---")
            stats = process_coreset_file(
                s3, uri, args.dst_bucket, args.dst_prefix,
                tokenizer, args, tmp_dir
            )
            if stats:
                all_stats.append(stats)
            
        # Global Manifest (optional summary)
        manifest = {
            "format": "curriculum_megatron_bin_idx",
            "idx_format": "spdl_v1",
            "block_size": args.block_size,
            "vocab_size": len(tokenizer),
            "input_uri": args.coreset_uri,
            "processed_files": all_stats,
            "total_tokens": sum(d.get("total_tokens", 0) for d in all_stats),
            "total_shards": sum(d.get("num_shards", 0) for d in all_stats),
            "timestamp": time.time(),
        }
        
        manifest_path = os.path.join(tmp_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
            
        manifest_key = os.path.join(args.dst_prefix, "manifest.json").replace("\\", "/")
        s3_upload_file(s3, manifest_path, args.dst_bucket, manifest_key)
        
        print(f"\nSaved global manifest to s3://{args.dst_bucket}/{manifest_key}")
        print("Done.")

    finally:
        # Cleanup
        try:
            os.rmdir(tmp_dir)
        except:
            pass

if __name__ == "__main__":
    main()
