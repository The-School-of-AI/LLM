#!/usr/bin/env python3
"""
S3 Curriculum Tokenization Pipeline
===================================

2-Level Architecture:
  T3 (coreset index) → T1 (raw text) — T2 is bypassed entirely.

  Each T3 coreset parquet file is a manifest containing:
    chunk_id      — unique document ID to extract
    t1_file_path  — relative path within T1 base URI to the source parquet
                    (e.g. "source=C4/part-00759-8299c866-....parquet")
    band          — curriculum band (B0 / B1 / B2)
    domain        — domain classification

  The full T1 URI is constructed as:
    args.t1_base_uri.rstrip("/") + "/" + t1_file_path

  Each T1 parquet contains: id (== chunk_id), text, and other columns.

Output Structure:
  <dst-uri>/shards/shard_NNN/tokens.bin + tokens.idx + metadata.json

  Shards are numbered globally across all coreset batches (sequential mode).
  Each shard's metadata.json carries source_file for full traceability.

Features:
  - Multi-File Support: Processes a single coreset file OR a folder of parquet files.
  - Per-File Isolation: Each coreset file gets its own output subfolder.
  - Manifest-Driven: Only includes specific chunks listed in the coreset.
  - Standard Output: 512 MB shards with spdl-compatible binary .idx.
  - Rich Metadata: Per-shard metadata.json includes tokenizer_hash, band, domain,
    stage, row counts, drop stats, and created_at timestamp.
  - File-Level Parallelism: Optional --file-parallelism for concurrent batch processing.
  - Spot Interrupt Handling: IMDS polling + SIGTERM handler for graceful EC2 Spot shutdown.
  - Progress State: progress_state.json tracks completed batch files across interrupts.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing
import os
import shutil
import signal
import struct
import tempfile
import threading
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from datasets import Dataset
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
# Spot Interrupt Handling (module-level, safe to import)
# ---------------------------------------------------------------------------

_TERMINATION_DETECTED = threading.Event()


def _handle_sigterm(signum, frame):
    """Signal handler for SIGTERM and SIGINT — triggers graceful shutdown."""
    print(f"\n[SIGNAL] Received signal {signum}. Initiating graceful shutdown...")
    _TERMINATION_DETECTED.set()


def _poll_spot_termination():
    """Daemon thread: polls EC2 IMDS every 5s for Spot termination notice."""
    imds_url = "http://169.254.169.254/latest/meta-data/spot/termination-time"
    while not _TERMINATION_DETECTED.is_set():
        try:
            req = urllib.request.Request(imds_url)
            req.add_header("X-aws-ec2-metadata-token-ttl-seconds", "21600")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    termination_time = resp.read().decode().strip()
                    print(f"\n[SPOT] Termination scheduled at: {termination_time}")
                    _TERMINATION_DETECTED.set()
                    return
        except Exception:
            pass  # IMDS not reachable (local run) or returns 404 (no notice yet)
        time.sleep(5)


def register_interrupt_handlers():
    """Register SIGTERM/SIGINT handlers and start IMDS polling thread."""
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)
    t = threading.Thread(target=_poll_spot_termination, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# S3 / Local File Helpers
# ---------------------------------------------------------------------------


def is_s3_uri(uri: str) -> bool:
    return uri.startswith("s3://")


def parse_s3_url(url: str) -> Tuple[str, str]:
    """Parse s3://bucket/key -> (bucket, key)."""
    parsed = urlparse(url)
    if parsed.scheme != "s3":
        raise ValueError(f"Invalid S3 URL: {url}")
    return parsed.netloc, parsed.path.lstrip("/")


def download_to_temp(s3, uri: str, tmp_dir: str) -> str:
    """Download an S3 object to a local temp file, or return local path if already local."""
    if is_s3_uri(uri):
        if s3 is None:
            import boto3
            s3 = boto3.client("s3")
        bucket, key = parse_s3_url(uri)
        basename = os.path.basename(key)
        local_path = os.path.join(tmp_dir, basename)
        s3.download_file(bucket, key, local_path)
        return local_path
    return uri


def upload_file(s3, local_path: str, dst_uri: str) -> None:
    """Upload a local file to S3 or copy to local destination."""
    if is_s3_uri(dst_uri):
        if s3 is None:
            import boto3
            s3 = boto3.client("s3")
        bucket, key = parse_s3_url(dst_uri)
        s3.upload_file(local_path, bucket, key)
    else:
        os.makedirs(os.path.dirname(dst_uri), exist_ok=True)
        shutil.copy2(local_path, dst_uri)


def key_exists(s3, uri: str) -> bool:
    """Check if an S3 key or local file/dir exists."""
    if is_s3_uri(uri):
        if s3 is None:
            import boto3
            s3 = boto3.client("s3")
        from botocore.exceptions import ClientError
        bucket, key = parse_s3_url(uri)
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise
    return os.path.exists(uri)


def list_parquet_files(s3, uri: str) -> List[str]:
    """List all .parquet files under an S3 prefix or a local folder."""
    files = []
    if is_s3_uri(uri):
        if s3 is None:
            import boto3
            s3 = boto3.client("s3")
        bucket, prefix = parse_s3_url(uri)
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".parquet"):
                    files.append(f"s3://{bucket}/{key}")
    else:
        if os.path.isfile(uri) and uri.endswith(".parquet"):
            files.append(uri)
        elif os.path.isdir(uri):
            for root, _, filenames in os.walk(uri):
                for filename in filenames:
                    if filename.endswith(".parquet"):
                        files.append(os.path.join(root, filename).replace("\\", "/"))
    return files


def upload_json(s3, data: dict, dst_uri: str, tmp_dir: str) -> None:
    """Write a dict as JSON and upload to S3 or local path."""
    tmp_path = os.path.join(tmp_dir, "_tmp_upload.json")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    upload_file(s3, tmp_path, dst_uri)
    os.remove(tmp_path)


def download_json(s3, uri: str, tmp_dir: str) -> Optional[dict]:
    """Download and parse a JSON file from S3 or local path. Returns None if not found."""
    if not key_exists(s3, uri):
        return None
    local = download_to_temp(s3, uri, tmp_dir)
    with open(local, "r", encoding="utf-8") as f:
        data = json.load(f)
    if is_s3_uri(uri) and os.path.exists(local):
        os.remove(local)
    return data


# ---------------------------------------------------------------------------
# Progress State (cross-interrupt resume)
# ---------------------------------------------------------------------------


def load_progress_state(s3, dst_uri: str, tmp_dir: str) -> dict:
    """Load progress_state.json from dst_uri. Returns empty state if not found."""
    uri = f"{dst_uri.rstrip('/')}/progress_state.json"
    state = download_json(s3, uri, tmp_dir)
    if state is None:
        return {"completed": [], "failed": []}
    return state


def save_progress_state(s3, dst_uri: str, state: dict, tmp_dir: str) -> None:
    """Atomically write progress_state.json to dst_uri."""
    uri = f"{dst_uri.rstrip('/')}/progress_state.json"
    state["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    upload_json(s3, state, uri, tmp_dir)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def get_tokenizer(tokenizer_path: Optional[str] = None):
    """Load the tokenizer."""
    if tokenizer_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tokenizer_path = os.path.join(script_dir, "tsai_131k_tokenizer")

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at: {tokenizer_path}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def compute_tokenizer_hash(tokenizer_dir: str) -> str:
    """Compute a stable SHA256 hash over tokenizer.json + special_tokens_map.json.

    Filenames are prepended to the hash input to ensure determinism even if file
    contents happen to be identical. Files are processed in sorted order.
    Matches the algorithm specified in TOKENIZER_TEAM_RECOMMENDATIONS.md §2.
    """
    files = sorted(["special_tokens_map.json", "tokenizer.json"])
    h = hashlib.sha256()
    for fname in files:
        fpath = os.path.join(tokenizer_dir, fname)
        with open(fpath, "rb") as f:
            h.update(fname.encode())
            h.update(f.read())
    return h.hexdigest()


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
    """Accumulates packed blocks and flushes 512 MB shards with rich metadata."""

    def __init__(
        self,
        s3_client,
        dst_uri: str,
        block_size: int,
        shard_size_mb: int,
        tmp_dir: str,
        vocab_size: int,
        pad_token_id: int,
        eos_token_id: int,
        tokenizer_hash: str = "",
        tokenizer_version: str = "v1",
        stage: int = 1,
        source_file: str = "",
        start_shard_idx: int = 1,
    ):
        self.s3 = s3_client
        self.dst_uri = dst_uri.rstrip("/")
        self.block_size = block_size
        self.tmp_dir = tmp_dir
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.tokenizer_hash = tokenizer_hash
        self.tokenizer_version = tokenizer_version
        self.stage = stage
        self.source_file = source_file

        # Calculate blocks per shard
        shard_bytes = shard_size_mb * 1024 * 1024
        tokens_per_shard = shard_bytes // BYTES_PER_TOKEN
        self.blocks_per_shard = tokens_per_shard // block_size

        # State
        self.shard_idx = start_shard_idx
        self.accumulated_blocks: List[List[int]] = []
        self.shard_stats: List[dict] = []

        # Per-shard row tracking (reset after each flush)
        self.shard_rows_input: int = 0
        self.shard_rows_with_eos: int = 0

        # Pending tail-handling stats (set by caller before finalize)
        self._pending_rows_dropped: int = 0
        self._pending_tokens_dropped: int = 0
        self._pending_drop_reason: str = ""

        # Accumulated band/domain distributions across current shard
        self._band_counts: Dict[str, int] = {}
        self._domain_counts: Dict[str, int] = {}

    def update_distributions(self, band_counts: Dict[str, int], domain_counts: Dict[str, int]) -> None:
        """Merge per-source-file band/domain counts into the running shard totals."""
        for k, v in band_counts.items():
            self._band_counts[k] = self._band_counts.get(k, 0) + v
        for k, v in domain_counts.items():
            self._domain_counts[k] = self._domain_counts.get(k, 0) + v

    def add_block(self, block: List[int]) -> None:
        """Add a packed block. Auto-flushes when shard is full."""
        self.accumulated_blocks.append(block)
        if len(self.accumulated_blocks) >= self.blocks_per_shard:
            self.flush_shard()

    def flush_shard(self) -> Optional[dict]:
        """Write accumulated blocks as a shard."""
        if not self.accumulated_blocks:
            return None

        shard_name = f"shard_{self.shard_idx:03d}"
        target_prefix = f"{self.dst_uri}/{shard_name}"
        num_blocks = len(self.accumulated_blocks)

        # Skip if shard already exists (resumability)
        meta_key = f"{target_prefix}/metadata.json"
        if key_exists(self.s3, meta_key):
            print(f"    [SKIP] {shard_name} already exists")
            self.shard_idx += 1
            self.accumulated_blocks = []
            self._reset_shard_counters()
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

        # Compute band/domain metadata
        total_band_tokens = sum(self._band_counts.values()) or 1
        band_distribution = {k: round(v / total_band_tokens, 4) for k, v in self._band_counts.items()}
        dominant_band = max(self._band_counts, key=self._band_counts.get) if self._band_counts else ""

        total_domain_tokens = sum(self._domain_counts.values()) or 1
        domain_distribution = {k: round(v / total_domain_tokens, 4) for k, v in self._domain_counts.items()}
        dominant_domain = max(self._domain_counts, key=self._domain_counts.get) if self._domain_counts else ""

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
            # Tokenizer identity
            "tokenizer_hash": self.tokenizer_hash,
            "tokenizer_version": self.tokenizer_version,
            # Curriculum metadata
            "band": dominant_band,
            "band_distribution": band_distribution,
            "domain": dominant_domain,
            "domain_distribution": domain_distribution,
            "stage": self.stage,
            "source_file": self.source_file,
            # Audit trail
            "rows_input": self.shard_rows_input,
            "rows_with_eos": self.shard_rows_input - self._pending_rows_dropped,
            "rows_dropped": self._pending_rows_dropped,
            "tokens_dropped": self._pending_tokens_dropped,
            "drop_reason": self._pending_drop_reason,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Upload to S3 or copy locally
        upload_file(self.s3, bin_path, f"{target_prefix}/tokens.bin")
        upload_file(self.s3, idx_path, f"{target_prefix}/tokens.idx")
        upload_file(self.s3, meta_path, f"{target_prefix}/metadata.json")

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

        print(
            f"    [UPLOADED] {shard_name}: "
            f"{num_blocks:,} blocks, "
            f"{file_size / 1024 / 1024:.1f} MB"
        )

        self.shard_idx += 1
        self.accumulated_blocks = []
        self._reset_shard_counters()
        return stats

    def _reset_shard_counters(self):
        self.shard_rows_input = 0
        self.shard_rows_with_eos = 0
        self._pending_rows_dropped = 0
        self._pending_tokens_dropped = 0
        self._pending_drop_reason = ""
        self._band_counts = {}
        self._domain_counts = {}

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
    dst_base_uri: str,
    tokenizer: Any,
    args: argparse.Namespace,
    tmp_dir: str,
    tokenizer_hash: str = "",
    start_shard_idx: int = 1,
    staging_mode: bool = False,
) -> dict:
    """Process a single coreset parquet file.

    Sequential mode (staging_mode=False):
      Shards are written directly into <dst_base_uri>/shards/ using global
      numbering that starts at start_shard_idx.  The returned dict includes
      next_shard_idx so the caller can thread the counter across batches.

    Staging mode (staging_mode=True):
      Shards are written into <dst_base_uri>/<coreset_name>/shard_NNN/ so
      each parallel worker has an isolated namespace with no name collisions.
      start_shard_idx is always 1 in this mode.
    """

    filename = (
        os.path.basename(urlparse(coreset_uri).path)
        if is_s3_uri(coreset_uri)
        else os.path.basename(coreset_uri)
    )
    coreset_name = os.path.splitext(filename)[0]

    if staging_mode:
        # Per-batch isolated dir — parallel workers can't collide
        shards_uri = f"{dst_base_uri.rstrip('/')}/{coreset_name}"
        print(f"\nProcessing Coreset: {filename}")
        print(f"Output folder (staging): {shards_uri}/")
    else:
        # Flat global layout — sequential mode with threaded counter
        shards_uri = f"{dst_base_uri.rstrip('/')}/shards"
        print(f"\nProcessing Coreset: {filename}")
        print(f"Output folder: {shards_uri}/ (start index: shard_{start_shard_idx:03d})")

    # Download Coreset (or use locally)
    local_path = download_to_temp(s3_client, coreset_uri, tmp_dir)

    try:
        df = pd.read_parquet(local_path)
    except Exception as e:
        print(f"ERROR: Failed to read parquet {filename}: {e}")
        return {}
    finally:
        if is_s3_uri(coreset_uri) and os.path.exists(local_path):
            os.remove(local_path)

    # Verify required columns (hardcoded — fixed by T3 schema)
    required_cols = ["t1_file_path", "chunk_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns in {filename}: {missing}")
        print(f"Available: {df.columns.tolist()}")
        return {}

    # 2-level architecture: group T3 rows by t1_file_path (fixed by T3 schema)
    group_col = "t1_file_path"

    # Initialize Writer for this coreset file
    writer = ShardWriter(
        s3_client=s3_client,
        dst_uri=shards_uri,
        block_size=args.block_size,
        shard_size_mb=args.shard_size_mb,
        tmp_dir=tmp_dir,
        vocab_size=len(tokenizer),
        pad_token_id=(
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        ),
        eos_token_id=tokenizer.eos_token_id,
        tokenizer_hash=tokenizer_hash,
        tokenizer_version=getattr(args, "tokenizer_version", "v1"),
        stage=getattr(args, "stage", 1),
        source_file=coreset_uri,
        start_shard_idx=start_shard_idx,
    )

    buffer: List[int] = []
    total_docs = 0
    t0_start = time.perf_counter()

    grouped_sources = df.groupby(group_col)
    print(f"  Unique source files to fetch: {len(grouped_sources)}")

    # Column names are hardcoded — fixed by T3/T1 schema (not CLI args)
    band_col = "band"
    domain_col = "domain"

    for t1_file_path_val, group in grouped_sources:
        # Check for Spot termination / SIGTERM before starting each T1 download
        if _TERMINATION_DETECTED.is_set():
            print(f"  [INTERRUPT] Stopping before next T1 file due to termination signal.")
            break

        # Construct full T1 URI from base + relative path stored in T3
        t1_uri = args.t1_base_uri.rstrip("/") + "/" + t1_file_path_val
        basename = os.path.basename(t1_file_path_val)

        valid_ids = set(group["chunk_id"].unique())

        # Collect band/domain distributions for this T1 group
        src_band_counts: Dict[str, int] = {}
        src_domain_counts: Dict[str, int] = {}
        if band_col in group.columns:
            src_band_counts = group[band_col].value_counts().to_dict()
        if domain_col in group.columns:
            src_domain_counts = group[domain_col].value_counts().to_dict()

        print(
            f"  Fetching {basename} ({len(valid_ids)} target chunks)... ",
            end="",
            flush=True,
        )

        # Download T1 parquet (direct lookup — no T2 involved)
        local_src = download_to_temp(s3_client, t1_uri, tmp_dir)
        t0 = time.perf_counter()

        try:
            # 1. Load T1 Parquet
            src_df = pd.read_parquet(local_src)

            # 2. Filter by chunk_id (T1.id == T3.chunk_id)
            if "id" not in src_df.columns:
                print("SKIP (missing ID col 'id')")
                continue

            df_filtered = src_df[src_df["id"].isin(valid_ids)]

            if df_filtered.empty:
                print("SKIP (0 matches)")
                continue

            # 3. Tokenize
            raw_dataset = Dataset.from_pandas(df_filtered)
            if "text" not in raw_dataset.column_names:
                print("SKIP (missing text col 'text')")
                continue

            tokenized = raw_dataset.map(
                lambda x: tokenize_function(x, tokenizer),
                batched=True,
                num_proc=min(args.num_proc, 4),
                remove_columns=raw_dataset.column_names,
            )

            # Merge band/domain BEFORE processing rows: a flush inside add_block
            # resets _band_counts, so registering here ensures the flushed shard
            # captures this group's band/domain even if a flush fires mid-drain.
            writer.update_distributions(src_band_counts, src_domain_counts)

            # 4. Pack into buffer — EOS appended manually after each document
            doc_count = 0
            for example in tokenized:
                ids = example.get("input_ids")
                if not ids:
                    continue
                doc_count += 1
                writer.shard_rows_input += 1

                buffer.extend(ids)
                if writer.eos_token_id is not None:
                    buffer.append(writer.eos_token_id)
                    writer.shard_rows_with_eos += 1

                while len(buffer) >= args.block_size:
                    prev_shard_idx = writer.shard_idx
                    writer.add_block(buffer[: args.block_size])
                    del buffer[: args.block_size]
                    # If an auto-flush fired (shard_idx advanced), the new shard
                    # starts with empty band/domain — re-seed it with the current
                    # group's distribution so mid-group shards are never empty.
                    if writer.shard_idx != prev_shard_idx:
                        writer.update_distributions(src_band_counts, src_domain_counts)

            total_docs += doc_count
            elapsed = time.perf_counter() - t0
            print(f"{doc_count} docs [{elapsed:.1f}s]")

        except Exception as e:
            print(f"ERROR: {e}")
        finally:
            if is_s3_uri(t1_uri) and os.path.exists(local_src):
                os.remove(local_src)

    # Handle tail buffer (only if no interruption)
    if buffer and not _TERMINATION_DETECTED.is_set():
        if not getattr(args, "drop_remainder", False):
            pad_token_id = (
                tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            )
            pad_len = args.block_size - len(buffer)
            print(
                f"  [Info] Padded final block with {pad_len} PAD tokens (preserved {len(buffer)} tokens)."
            )
            writer._pending_rows_dropped = 0
            writer._pending_tokens_dropped = pad_len
            writer._pending_drop_reason = "padded"
            buffer.extend([pad_token_id] * pad_len)
            writer.add_block(buffer)
        else:
            eos_token_id = tokenizer.eos_token_id
            dropped_tokens = len(buffer)
            dropped_finished_rows = (
                buffer.count(eos_token_id) if eos_token_id is not None else 0
            )
            partial_row = (
                1 if (eos_token_id is None or buffer[-1] != eos_token_id) else 0
            )
            dropped_rows = dropped_finished_rows + partial_row
            writer._pending_rows_dropped = dropped_rows
            writer._pending_tokens_dropped = dropped_tokens
            writer._pending_drop_reason = "tail_truncation_at_block_boundary"
            print(
                f"  [Warning] Dropped remainder: {dropped_tokens} tokens "
                f"(approx {dropped_rows} rows dropped/truncated)."
            )

    batch_tokens_dropped = writer._pending_tokens_dropped

    if _TERMINATION_DETECTED.is_set():
        # Discard partial accumulated blocks — only fully uploaded shards are valid
        if writer.accumulated_blocks:
            print(
                f"  [INTERRUPT] Discarding {len(writer.accumulated_blocks)} partially "
                f"accumulated blocks for {coreset_name}."
            )
            writer.accumulated_blocks = []
        shard_stats = writer.shard_stats
    else:
        shard_stats = writer.finalize()

    total_time = time.perf_counter() - t0_start

    num_shards_written = len(shard_stats)
    return {
        "coreset_file": filename,
        "coreset_name": coreset_name,
        "num_source_files": len(grouped_sources),
        "num_docs": total_docs,
        "num_shards": num_shards_written,
        "shard_start": start_shard_idx,
        "shard_end": start_shard_idx + num_shards_written - 1,
        "next_shard_idx": start_shard_idx + num_shards_written,
        "total_tokens": sum(s["total_tokens"] for s in shard_stats),
        "total_size_bytes": sum(s["file_size_bytes"] for s in shard_stats),
        "tokens_dropped": batch_tokens_dropped,
        "elapsed_seconds": round(total_time, 1),
    }


# ---------------------------------------------------------------------------
# Parallel Worker (used by multiprocessing.Pool)
# ---------------------------------------------------------------------------


def _worker_process_coreset(worker_args: tuple) -> dict:
    """Pool worker: creates its own S3 client and tokenizer after fork/spawn.

    Each worker writes to its own isolated staging dir:
      <dst>/_staging/<coreset_name>/shard_NNN/

    This gives every worker a collision-free namespace regardless of how many
    shards other workers produce.  The main process calls run_flatten() after
    pool.map() completes to move all staging shards into the final flat
    <dst>/shards/ layout with global continuous numbering.
    """
    uri, dst_base_uri, tokenizer_path, args_dict, worker_id, tokenizer_hash = worker_args

    worker_tmp = os.path.join(args_dict.get("tmp_dir") or tempfile.gettempdir(),
                              "tokenize_curriculum_tmp", f"worker_{worker_id:03d}")
    os.makedirs(worker_tmp, exist_ok=True)

    # Boto3 client must be created AFTER fork (not safe to share across processes)
    s3 = None
    if is_s3_uri(uri) or is_s3_uri(dst_base_uri):
        try:
            import boto3
            s3 = boto3.client("s3")
        except Exception as e:
            print(f"[Worker {worker_id}] S3 setup warn: {e}")

    # Tokenizer must be loaded AFTER fork
    tokenizer = get_tokenizer(tokenizer_path)

    # Workers write to <dst>/_staging/<coreset_name>/shard_NNN/ (staging_mode=True)
    # The staging base is one level up from dst_base_uri so the coreset_name subdir
    # lands inside _staging/, not directly in dst/.
    staging_base_uri = f"{dst_base_uri.rstrip('/')}/_staging"

    args = argparse.Namespace(**args_dict)
    try:
        result = process_coreset_file(
            s3, uri, staging_base_uri, tokenizer, args, worker_tmp, tokenizer_hash,
            start_shard_idx=1,
            staging_mode=True,
        )
    except Exception as e:
        print(f"[Worker {worker_id}] ERROR processing {uri}: {e}")
        result = {}
    finally:
        shutil.rmtree(worker_tmp, ignore_errors=True)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="S3 Curriculum Tokenizer")

    # Paths
    p.add_argument(
        "--coreset-uri",
        required=True,
        help="S3 URI or local path to T3 coreset parquet file OR folder prefix",
    )
    p.add_argument(
        "--dst-uri", required=True, help="Destination S3 URI or local prefix"
    )
    p.add_argument("--tokenizer-path", default=None, help="Tokenizer path")
    p.add_argument(
        "--t1-base-uri",
        default="s3://t1-dataacquisition-datasets/processed_dataset/normalized_data",
        help="Base URI for T1 raw-text parquets. Full T1 path = t1_base_uri + '/' + t1_file_path "
             "(from T3 column). Override to a local folder for testing.",
    )

    # Config
    p.add_argument("--block-size", type=int, default=4096)
    p.add_argument("--shard-size-mb", type=float, default=512)
    p.add_argument("--num-proc", type=int, default=8,
                   help="Max HF tokenizer subprocesses per worker (capped at 4)")
    p.add_argument(
        "--drop-remainder",
        action="store_true",
        help="Drop tail blocks shorter than --block-size instead of padding.",
    )
    p.add_argument("--tmp-dir", default=None)

    # Curriculum metadata
    p.add_argument("--stage", type=int, default=1,
                   help="Training stage number (written to shard metadata)")
    p.add_argument("--tokenizer-version", default="v1",
                   help="Tokenizer version string (written to shard metadata)")

    # Parallelism
    p.add_argument("--file-parallelism", type=int, default=1,
                   help="Number of coreset batch files to process in parallel (default: 1=sequential)")

    return p.parse_args()


def main():
    args = parse_args()

    # Register Spot/SIGTERM interrupt handlers
    register_interrupt_handlers()

    # Setup S3 client (used for manifest + progress state; workers create their own)
    s3 = None
    if is_s3_uri(args.coreset_uri) or args.dst_uri.startswith("s3://"):
        try:
            import boto3
            s3 = boto3.client("s3")
        except Exception as e:
            print(f"S3 setup warn: {e}")

    tmp_base = args.tmp_dir or tempfile.gettempdir()
    tmp_dir = os.path.join(tmp_base, "tokenize_curriculum_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    print("=" * 70)
    print("Curriculum Tokenizer (Multi-File)")
    print(f"Coreset Input:    {args.coreset_uri}")
    print(f"Output Base:      {args.dst_uri}")
    print(f"File Parallelism: {args.file_parallelism}")
    print(f"Stage:            {args.stage}")
    print(f"Tokenizer:        {args.tokenizer_path or '(default)'}")
    print("=" * 70)

    # Load Tokenizer (main process — used for hash + sequential fallback)
    tokenizer = get_tokenizer(args.tokenizer_path)
    print(f"Vocab size: {len(tokenizer):,}")

    # Compute tokenizer hash once
    tokenizer_dir = args.tokenizer_path
    if tokenizer_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tokenizer_dir = os.path.join(script_dir, "tsai_131k_tokenizer")
    tokenizer_hash = compute_tokenizer_hash(tokenizer_dir)
    print(f"Tokenizer hash:   {tokenizer_hash[:16]}...")

    # Determine Input Files
    if args.coreset_uri.endswith(".parquet"):
        target_files = [args.coreset_uri]
    else:
        print(f"Listing parquet files under {args.coreset_uri} ...")
        target_files = list_parquet_files(s3, args.coreset_uri)

    print(f"Found {len(target_files)} coreset files to process.")

    # Load progress state (cross-interrupt resume)
    progress = load_progress_state(s3, args.dst_uri, tmp_dir)
    completed_set = set(progress.get("completed", []))
    # next_shard_idx is persisted so resume picks up the global counter correctly
    next_shard_idx = progress.get("next_shard_idx", 1)
    pending_files = [uri for uri in target_files if uri not in completed_set]
    if len(completed_set) > 0:
        print(f"Resuming: {len(completed_set)} already complete, {len(pending_files)} remaining.")
        print(f"Resuming global shard counter at: shard_{next_shard_idx:03d}")

    all_stats = []

    try:
        if args.file_parallelism <= 1:
            # Sequential path — global shard counter threads across batches
            for idx, uri in enumerate(pending_files):
                if _TERMINATION_DETECTED.is_set():
                    print("\n[INTERRUPT] Stopping file loop.")
                    break
                print(f"\n--- File {idx+1}/{len(pending_files)} ---")
                stats = process_coreset_file(
                    s3, uri, args.dst_uri, tokenizer, args, tmp_dir, tokenizer_hash,
                    start_shard_idx=next_shard_idx,
                )
                if stats:
                    all_stats.append(stats)
                    next_shard_idx = stats["next_shard_idx"]
                    completed_set.add(uri)
                    progress["completed"] = list(completed_set)
                    progress["next_shard_idx"] = next_shard_idx
                    save_progress_state(s3, args.dst_uri, progress, tmp_dir)
        else:
            # Parallel path — workers write to <dst>/_staging/<coreset_name>/shard_NNN/
            # (each worker has an isolated namespace, no shard name collisions).
            # After pool.map() completes, run_flatten() moves everything into the
            # final <dst>/shards/ layout with global continuous numbering.
            from flatten_shards import run_flatten

            staging_uri = f"{args.dst_uri.rstrip('/')}/_staging"

            args_dict = vars(args)
            args_dict["tmp_dir"] = tmp_dir  # share base tmp dir; workers create subdirs

            worker_inputs = [
                (uri, args.dst_uri, args.tokenizer_path, args_dict, idx, tokenizer_hash)
                for idx, uri in enumerate(pending_files)
            ]

            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=args.file_parallelism) as pool:
                results = pool.map(_worker_process_coreset, worker_inputs)

            for uri, stats in zip(pending_files, results):
                if stats:
                    all_stats.append(stats)
                    completed_set.add(uri)

            progress["completed"] = list(completed_set)
            save_progress_state(s3, args.dst_uri, progress, tmp_dir)

            # Flatten staging shards into the final flat <dst>/shards/ layout
            if not _TERMINATION_DETECTED.is_set():
                print("\n[FLATTEN] Moving staging shards to flat global layout...")
                flatten_result = run_flatten(
                    src=staging_uri,
                    dst=args.dst_uri,
                    s3_client=s3,
                )
                if flatten_result["failed"] == 0:
                    # Clean up staging dir — all shards successfully moved
                    print("[FLATTEN] Cleaning up staging directory...")
                    if is_s3_uri(staging_uri):
                        # S3: staging files were already deleted streaming during flatten
                        pass
                    else:
                        shutil.rmtree(staging_uri, ignore_errors=True)
                else:
                    print(
                        f"[FLATTEN] WARNING: {flatten_result['failed']} shard(s) failed to move. "
                        f"Staging directory preserved at: {staging_uri}"
                    )

        if not _TERMINATION_DETECTED.is_set():
            # Global Manifest
            manifest = {
                "format": "curriculum_megatron_bin_idx",
                "idx_format": "spdl_v1",
                "block_size": args.block_size,
                "vocab_size": len(tokenizer),
                "tokenizer_hash": tokenizer_hash,
                "tokenizer_version": args.tokenizer_version,
                "stage": args.stage,
                "input_uri": args.coreset_uri,
                "processed_files": all_stats,
                "total_tokens": sum(d.get("total_tokens", 0) for d in all_stats),
                "total_tokens_dropped": sum(d.get("tokens_dropped", 0) for d in all_stats),
                "total_shards": sum(d.get("num_shards", 0) for d in all_stats),
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

            manifest_uri = f"{args.dst_uri.rstrip('/')}/manifest.json"
            upload_json(s3, manifest, manifest_uri, tmp_dir)
            print(f"\nSaved global manifest to {manifest_uri}")
            print("Done.")
        else:
            print("\n[INTERRUPT] Run was interrupted. Resume by re-running with the same arguments.")
            print(f"  Completed: {len(completed_set)}/{len(target_files)} files")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
