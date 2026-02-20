"""
Optimized batch processing utilities for large-scale coreset selection.
Handles 2 trillion+ token datasets with streaming and checkpointing.
"""

import json
import logging
import os
import pickle
import re
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import xxhash

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint"""

    stage_name: str
    batch_num: int
    chunks_processed: int
    tokens_processed: int
    selected_chunks: int
    timestamp: str
    config_hash: str


class BatchProcessor:
    """
    Process data in batches to avoid memory overload on 2T token datasets.
    Enables streaming, checkpointing, and resumption.
    """

    def __init__(self, batch_size: int = 10_000, checkpoint_dir: Optional[str] = None):
        """
        Args:
            batch_size: Chunks to process per batch (default 10k)
            checkpoint_dir: Dir for checkpoints; if None, no checkpointing
        """
        self.batch_size = batch_size
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def stream_chunks_from_jsonl(
        self,
        filepath: str,
        max_chunks: Optional[int] = None,
        *,
        shard_id: int = 0,
        num_shards: int = 1,
        shard_key: str = "chunk_id",
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """
        Stream chunks from JSONL without loading entire file into memory.

        Yields:
            (chunk_id, chunk_dict)
        """
        shard_id = int(shard_id)
        num_shards = int(num_shards)
        count = 0
        emitted = 0
        # Support both local filesystem and S3 (s3://bucket/key)
        if self._is_s3_path(filepath):
            line_iter: Iterable[str] = self._iter_s3_text_lines(filepath)
        else:

            def _local_iter() -> Iterator[str]:
                with open(filepath, "r", encoding="utf-8") as f:
                    yield from f

            line_iter = _local_iter()

        for line in line_iter:
            if max_chunks and emitted >= max_chunks:
                break
            try:
                data = json.loads(line)
                # Normalize the unique chunk identifier.
                # Some datasets use uid/guid/id instead of chunk_id.
                chunk_id = (
                    data.get("chunk_id")
                    or data.get("uid")
                    or data.get("guid")
                    or data.get("id")
                )

                # Optional row-level sharding (useful when input is a single huge file).
                if num_shards > 1:
                    key_val = data.get(shard_key)

                    # Treat empty/falsey ids as missing for sharding.
                    # This matters when the raw field exists (e.g., chunk_id="") but the
                    # real identifier is in uid/guid/id; without this, all such rows hash
                    # to the same shard and create severe imbalance.
                    if shard_key == "chunk_id" and not key_val:
                        key_val = chunk_id

                    if not key_val:
                        # Fallback: shard by line index so every row deterministically belongs
                        # to exactly one shard even if chunk_id is missing.
                        key_bytes = str(count).encode("utf-8")
                    else:
                        key_bytes = str(key_val).encode("utf-8")
                    h = xxhash.xxh64(key_bytes).intdigest()
                    if int(h % num_shards) != shard_id:
                        continue

                yield chunk_id, data
                emitted += 1
            except json.JSONDecodeError as e:
                logger.warning(f"Skipped malformed JSON line {count}: {e}")
            finally:
                # count is the physical line index (0-based) and must advance
                # even when a line is malformed or skipped due to sharding.
                count += 1

    def batch_iterator(
        self,
        filepath: str,
        max_chunks: Optional[int] = None,
        *,
        shard_id: int = 0,
        num_shards: int = 1,
        shard_key: str = "chunk_id",
    ) -> Iterator[List[Tuple[str, Dict[str, Any]]]]:
        """
        Iterate over chunks in batches from JSONL file.

        Yields:
            List of (chunk_id, chunk_dict) tuples
        """
        batch = []
        for chunk_id, data in self.stream_chunks_from_jsonl(
            filepath,
            max_chunks,
            shard_id=shard_id,
            num_shards=num_shards,
            shard_key=shard_key,
        ):
            batch.append((chunk_id, data))
            if len(batch) >= self.batch_size:
                yield batch
                batch = []

        if batch:  # Yield remaining
            yield batch

    def list_input_files(self, base_path: str, format: str) -> List[str]:
        """List input files for datasets.

        Supports:
        - Local filesystem paths (file or directory)
        - S3 prefixes and objects via s3://bucket/prefix
        """

        fmt = format.lower()

        # S3: accept either a single object (endswith .jsonl/.parquet) or a prefix.
        if self._is_s3_path(base_path):
            # If base_path looks like an object for the requested format, just return it.
            if base_path.lower().endswith(f".{fmt}"):
                return [base_path]

            suffix = f".{fmt}"
            return self._list_s3_objects(base_path, suffix=suffix)

        # Local filesystem
        root = Path(base_path)
        if root.is_file():
            return [str(root)]
        if not root.exists():
            return []
        if fmt == "jsonl":
            # Support both .jsonl and .json extensions strictly.
            all_files = list(root.glob("**/*.jsonl")) + list(root.glob("**/*.json"))
            # Filter out stats/ and _SUCCESS
            filtered = [
                f
                for f in all_files
                if "/stats/" not in f.as_posix()
                and not f.as_posix().startswith("stats/")
                and f.name != "_SUCCESS"
            ]
            return [str(p) for p in sorted(set(filtered))]

        if fmt == "parquet":
            all_files = list(root.glob("**/*.parquet"))
            filtered = [
                f
                for f in all_files
                if "/stats/" not in f.as_posix()
                and not f.as_posix().startswith("stats/")
                and f.name != "_SUCCESS"
            ]
            return [str(p) for p in sorted(set(filtered))]
        return []

    def shard_files(
        self, files: List[str], shard_id: int, num_shards: int
    ) -> List[str]:
        """Deterministically shard file identifiers across workers using xxhash of the path."""
        if num_shards <= 1:
            return files
        out: List[str] = []
        for p in files:
            h = xxhash.xxh64(str(p).encode("utf-8")).intdigest()
            if int(h % num_shards) == int(shard_id):
                out.append(str(p))
        return out

    # =========================
    # S3 helpers (streaming)
    # =========================

    _S3_URL_RE = re.compile(r"^s3://", re.IGNORECASE)

    def _is_s3_path(self, path: str) -> bool:
        return bool(path) and bool(self._S3_URL_RE.match(str(path)))

    def _parse_s3_url(self, url: str) -> Tuple[str, str]:
        """Parse s3://bucket/key into (bucket, key). Key may be empty for bucket root."""
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme.lower() != "s3":
            raise ValueError(f"Not an s3 url: {url}")
        bucket = parsed.netloc
        key = (parsed.path or "").lstrip("/")
        if not bucket:
            raise ValueError(f"Invalid s3 url (missing bucket): {url}")
        return bucket, key

    def _get_s3_client(self):
        try:
            import boto3
        except Exception as e:
            raise RuntimeError(
                "boto3 is required for S3 streaming input; install boto3 or avoid s3:// input-path"
            ) from e
        return boto3.client("s3")

    def _list_s3_objects(self, s3_prefix_url: str, *, suffix: str) -> List[str]:
        """List s3:// objects under the given prefix that end with suffix."""
        client = self._get_s3_client()
        bucket, key_prefix = self._parse_s3_url(s3_prefix_url)

        # Interpret the provided URL as a prefix. If it doesn't end with '/', still treat it
        # as a prefix (common when users pass s3://bucket/path without trailing slash).
        prefix = key_prefix

        suffix = suffix.lower()
        paginator = client.get_paginator("list_objects_v2")
        results: List[str] = []

        # Define allowed suffixes based on format
        allowed_suffixes = [suffix]
        if suffix == ".jsonl":
            allowed_suffixes.append(".json")

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                key = obj.get("Key")
                if not key:
                    continue

                # Filter out _SUCCESS and stats/ folder.
                # stats/ can be anywhere under the prefix.
                key_lower = key.lower()
                if (
                    key_lower.endswith("_success")
                    or "/stats/" in key_lower
                    or key_lower.startswith("stats/")
                ):
                    continue

                # Match if key ends with any allowed suffix
                if any(key_lower.endswith(s) for s in allowed_suffixes):
                    results.append(f"s3://{bucket}/{key}")

        return sorted(results)

    def _iter_s3_text_lines(self, s3_url: str) -> Iterator[str]:
        """Yield UTF-8 decoded lines from an S3 object without loading into memory."""
        client = self._get_s3_client()
        bucket, key = self._parse_s3_url(s3_url)

        obj = client.get_object(Bucket=bucket, Key=key)
        body = obj["Body"]

        # StreamingBody supports iter_lines() which is the safest way to stream huge objects.
        for bline in body.iter_lines():
            if not bline:
                yield ""
                continue
            try:
                yield bline.decode("utf-8")
            except Exception:
                # Be permissive on decode errors; replace invalid bytes.
                yield bline.decode("utf-8", errors="replace")

    def parquet_batch_iterator(
        self,
        path: str,
        batch_size_rows: int = 10_000,
        columns: Optional[List[str]] = None,
        max_rows: Optional[int] = None,
    ) -> Iterator[List[Dict[str, Any]]]:
        """Stream Parquet rows in batches using pyarrow.dataset.

        Notes:
        - Works for a single parquet file or a directory of parquet files.
        - If path is an s3:// URL and pyarrow S3 support is available, it will attempt to read it.
        """
        try:
            import pyarrow as pa
            import pyarrow.dataset as ds
        except Exception as e:
            raise RuntimeError(
                "pyarrow is required for parquet streaming; install pyarrow"
            ) from e

        dataset = ds.dataset(path, format="parquet")

        # If the caller asks for optional columns that are not present in a particular parquet
        # dataset, pyarrow will raise at scan time. Filter to existing schema names so
        # callers can request a superset of columns safely.
        scan_columns = None
        if columns is not None:
            try:
                available = set(getattr(dataset.schema, "names", []) or [])
            except Exception:
                available = set()
            if available:
                scan_columns = [c for c in columns if c in available]
            else:
                scan_columns = list(columns)

        # pyarrow.dataset API differs across versions:
        # - newer versions: Dataset.scan(...)
        # - others: Dataset.scanner(...)
        try:
            scanner = dataset.scan(
                columns=scan_columns, batch_size=int(batch_size_rows)
            )
        except AttributeError:
            scanner = dataset.scanner(
                columns=scan_columns, batch_size=int(batch_size_rows)
            )
        emitted = 0

        for record_batch in scanner.to_batches():
            table = pa.Table.from_batches([record_batch])
            rows = table.to_pylist()
            if not rows:
                continue
            if max_rows is not None:
                remaining = int(max_rows) - emitted
                if remaining <= 0:
                    break
                if len(rows) > remaining:
                    rows = rows[:remaining]

            yield rows
            emitted += len(rows)

    def save_checkpoint(
        self,
        stage_name: str,
        batch_num: int,
        state: Dict[str, Any],
        metadata: CheckpointMetadata,
    ) -> Path:
        """Save batch checkpoint for resumption."""
        if not self.checkpoint_dir:
            return None

        checkpoint_path = (
            self.checkpoint_dir / f"checkpoint_{stage_name}_batch_{batch_num:06d}.pkl"
        )

        with open(checkpoint_path, "wb") as f:
            pickle.dump({"state": state, "metadata": asdict(metadata)}, f)

        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Test hook: simulate a crash immediately after persisting a checkpoint.
        # This is intentionally gated behind environment variables so production
        # runs are unaffected.
        crash_stage = os.environ.get("CORESET_SIMULATE_CRASH_AFTER_CHECKPOINT_STAGE")
        crash_batch_raw = os.environ.get(
            "CORESET_SIMULATE_CRASH_AFTER_CHECKPOINT_BATCH"
        )
        if crash_batch_raw is not None:
            try:
                crash_batch = int(crash_batch_raw)
            except Exception:
                crash_batch = None
            if crash_batch is not None and int(batch_num) == crash_batch:
                if not crash_stage or str(crash_stage) == str(stage_name):
                    logger.error(
                        "Simulating hard crash after checkpoint (stage=%s, batch=%s)",
                        stage_name,
                        batch_num,
                    )
                    # Use SystemExit so it is not caught by `except Exception` blocks
                    # in upstream batch loops.
                    raise SystemExit(2)

        return checkpoint_path

    def load_checkpoint(
        self, stage_name: str, batch_num: int
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Load batch checkpoint if exists."""
        if not self.checkpoint_dir:
            return None

        checkpoint_path = (
            self.checkpoint_dir / f"checkpoint_{stage_name}_batch_{batch_num:06d}.pkl"
        )

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, "rb") as f:
                data = pickle.load(f)
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return data["state"], data["metadata"]
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def find_last_checkpoint(self, stage_name: str) -> Optional[int]:
        """Find the last completed batch checkpoint for a stage."""
        if not self.checkpoint_dir:
            return None

        checkpoints = sorted(
            self.checkpoint_dir.glob(f"checkpoint_{stage_name}_batch_*.pkl"),
            key=lambda p: int(p.stem.split("_")[-1]),
            reverse=True,
        )

        if checkpoints:
            batch_num = int(checkpoints[0].stem.split("_")[-1])
            logger.info(f"Found last checkpoint for {stage_name}: batch {batch_num}")
            return batch_num

        return None
