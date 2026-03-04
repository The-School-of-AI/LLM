"""
S3 utilities for BinIdx dataloader prefetching.

Provides transparent S3 → local prefetching so the training loop never waits
on a network download. A bounded background thread keeps at most
`prefetch_count` shards on local disk at any given time, which tightly caps
memory/disk overhead while eliminating download stalls during training.

Public API
----------
is_s3_path(path)                    bool – is this an S3 URI?
parse_s3_uri(uri)                   (bucket, key_prefix)
list_s3_shard_uris(uri, rank, ws)   rank-sharded list of shard URIs  ← used by _build_shard_list
download_shard_files(uri, local_dir) download one shard's files
S3ShardPrefetcher                   context-manager / iterator over pre-downloaded local dirs

Overhead profile (worst case)
------------------------------
- Disk:   prefetch_count × shard_size  (default 2 shards ahead)
- Memory: negligible – tensors stay on disk until _iter_sequences_from_shard reads them
- CPU:    1 daemon thread per DataLoader worker (idle 99% of the time)
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Sentinel: signals the consumer that the prefetch thread finished.
_SENTINEL = object()

# Required files per shard; metadata.json is optional (legacy shards may omit it).
_REQUIRED_FILES = ("tokens.bin", "tokens.idx")
_OPTIONAL_FILES = ("metadata.json",)


# ---------------------------------------------------------------------------
# URI helpers
# ---------------------------------------------------------------------------

def is_s3_path(path: str) -> bool:
    """Return True if *path* looks like an S3 URI (``s3://…``)."""
    return isinstance(path, str) and path.startswith("s3://")


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """
    Split ``s3://bucket/key/prefix`` into ``(bucket, key_prefix)``.

    The key prefix has no leading or trailing slash.
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {uri!r}")
    rest = uri[len("s3://"):]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.strip("/")


# ---------------------------------------------------------------------------
# Shard listing
# ---------------------------------------------------------------------------

def list_s3_shard_uris(
    s3_uri: str,
    rank: int,
    world_size: int,
) -> Tuple[List[str], int]:
    """
    List shard sub-directory URIs at *s3_uri* and return those assigned to
    *rank* via round-robin.

    Expected S3 layout (mirrors local on-disk layout):
        s3://bucket/prefix/shard_001/tokens.bin
        s3://bucket/prefix/shard_001/tokens.idx
        s3://bucket/prefix/shard_001/metadata.json   (optional)
        s3://bucket/prefix/shard_002/…

    Returns
    -------
    (rank_shard_uris, total_shards)
        rank_shard_uris – list of ``s3://bucket/prefix/shard_NNN`` URIs for
                          this rank; sorted for determinism.
        total_shards    – total shard count across all ranks.
    """
    try:
        import boto3
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for S3 support. Install it with:\n"
            "  pip install boto3"
        ) from exc

    bucket, prefix = parse_s3_uri(s3_uri)
    s3_client = boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")

    # Use Delimiter="/" to get virtual directory entries (CommonPrefixes).
    list_prefix = f"{prefix}/" if prefix else ""
    shard_dirs: List[str] = []

    for page in paginator.paginate(
        Bucket=bucket, Prefix=list_prefix, Delimiter="/"
    ):
        for cp in page.get("CommonPrefixes", []):
            # cp["Prefix"] ends with "/", e.g. "datasets/myrun/shard_001/"
            key = cp["Prefix"].rstrip("/")
            shard_dirs.append(f"s3://{bucket}/{key}")

    shard_dirs.sort()
    total_shards = len(shard_dirs)

    if not shard_dirs:
        raise FileNotFoundError(
            f"No shard subdirectories found at {s3_uri}.\n"
            "Expected layout: s3://bucket/prefix/<shard_name>/tokens.bin + tokens.idx\n"
            "Verify the S3 URI and that the tokenizer team has uploaded the shards."
        )

    return shard_dirs[rank::world_size], total_shards


# ---------------------------------------------------------------------------
# Single-shard downloader
# ---------------------------------------------------------------------------

def download_shard_files(
    s3_shard_uri: str,
    local_dir: str,
    extra_files: Optional[List[str]] = None,
) -> str:
    """
    Download ``tokens.bin``, ``tokens.idx``, and ``metadata.json`` (if present)
    from *s3_shard_uri* into *local_dir*.

    Parameters
    ----------
    s3_shard_uri:
        S3 URI of the shard sub-directory, e.g. ``s3://bucket/prefix/shard_001``.
    local_dir:
        Local directory to download files into. Created if it does not exist.
    extra_files:
        Additional file names to download (optional).

    Returns
    -------
    str – *local_dir* (convenience for callers that want to chain calls).
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for S3 support. Install it with:\n"
            "  pip install boto3"
        ) from exc

    bucket, key_prefix = parse_s3_uri(s3_shard_uri)
    s3_client = boto3.client("s3")
    os.makedirs(local_dir, exist_ok=True)

    files_to_fetch = (
        list(_REQUIRED_FILES) + list(_OPTIONAL_FILES) + (extra_files or [])
    )

    for fname in files_to_fetch:
        s3_key = f"{key_prefix}/{fname}"
        local_path = os.path.join(local_dir, fname)
        is_optional = fname in _OPTIONAL_FILES

        try:
            s3_client.download_file(bucket, s3_key, local_path)
            logger.debug("Downloaded s3://%s/%s → %s", bucket, s3_key, local_path)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey") and is_optional:
                logger.debug(
                    "Optional file s3://%s/%s not found — skipped.", bucket, s3_key
                )
                continue
            raise FileNotFoundError(
                f"Required S3 file not found: s3://{bucket}/{s3_key}\n"
                f"Check that the shard was fully uploaded."
            ) from exc

    return local_dir


def download_metadata_only(s3_shard_uri: str, local_dir: str) -> str:
    """
    Download only ``metadata.json`` from *s3_shard_uri* into *local_dir*.

    Used during dataset initialisation for lightweight tokenizer validation
    without downloading the full shard binary.

    Returns *local_dir*; the file may be absent (legacy shard) — no error raised.
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for S3 support. Install it with:\n"
            "  pip install boto3"
        ) from exc

    bucket, key_prefix = parse_s3_uri(s3_shard_uri)
    s3_key = f"{key_prefix}/metadata.json"
    local_path = os.path.join(local_dir, "metadata.json")
    s3_client = boto3.client("s3")
    os.makedirs(local_dir, exist_ok=True)

    try:
        s3_client.download_file(bucket, s3_key, local_path)
        logger.debug("Downloaded metadata s3://%s/%s → %s", bucket, s3_key, local_path)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey"):
            logger.debug(
                "metadata.json not found at s3://%s/%s (legacy shard).", bucket, s3_key
            )
        else:
            raise

    return local_dir


# ---------------------------------------------------------------------------
# Background prefetcher
# ---------------------------------------------------------------------------

class S3ShardPrefetcher:
    """
    Background-thread S3 shard prefetcher.

    Downloads shards from S3 to a local temp directory ahead of consumption,
    keeping at most ``prefetch_count`` shards on disk simultaneously.

    Designed as a context manager / iterator:

    .. code-block:: python

        with S3ShardPrefetcher(shard_uris, cache_dir, prefetch_count=2) as pf:
            for local_dir in pf:
                # tokens.bin + tokens.idx (+ metadata.json) are ready here
                process_shard(local_dir)
                pf.release(local_dir)   # delete after use (saves disk space)

    Overhead
    --------
    - Spawns **one daemon thread** (no extra processes, no GPU interaction).
    - Holds at most ``prefetch_count`` shard directories on disk at once.
    - Queue is bounded — the background thread blocks automatically when the
      consumer is slower than the download bandwith, preventing run-away disk
      usage.

    Parameters
    ----------
    shard_uris:
        Ordered list of S3 shard URIs to download.
    cache_dir:
        Local directory under which per-shard temp dirs are created.
        Defaults to the system temp directory.
    prefetch_count:
        How many shards to pre-download ahead of the consumer.
        2–4 is recommended; larger values waste disk for small gain.
    delete_after_use:
        If True (default), ``release()`` removes the local shard directory
        after the consumer is done. Set False to keep a persistent disk cache.
    """

    def __init__(
        self,
        shard_uris: List[str],
        cache_dir: Optional[str] = None,
        prefetch_count: int = 2,
        delete_after_use: bool = True,
    ) -> None:
        self._shard_uris = list(shard_uris)
        self._cache_dir = cache_dir or tempfile.gettempdir()
        self._prefetch_count = max(1, prefetch_count)
        self._delete_after_use = delete_after_use

        # Bounded queue: limits simultaneous on-disk shards to prefetch_count.
        self._queue: "queue.Queue[object]" = queue.Queue(
            maxsize=self._prefetch_count
        )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[Exception] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background download thread."""
        self._stop_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._download_loop,
            name="S3ShardPrefetcher",
            daemon=True,
        )
        self._thread.start()
        logger.debug(
            "S3ShardPrefetcher started: %d shards, prefetch_count=%d, cache=%s",
            len(self._shard_uris),
            self._prefetch_count,
            self._cache_dir,
        )

    def stop(self) -> None:
        """
        Signal the download thread to stop and drain the queue.

        Orphaned shard directories in the queue are cleaned up if
        ``delete_after_use=True``.
        """
        self._stop_event.set()

        # Drain so the background thread is not blocked on queue.put().
        while True:
            try:
                item = self._queue.get_nowait()
                if item is not _SENTINEL and self._delete_after_use:
                    self._cleanup(str(item))
            except queue.Empty:
                break

        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

    def __enter__(self) -> "S3ShardPrefetcher":
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def release(self, local_dir: str) -> None:
        """
        Mark *local_dir* as consumed.  Deletes the directory if
        ``delete_after_use=True`` (the default).  Call this after you have
        finished reading a shard to reclaim disk space immediately rather
        than waiting until the prefetcher is stopped.
        """
        if self._delete_after_use:
            self._cleanup(local_dir)

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over pre-downloaded local shard directories.

        Blocks (briefly) only if the consumer outruns the prefetch thread,
        which should rarely happen — downloading a shard takes far longer
        than iterating through it.
        """
        while True:
            item = self._queue.get()   # blocks until a shard is ready
            if item is _SENTINEL:
                if self._error is not None:
                    raise RuntimeError(
                        f"S3 prefetch thread failed: {self._error}"
                    ) from self._error
                return
            yield str(item)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cleanup(self, local_dir: str) -> None:
        try:
            shutil.rmtree(local_dir, ignore_errors=True)
            logger.debug("Removed shard cache dir: %s", local_dir)
        except Exception as exc:
            logger.warning("Failed to remove shard cache dir %s: %s", local_dir, exc)

    def _download_loop(self) -> None:
        """Runs in the background daemon thread."""
        for uri in self._shard_uris:
            if self._stop_event.is_set():
                logger.debug("S3ShardPrefetcher: stop requested, exiting download loop.")
                break

            try:
                local_dir = tempfile.mkdtemp(
                    prefix="shard_cache_", dir=self._cache_dir
                )
                download_shard_files(uri, local_dir)
                logger.debug("Prefetched %s → %s", uri, local_dir)
            except Exception as exc:
                logger.error(
                    "S3ShardPrefetcher: failed to download %s: %s", uri, exc
                )
                self._error = exc
                self._queue.put(_SENTINEL)
                return

            # Block here if the consumer hasn't caught up (bounded disk use).
            while not self._stop_event.is_set():
                try:
                    self._queue.put(local_dir, timeout=0.5)
                    break
                except queue.Full:
                    continue

        # Signal end of iteration.
        self._queue.put(_SENTINEL)
        logger.debug("S3ShardPrefetcher: download loop complete.")
