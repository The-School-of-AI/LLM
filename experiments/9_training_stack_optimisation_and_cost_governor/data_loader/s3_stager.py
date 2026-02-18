"""
S3 → NVMe shard stager.

Downloads pre-tokenized ``.npy`` shards from S3 to local NVMe instance store.
Two-phase approach:
  Phase 1 (blocking)  — ``stage_initial`` downloads the first N shards.
  Phase 2 (background) — ``stage_background`` keeps a read-ahead window
                          ahead of training consumption.

Only **rank 0** downloads; other ranks wait on ``torch.distributed.barrier()``.
"""

import hashlib
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import boto3
from botocore.config import Config as BotoConfig

logger = logging.getLogger(__name__)


class S3Stager:
    """Stage shards from S3 to local NVMe storage.

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    prefix : str
        S3 key prefix where tokenized shards live.
    local_dir : str
        Local directory to download shards into.
    region : str
        AWS region for the S3 client.
    download_workers : int
        Number of concurrent download threads.
    max_retries : int
        Maximum number of retries for each shard download.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        local_dir: str,
        region: str = "us-east-1",
        download_workers: int = 8,
        max_retries: int = 3,
    ):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.local_dir = local_dir
        self.download_workers = download_workers
        self.max_retries = max_retries

        # Boto3 client with retry config
        boto_config = BotoConfig(
            region_name=region,
            retries={"max_attempts": max_retries, "mode": "adaptive"},
        )
        self._s3 = boto3.client("s3", config=boto_config)

        # Track which shards are ready
        self._ready_events: Dict[str, threading.Event] = {}
        self._staged_paths: List[str] = []
        self._lock = threading.Lock()

        # Background stager handle
        self._bg_thread: Optional[threading.Thread] = None
        self._bg_executor: Optional[ThreadPoolExecutor] = None

        os.makedirs(self.local_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Shard discovery
    # ------------------------------------------------------------------

    def discover_shards(self) -> List[str]:
        """List all ``.npy`` shard keys under the S3 prefix.

        Returns a **deterministic sorted** list of full S3 keys.
        """
        paginator = self._s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix + "/")

        keys: List[str] = []
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".npy"):
                    keys.append(key)

        keys.sort()
        logger.info("Discovered %d shards in s3://%s/%s/", len(keys), self.bucket, self.prefix)
        return keys

    # ------------------------------------------------------------------
    # Phase 1: Initial staging (blocking)
    # ------------------------------------------------------------------

    def stage_initial(self, shard_keys: List[str], start_idx: int, num_shards: int) -> List[str]:
        """Download the first *num_shards* starting from *start_idx* (blocking).

        Returns list of local paths that were staged.
        """
        to_download = shard_keys[start_idx : start_idx + num_shards]
        logger.info(
            "Initial staging: %d shards (idx %d → %d)",
            len(to_download), start_idx, start_idx + len(to_download) - 1,
        )

        staged: List[str] = []
        with ThreadPoolExecutor(max_workers=self.download_workers) as executor:
            futures = {
                executor.submit(self._download_shard, key): key
                for key in to_download
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    local_path = future.result()
                    staged.append(local_path)
                except Exception:
                    logger.exception("Failed to stage shard: %s", key)
                    raise

        staged.sort()
        with self._lock:
            self._staged_paths.extend(staged)
        logger.info("Initial staging complete: %d shards ready", len(staged))
        return staged

    def stage_initial_async(
        self, shard_keys: List[str], start_idx: int, num_shards: int
    ) -> threading.Thread:
        """Like :meth:`stage_initial` but runs in a background thread.

        Returns the ``Thread`` handle so the caller can ``.join()`` later
        (e.g. after DeepSpeed init has finished).
        """
        t = threading.Thread(
            target=self.stage_initial,
            args=(shard_keys, start_idx, num_shards),
            daemon=True,
            name="s3-initial-stager",
        )
        t.start()
        return t

    # ------------------------------------------------------------------
    # Phase 2: Background staging (non-blocking)
    # ------------------------------------------------------------------

    def stage_background(self, remaining_keys: List[str]) -> None:
        """Start a daemon thread that downloads *remaining_keys* in the background.

        Downloads are sequential in deterministic order but use a thread-pool
        for parallelism within a window.  The background thread is a daemon
        so it does not prevent process shutdown.
        """
        if not remaining_keys:
            logger.info("No remaining shards to stage in background")
            return

        def _bg_worker():
            logger.info("Background staging started: %d shards", len(remaining_keys))
            with ThreadPoolExecutor(max_workers=self.download_workers) as executor:
                futures = {
                    executor.submit(self._download_shard, key): key
                    for key in remaining_keys
                }
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        local_path = future.result()
                        with self._lock:
                            self._staged_paths.append(local_path)
                        # Signal anyone waiting on this shard
                        event = self._ready_events.get(local_path)
                        if event:
                            event.set()
                    except Exception:
                        logger.exception("Background stage failed: %s", key)
            logger.info("Background staging complete")

        self._bg_thread = threading.Thread(
            target=_bg_worker, daemon=True, name="s3-bg-stager"
        )
        self._bg_thread.start()

    # ------------------------------------------------------------------
    # Shard readiness
    # ------------------------------------------------------------------

    def get_staged_shards(self) -> List[str]:
        """Return sorted list of all locally staged shard paths."""
        with self._lock:
            return sorted(self._staged_paths)

    def wait_for_shard(self, shard_path: str, timeout: Optional[float] = None) -> None:
        """Block until *shard_path* has been downloaded.

        Parameters
        ----------
        shard_path : str
            Local path of the shard to wait for.
        timeout : float, optional
            Maximum seconds to wait. ``None`` means wait forever.

        Raises
        ------
        TimeoutError
            If *timeout* is exceeded.
        """
        # If already on disk, return immediately
        if os.path.exists(shard_path):
            return

        event = self._ready_events.setdefault(shard_path, threading.Event())
        if not event.wait(timeout=timeout):
            raise TimeoutError(f"Timed out waiting for shard: {shard_path}")

    # ------------------------------------------------------------------
    # Internal: single-shard download
    # ------------------------------------------------------------------

    def _download_shard(self, s3_key: str) -> str:
        """Download a single shard from S3 to local storage.

        Returns the local file path.
        """
        filename = os.path.basename(s3_key)
        local_path = os.path.join(self.local_dir, filename)

        # Skip if already downloaded (idempotent)
        if os.path.exists(local_path):
            logger.debug("Shard already staged: %s", local_path)
            return local_path

        tmp_path = local_path + ".tmp"
        logger.debug("Downloading s3://%s/%s → %s", self.bucket, s3_key, local_path)

        self._s3.download_file(self.bucket, s3_key, tmp_path)

        # Verify integrity via ETag (MD5 for single-part uploads)
        self._verify_integrity(s3_key, tmp_path)

        # Atomic rename
        os.replace(tmp_path, local_path)
        logger.debug("Staged shard: %s", local_path)

        # Signal waiters
        event = self._ready_events.get(local_path)
        if event:
            event.set()

        return local_path

    def _verify_integrity(self, s3_key: str, local_path: str) -> None:
        """Verify local file matches S3 ETag (MD5 for non-multipart uploads).

        For multipart uploads the ETag is not a simple MD5, so we skip
        verification in that case (indicated by a ``-`` in the ETag).
        """
        try:
            head = self._s3.head_object(Bucket=self.bucket, Key=s3_key)
            etag = head["ETag"].strip('"')

            # Multipart ETags contain a dash — skip verification
            if "-" in etag:
                return

            md5 = hashlib.md5()
            with open(local_path, "rb") as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    md5.update(chunk)

            local_md5 = md5.hexdigest()
            if local_md5 != etag:
                raise ValueError(
                    f"Integrity check failed for {s3_key}: "
                    f"local={local_md5}, S3 ETag={etag}"
                )
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            logger.warning("Could not verify integrity for %s: %s", s3_key, e)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def s3_key_to_local_path(self, s3_key: str) -> str:
        """Convert an S3 key to the expected local file path."""
        filename = os.path.basename(s3_key)
        return os.path.join(self.local_dir, filename)
