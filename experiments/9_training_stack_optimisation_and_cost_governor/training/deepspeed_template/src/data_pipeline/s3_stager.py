"""
S3 Stager for downloading tokenized data shards from S3.

This module provides deterministic, resumable shard staging:
- Phase 1 (blocking): Pre-stage initial shards before training starts
- Phase 2 (background): Continuously download upcoming shards during training

Design Principles:
- Deterministic shard ordering: shards are always sorted by key name
- Rank 0 downloads; other ranks wait on distributed barrier
- Integrity verification via content-length check
- Retry with exponential backoff on transient failures
"""

import hashlib
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import torch.distributed as dist
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3Stager:
    """
    Downloads pre-tokenized data shards from S3 to local instance store.

    Provides two-phase staging:
    1. **Initial staging** (blocking): downloads the first N shards so
       training can begin immediately after model init.
    2. **Background staging** (non-blocking): a thread pool continuously
       downloads upcoming shards, staying ahead of training consumption.

    All shard keys are sorted deterministically so that every restart
    sees the same ordering, enabling exact checkpoint/resume.

    Args:
        s3_bucket: S3 bucket name.
        s3_prefix: S3 key prefix for the tokenized data.
        local_data_dir: Local directory to save downloaded shards.
        s3_region: AWS region (default: us-east-1).
        download_workers: Number of parallel download threads (default: 8).
        max_retries: Max retry attempts per shard (default: 3).
        retry_backoff_base: Base for exponential backoff in seconds.
        shard_extension: Expected shard file extension (default: .npy).
    """

    def __init__(
        self,
        s3_bucket: str,
        s3_prefix: str,
        local_data_dir: str,
        s3_region: str = "us-east-1",
        download_workers: int = 8,
        max_retries: int = 3,
        retry_backoff_base: float = 2.0,
        shard_extension: str = ".npy",
    ):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix.rstrip("/")
        self.local_data_dir = local_data_dir
        self.s3_region = s3_region
        self.download_workers = download_workers
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self.shard_extension = shard_extension

        # Create local data directory
        os.makedirs(self.local_data_dir, exist_ok=True)

        # S3 client (created lazily or on init for rank 0)
        self._s3_client: Optional[boto3.client] = None

        # Background staging state
        self._background_executor: Optional[ThreadPoolExecutor] = None
        self._background_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Track download status: shard_key -> local_path
        self._download_status: Dict[str, str] = {}
        self._status_lock = threading.Lock()

        # Ready event per shard: when set, shard is fully downloaded
        self._shard_ready_events: Dict[str, threading.Event] = {}

    # ------------------------------------------------------------------
    # S3 Client
    # ------------------------------------------------------------------

    @property
    def s3_client(self):
        """Lazily initialize the S3 client."""
        if self._s3_client is None:
            boto_config = BotoConfig(
                region_name=self.s3_region,
                max_pool_connections=self.download_workers + 4,
                retries={"max_attempts": self.max_retries, "mode": "adaptive"},
            )
            self._s3_client = boto3.client("s3", config=boto_config)
        return self._s3_client

    # ------------------------------------------------------------------
    # Shard Discovery
    # ------------------------------------------------------------------

    def discover_shards(self) -> List[str]:
        """
        List all shard keys in the S3 prefix, in deterministic sorted order.

        Returns:
            Sorted list of S3 keys (e.g., 'dolmo-tokenized/shard-00000.npy').
        """
        logger.info(
            "Discovering shards in s3://%s/%s/ ...",
            self.s3_bucket,
            self.s3_prefix,
        )

        shard_keys: List[str] = []
        paginator = self.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=self.s3_bucket,
            Prefix=f"{self.s3_prefix}/",
        )

        for page in pages:
            if "Contents" not in page:
                continue
            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith(self.shard_extension):
                    shard_keys.append(key)

        # CRITICAL: deterministic sort for reproducible ordering
        shard_keys.sort()

        logger.info(
            "Discovered %d shards in s3://%s/%s/",
            len(shard_keys),
            self.s3_bucket,
            self.s3_prefix,
        )

        return shard_keys

    # ------------------------------------------------------------------
    # Key → Local Path Mapping
    # ------------------------------------------------------------------

    def _s3_key_to_local_path(self, s3_key: str) -> str:
        """
        Derive the deterministic local file path from an S3 key.

        Args:
            s3_key: Full S3 key (e.g., 'dolmo-tokenized/shard-00042.npy').

        Returns:
            Absolute local path (e.g., '/data/dolmo/shard-00042.npy').
        """
        filename = os.path.basename(s3_key)
        return os.path.join(self.local_data_dir, filename)

    # ------------------------------------------------------------------
    # Single Shard Download
    # ------------------------------------------------------------------

    def _download_shard(self, s3_key: str) -> str:
        """
        Download a single shard from S3 with retry + integrity verification.

        If the shard already exists locally and passes the integrity check,
        the download is skipped.

        Args:
            s3_key: S3 key of the shard.

        Returns:
            Local path of the downloaded shard.

        Raises:
            RuntimeError: If download fails after all retries.
        """
        local_path = self._s3_key_to_local_path(s3_key)
        shard_name = os.path.basename(s3_key)

        # Check if already downloaded and valid
        if self._is_shard_valid(s3_key, local_path):
            logger.debug("Shard already staged: %s", shard_name)
            self._mark_shard_ready(s3_key, local_path)
            return local_path

        # Download with retries
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "Downloading shard: %s (attempt %d/%d)",
                    shard_name,
                    attempt,
                    self.max_retries,
                )

                # Download to a temp file first, then rename (atomic-ish)
                tmp_path = local_path + ".download"
                self.s3_client.download_file(
                    self.s3_bucket, s3_key, tmp_path
                )

                # Verify integrity (size check)
                if not self._verify_download(s3_key, tmp_path):
                    os.remove(tmp_path)
                    raise RuntimeError(
                        f"Integrity check failed for {shard_name}"
                    )

                # Atomic rename
                os.replace(tmp_path, local_path)
                logger.info("Downloaded shard: %s", shard_name)

                self._mark_shard_ready(s3_key, local_path)
                return local_path

            except Exception as e:
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"Failed to download shard '{shard_name}' "
                        f"after {self.max_retries} attempts: {e}"
                    ) from e

                sleep_time = self.retry_backoff_base ** (attempt - 1)
                logger.warning(
                    "Download attempt %d failed for %s: %s. "
                    "Retrying in %.1fs...",
                    attempt,
                    shard_name,
                    e,
                    sleep_time,
                )
                time.sleep(sleep_time)

        # Should never reach here, but satisfy type checker
        raise RuntimeError(f"Failed to download shard '{shard_name}'")

    def _is_shard_valid(self, s3_key: str, local_path: str) -> bool:
        """
        Check if a local shard file exists and matches the S3 object size.

        Args:
            s3_key: S3 key of the shard.
            local_path: Local path to check.

        Returns:
            True if shard exists and size matches S3.
        """
        if not os.path.isfile(local_path):
            return False

        try:
            response = self.s3_client.head_object(
                Bucket=self.s3_bucket, Key=s3_key
            )
            expected_size = response["ContentLength"]
            actual_size = os.path.getsize(local_path)
            return actual_size == expected_size
        except (ClientError, OSError):
            return False

    def _verify_download(self, s3_key: str, local_path: str) -> bool:
        """
        Verify a downloaded file matches the expected S3 object size.

        Args:
            s3_key: S3 key of the source object.
            local_path: Path to the downloaded file.

        Returns:
            True if file size matches the S3 object.
        """
        try:
            response = self.s3_client.head_object(
                Bucket=self.s3_bucket, Key=s3_key
            )
            expected_size = response["ContentLength"]
            actual_size = os.path.getsize(local_path)

            if actual_size != expected_size:
                logger.error(
                    "Size mismatch for %s: expected %d, got %d",
                    os.path.basename(s3_key),
                    expected_size,
                    actual_size,
                )
                return False

            return True
        except Exception as e:
            logger.error("Integrity verification failed: %s", e)
            return False

    def _mark_shard_ready(self, s3_key: str, local_path: str) -> None:
        """Mark a shard as ready (downloaded and verified)."""
        with self._status_lock:
            self._download_status[s3_key] = local_path

            # Signal waiting threads
            if s3_key not in self._shard_ready_events:
                self._shard_ready_events[s3_key] = threading.Event()
            self._shard_ready_events[s3_key].set()

    # ------------------------------------------------------------------
    # Phase 1: Initial Staging (blocking)
    # ------------------------------------------------------------------

    def stage_initial(
        self,
        shard_keys: List[str],
        start_shard_idx: int = 0,
        num_shards: int = 16,
    ) -> List[str]:
        """
        Download the first N shards starting from start_shard_idx. Blocking.

        This should run in parallel with model initialization so GPUs
        are not idle. Call this, then join the thread after deepspeed.init().

        Args:
            shard_keys: Full sorted list of all S3 shard keys.
            start_shard_idx: Index to start from (from checkpoint on resume).
            num_shards: Number of shards to pre-stage.

        Returns:
            List of local paths for the staged shards.
        """
        end_idx = min(start_shard_idx + num_shards, len(shard_keys))
        keys_to_stage = shard_keys[start_shard_idx:end_idx]

        logger.info(
            "Phase 1: Staging %d initial shards [%d → %d) ...",
            len(keys_to_stage),
            start_shard_idx,
            end_idx,
        )

        staged_paths: List[str] = []

        with ThreadPoolExecutor(max_workers=self.download_workers) as executor:
            futures = {
                executor.submit(self._download_shard, key): key
                for key in keys_to_stage
            }

            for future in as_completed(futures):
                key = futures[future]
                try:
                    local_path = future.result()
                    staged_paths.append(local_path)
                except Exception as e:
                    logger.error(
                        "Failed to stage shard %s: %s",
                        os.path.basename(key),
                        e,
                    )
                    raise

        # Return paths in deterministic sorted order
        staged_paths.sort()

        logger.info(
            "Phase 1 complete: %d shards staged to %s",
            len(staged_paths),
            self.local_data_dir,
        )

        return staged_paths

    def stage_initial_async(
        self,
        shard_keys: List[str],
        start_shard_idx: int = 0,
        num_shards: int = 16,
    ) -> threading.Thread:
        """
        Start initial staging in a background thread.

        This allows overlapping shard downloads with model initialization.
        Call `thread.join()` after `deepspeed.initialize()` to ensure
        initial shards are ready.

        Args:
            shard_keys: Full sorted list of all S3 shard keys.
            start_shard_idx: Index to start from (from checkpoint).
            num_shards: Number of shards to pre-stage.

        Returns:
            The background thread (call .join() to wait for completion).
        """
        thread = threading.Thread(
            target=self.stage_initial,
            args=(shard_keys, start_shard_idx, num_shards),
            daemon=True,
            name="S3-InitialStager",
        )
        thread.start()
        return thread

    # ------------------------------------------------------------------
    # Phase 2: Background Staging (non-blocking)
    # ------------------------------------------------------------------

    def stage_background(self, remaining_shard_keys: List[str]) -> None:
        """
        Begin background downloading of remaining shards during training.

        Downloads proceed continuously in the background using a thread pool.
        The training loop can call ``wait_for_shard()`` if it needs a
        specific shard that hasn't been downloaded yet.

        Args:
            remaining_shard_keys: S3 keys of shards not yet staged,
                in deterministic order.
        """
        if not remaining_shard_keys:
            logger.info("No remaining shards to stage in background.")
            return

        logger.info(
            "Phase 2: Starting background staging of %d remaining shards...",
            len(remaining_shard_keys),
        )

        self._stop_event.clear()

        def _background_worker():
            """Worker that downloads shards sequentially in the background."""
            with ThreadPoolExecutor(
                max_workers=self.download_workers
            ) as executor:
                futures = {}
                for key in remaining_shard_keys:
                    if self._stop_event.is_set():
                        break
                    futures[executor.submit(self._download_shard, key)] = key

                for future in as_completed(futures):
                    if self._stop_event.is_set():
                        break
                    key = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(
                            "Background staging failed for %s: %s",
                            os.path.basename(key),
                            e,
                        )

            logger.info("Background staging complete.")

        self._background_thread = threading.Thread(
            target=_background_worker,
            daemon=True,
            name="S3-BackgroundStager",
        )
        self._background_thread.start()

    def stop_background(self) -> None:
        """Signal the background stager to stop."""
        self._stop_event.set()
        if self._background_thread is not None:
            self._background_thread.join(timeout=30)
            logger.info("Background staging stopped.")

    # ------------------------------------------------------------------
    # Shard Readiness
    # ------------------------------------------------------------------

    def wait_for_shard(self, s3_key: str, timeout: float = 600.0) -> str:
        """
        Block until a specific shard is downloaded and ready.

        Args:
            s3_key: The S3 key of the shard to wait for.
            timeout: Maximum time to wait in seconds.

        Returns:
            Local path of the ready shard.

        Raises:
            TimeoutError: If the shard is not ready within the timeout.
            KeyError: If the shard was never queued for download.
        """
        # Initialize event if not already present
        with self._status_lock:
            if s3_key not in self._shard_ready_events:
                self._shard_ready_events[s3_key] = threading.Event()
            event = self._shard_ready_events[s3_key]

        if not event.wait(timeout=timeout):
            raise TimeoutError(
                f"Shard '{os.path.basename(s3_key)}' not ready "
                f"after {timeout}s. Background staging may be too slow."
            )

        with self._status_lock:
            return self._download_status[s3_key]

    def get_staged_shards(self) -> List[str]:
        """
        Get all currently staged (downloaded and verified) shard paths.

        Returns:
            Sorted list of local paths for all ready shards.
        """
        with self._status_lock:
            paths = sorted(self._download_status.values())
        return paths

    def is_shard_ready(self, s3_key: str) -> bool:
        """
        Check if a shard has been downloaded without blocking.

        Args:
            s3_key: The S3 key of the shard.

        Returns:
            True if the shard is ready on local disk.
        """
        with self._status_lock:
            return s3_key in self._download_status

    # ------------------------------------------------------------------
    # Distributed Coordination
    # ------------------------------------------------------------------

    @staticmethod
    def barrier_all_ranks() -> None:
        """
        Synchronize all ranks after staging.

        Only rank 0 downloads; other ranks call this barrier to wait
        until the data is available on the shared filesystem (or
        local NVMe if each node stages independently).
        """
        if dist.is_initialized():
            dist.barrier()
