"""
Non-blocking Checkpoint Management with S3 Upload.

This module provides a universal checkpoint manager that:
- Saves DeepSpeed checkpoints locally
- Uploads checkpoints to S3 in the background (non-blocking)
- Works seamlessly with single-GPU, multi-GPU, and multi-node setups
- Automatically detects distributed training configuration
- Manages local checkpoint cleanup
- Provides progress tracking and error handling
"""

import multiprocessing
import os
import shutil
import threading
import time
from multiprocessing import Queue as MPQueue
from typing import Any, Dict, Optional

import boto3
import torch
import torch.distributed as dist
from botocore.exceptions import ClientError

from lightninglm.aws.config import S3Config


def _upload_process_worker(
    upload_queue: MPQueue,
    done_queue: MPQueue,
    config_dict: dict,
    node_rank: int,
    global_rank: int,
    num_nodes: int,
    gpus_per_node: int,
):
    """Standalone upload worker that runs in a separate process.

    Has its own GIL, its own boto3 client, and its own CPU scheduling.
    This prevents S3 upload threads from starving the training process.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config as BotoConfig

    # Reconstruct config in child process
    region = config_dict["region"]
    bucket_name = config_dict["bucket_name"]
    s3_prefix = config_dict["s3_prefix"]
    max_retries = config_dict["max_retries"]
    retry_backoff_base = config_dict["retry_backoff_base"]
    max_file_parallelism = config_dict["max_file_parallelism"]
    max_concurrency = config_dict["max_concurrency"]
    multipart_threshold = config_dict["multipart_threshold"]
    multipart_chunksize = config_dict["multipart_chunksize"]
    verbose = config_dict["verbose"]
    log_upload_progress = config_dict["log_upload_progress"]
    cleanup_after_upload = config_dict["cleanup_after_upload"]

    # Lower this process's scheduling priority so training gets CPU first
    try:
        os.nice(10)
    except OSError:
        pass

    # Ignore SIGUSR1/SIGUSR2 — these are for training workers, not the uploader.
    # Without this, run.sh's pgrep sends SIGUSR1 to us too (we inherit the
    # parent's cmdline after fork), which would invoke the default handler and
    # terminate this process.
    import signal

    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    signal.signal(signal.SIGUSR2, signal.SIG_IGN)

    # Create boto3 client in child process
    boto_config = BotoConfig(
        max_pool_connections=max_concurrency,
        retries={"max_attempts": max_retries, "mode": "adaptive"},
    )
    s3_client = boto3.client("s3", region_name=region, config=boto_config)

    transfer_config = TransferConfig(
        multipart_threshold=multipart_threshold,
        multipart_chunksize=multipart_chunksize,
        max_concurrency=max_concurrency,
        use_threads=True,
    )

    node_str = f"Node {node_rank}" if num_nodes > 1 else "Single-node"

    def _should_upload_file(filename: str) -> bool:
        if num_nodes == 1:
            return True
        if filename.startswith("mp_rank_"):
            try:
                parts = filename.split("_")
                if len(parts) >= 3:
                    file_rank = int(parts[2])
                    file_node = file_rank // gpus_per_node
                    return file_node == node_rank
            except (ValueError, IndexError):
                return node_rank == 0
        return node_rank == 0

    def _upload_one(local_path, s3_key):
        for attempt in range(max_retries):
            try:
                s3_client.upload_file(
                    local_path, bucket_name, s3_key, Config=transfer_config
                )
                return True
            except Exception as e:
                if attempt == max_retries - 1:
                    print(
                        f"[Rank {global_rank}] "
                        f"FAILED upload '{os.path.basename(local_path)}' "
                        f"after {max_retries} attempts: {e}"
                    )
                    return False
                sleep_time = retry_backoff_base**attempt
                if verbose:
                    print(
                        f"[Rank {global_rank}] "
                        f"Upload attempt {attempt + 1} failed for "
                        f"'{os.path.basename(local_path)}', "
                        f"retrying in {sleep_time}s..."
                    )
                time.sleep(sleep_time)
        return False

    while True:
        try:
            item = upload_queue.get()
            if item is None:
                break  # Poison pill — shutdown

            checkpoint_dir, tag, step, urgent = item

            if verbose:
                print(
                    f"[{node_str}, Rank {global_rank}] "
                    f"Starting upload: step {step} "
                    f"(parallel={max_file_parallelism})"
                )

            upload_start = time.time()

            # Collect files
            upload_tasks = []
            for root, dirs, files in os.walk(checkpoint_dir):
                for file in files:
                    if not _should_upload_file(file):
                        continue
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, checkpoint_dir)
                    if num_nodes > 1:
                        s3_key = f"{s3_prefix}/{tag}/node_{node_rank}/{relative_path}"
                    else:
                        s3_key = f"{s3_prefix}/{tag}/{relative_path}"
                    upload_tasks.append((local_path, s3_key))

            # Upload in parallel
            file_count = 0
            total_bytes = 0
            failed_files = []

            with ThreadPoolExecutor(max_workers=max_file_parallelism) as pool:
                futures = {
                    pool.submit(_upload_one, lp, sk): (lp, os.path.basename(lp))
                    for lp, sk in upload_tasks
                }
                for future in as_completed(futures):
                    local_path, filename = futures[future]
                    try:
                        if future.result():
                            file_count += 1
                            total_bytes += os.path.getsize(local_path)
                        else:
                            failed_files.append(filename)
                    except Exception as e:
                        failed_files.append(filename)
                        print(
                            f"[{node_str}, Rank {global_rank}] "
                            f"FAILED uploading {filename}: {e}"
                        )

            upload_time = time.time() - upload_start
            total_mb = total_bytes / (1024 * 1024)

            if failed_files:
                print(
                    f"[{node_str}, Rank {global_rank}] "
                    f"Uploaded step {step} with errors: "
                    f"{file_count} files succeeded, {len(failed_files)} failed"
                )
                print(
                    f"    Failed files: {', '.join(failed_files[:5])}"
                    f"{'...' if len(failed_files) > 5 else ''}"
                )
            else:
                if verbose or log_upload_progress:
                    throughput = total_mb / upload_time if upload_time > 0 else 0
                    print(
                        f"[{node_str}, Rank {global_rank}] "
                        f"Uploaded step {step}: {file_count} files, "
                        f"{total_mb:.1f}MB in {upload_time:.1f}s "
                        f"({throughput:.1f} MB/s)"
                    )

            # Clean up local checkpoint after successful upload
            if cleanup_after_upload and not failed_files:
                try:
                    shutil.rmtree(checkpoint_dir, ignore_errors=True)
                    if verbose:
                        print(
                            f"[{node_str}, Rank {global_rank}] "
                            f"Cleaned up local checkpoint after upload: {tag}"
                        )
                except Exception:
                    pass

            # Signal completion back to parent
            done_queue.put((step, len(failed_files) == 0))

        except Exception as e:
            print(f"[Rank {global_rank}] " f"Upload process error: {e}")
            try:
                done_queue.put((step, False))
            except Exception:
                pass


class S3CheckpointManager:
    """
    Universal S3 Checkpoint Manager for DeepSpeed.

    Features:
    - Automatic detection of single-node, multi-GPU, and multi-node setups
    - Non-blocking background upload to S3
    - Per-node upload threads (one uploader per node)
    - Automatic file distribution in multi-node scenarios
    - Retry logic with exponential backoff
    - Progress tracking and detailed logging
    - Graceful error handling

    Usage:
        >>> from lightninglm.aws.config import S3Config
        >>> config = S3Config(bucket_name="my-bucket", s3_prefix="training")
        >>> checkpoint_mgr = S3CheckpointManager(config)
        >>>
        >>> # During training
        >>> checkpoint_mgr.save_checkpoint(model_engine, step=100)
        >>>
        >>> # At the end
        >>> checkpoint_mgr.wait_for_uploads()
        >>> checkpoint_mgr.cleanup_old_checkpoints()
    """

    def __init__(self, config: S3Config):
        """
        Initialize checkpoint manager.

        Args:
            config: S3Config instance with AWS and checkpoint settings
        """
        self.config = config
        config.validate()

        # Detect distributed training setup
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Calculate node information
        self.gpus_per_node = (
            torch.cuda.device_count() if torch.cuda.is_available() else 1
        )
        self.node_rank = self.global_rank // self.gpus_per_node
        self.num_nodes = (
            self.world_size + self.gpus_per_node - 1
        ) // self.gpus_per_node

        # Determine upload responsibility
        self.is_local_main = self.local_rank == 0  # One per node
        self.is_global_main = self.global_rank == 0  # Only one globally

        # Create local checkpoint directory
        os.makedirs(self.config.local_checkpoint_dir, exist_ok=True)

        # Log setup information
        if self.is_global_main and self.config.verbose:
            self._log_setup_info()

        # Initialize S3 client and uploader thread (one per node)
        if self.is_local_main:
            self._init_s3_client()
            self._init_upload_thread()

        # Track active uploads
        self.active_uploads = []
        self._upload_lock = threading.Lock()

    def _log_setup_info(self):
        """Log checkpoint manager configuration."""
        print(f"\n{'=' * 70}")
        print(f"{'S3 Checkpoint Manager Configuration':^70}")
        print(f"{'=' * 70}")
        print(f"  World Size:        {self.world_size} GPUs")
        print(f"  Number of Nodes:   {self.num_nodes}")
        print(f"  GPUs per Node:     {self.gpus_per_node}")
        print(
            f"  S3 Bucket:         s3://{self.config.bucket_name}/{self.config.s3_prefix}"
        )
        print(f"  Local Directory:   {self.config.local_checkpoint_dir}")
        print(f"  Keep Checkpoints:  {self.config.keep_last_n_checkpoints}")
        print(
            f"  Upload Strategy:   {'Multi-node' if self.num_nodes > 1 else 'Single-node'}"
        )
        print(f"{'=' * 70}\n")

    def _init_s3_client(self):
        """Initialize S3 client with configuration."""
        try:
            boto3_config = self.config.get_boto3_config()
            self.s3_client = boto3.client("s3", **boto3_config)

            # Test S3 connectivity
            self.s3_client.head_bucket(Bucket=self.config.bucket_name)

            if self.config.verbose:
                print(
                    f"[Node {self.node_rank}, Rank {self.global_rank}] "
                    f"S3 client initialized and connected"
                )

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                print(
                    f"[Node {self.node_rank}, Rank {self.global_rank}] "
                    f"Bucket '{self.config.bucket_name}' not found. "
                    f"Please create it before training."
                )
            else:
                print(
                    f"[Node {self.node_rank}, Rank {self.global_rank}] "
                    f"S3 connectivity error: {e}"
                )
            raise

        except Exception as e:
            print(
                f"[Node {self.node_rank}, Rank {self.global_rank}] "
                f"FAILED to initialize S3 client: {e}"
            )
            raise

    def _init_upload_thread(self):
        """Initialize background upload process and completion listener."""
        self.upload_queue = MPQueue()
        self._done_queue = MPQueue()

        # Serialize config for the child process
        config_dict = {
            "region": self.config.region,
            "bucket_name": self.config.bucket_name,
            "s3_prefix": self.config.s3_prefix,
            "max_retries": self.config.max_retries,
            "retry_backoff_base": self.config.retry_backoff_base,
            "max_file_parallelism": self.config.max_file_parallelism,
            "max_concurrency": self.config.max_concurrency,
            "multipart_threshold": self.config.multipart_threshold,
            "multipart_chunksize": self.config.multipart_chunksize,
            "verbose": self.config.verbose,
            "log_upload_progress": self.config.log_upload_progress,
            "cleanup_after_upload": self.config.cleanup_after_upload,
        }

        self.upload_process = multiprocessing.Process(
            target=_upload_process_worker,
            args=(
                self.upload_queue,
                self._done_queue,
                config_dict,
                self.node_rank,
                self.global_rank,
                self.num_nodes,
                self.gpus_per_node,
            ),
            daemon=True,
            name=f"S3-Uploader-Node{self.node_rank}",
        )
        self.upload_process.start()

        # Lightweight listener thread to drain done_queue and update active_uploads
        self._done_listener = threading.Thread(
            target=self._done_listener_loop,
            daemon=True,
            name=f"S3-Done-Listener-Node{self.node_rank}",
        )
        self._done_listener.start()

        if self.config.verbose:
            print(
                f"[Node {self.node_rank}, Rank {self.global_rank}] "
                f"Upload process started (PID {self.upload_process.pid})"
            )

    def _done_listener_loop(self):
        """Listen for upload completions from the child process."""
        while True:
            try:
                step, success = self._done_queue.get()
                with self._upload_lock:
                    if step in self.active_uploads:
                        self.active_uploads.remove(step)
            except Exception:
                break

    def save_checkpoint(
        self,
        model_engine,
        step: int,
        client_state: Optional[Dict[str, Any]] = None,
        tag: Optional[str] = None,
        urgent: bool = False,
    ):
        """
        Save checkpoint locally and queue for S3 upload.

        Args:
            model_engine: DeepSpeed model engine
            step: Training step number
            client_state: Optional client state dictionary to save
            tag: Optional custom tag (defaults to f"step_{step}")
            urgent: If True, upload at full speed with no throttling
                    (used for spot termination — every second counts)
        """
        if tag is None:
            tag = f"step_{step}"

        if client_state is None:
            client_state = {"step": step}
        elif "step" not in client_state:
            client_state["step"] = step

        # All ranks save locally (DeepSpeed requirement)
        start_time = time.time()

        try:
            model_engine.save_checkpoint(
                save_dir=self.config.local_checkpoint_dir,
                tag=tag,
                client_state=client_state,
            )
            save_time = time.time() - start_time

            if self.is_local_main and self.config.verbose:
                node_str = (
                    f"Node {self.node_rank}" if self.num_nodes > 1 else "Single-node"
                )
                print(
                    f"[{node_str}, Rank {self.global_rank}] "
                    f"Saved locally in {save_time:.2f}s: {tag}"
                )

        except Exception as e:
            print(f"[Rank {self.global_rank}] FAILED to save checkpoint '{tag}': {e}")
            raise

        # Synchronize all ranks before uploading
        if dist.is_initialized():
            dist.barrier()

        # Each node's local_rank=0 uploads
        if self.is_local_main:
            checkpoint_dir = os.path.join(self.config.local_checkpoint_dir, tag)

            with self._upload_lock:
                self.upload_queue.put((checkpoint_dir, tag, step, urgent))
                self.active_uploads.append(step)

            if self.config.verbose:
                if self.num_nodes > 1:
                    print(
                        f"[Node {self.node_rank}, Rank {self.global_rank}] "
                        f"Queued for S3 upload: step {step}"
                    )
                else:
                    print(
                        f"[Rank {self.global_rank}] Queued for S3 upload: step {step}"
                    )

    def load_checkpoint(
        self, model_engine, step: int, tag: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint from S3 (downloads if needed) and restore model state.

        Args:
            model_engine: DeepSpeed model engine
            step: Training step number
            tag: Optional custom tag (defaults to f"step_{step}")

        Returns:
            Client state dictionary from checkpoint

        Example:
            >>> client_state = checkpoint_mgr.load_checkpoint(model_engine, step=1000)
            >>> epoch = client_state.get('epoch', 0)
        """
        if tag is None:
            tag = f"step_{step}"

        if self.is_global_main and self.config.verbose:
            print(f"Loading checkpoint: step {step} (tag: {tag})")

        local_checkpoint_dir = os.path.join(self.config.local_checkpoint_dir, tag)

        # Download from S3 if not present locally
        if not os.path.exists(local_checkpoint_dir):
            if self.is_local_main:
                self._download_checkpoint(local_checkpoint_dir, tag)

            # Synchronize after download
            if dist.is_initialized():
                dist.barrier()

        # Load checkpoint (all ranks)
        try:
            _, client_state = model_engine.load_checkpoint(
                load_dir=self.config.local_checkpoint_dir, tag=tag
            )

            if self.is_global_main and self.config.verbose:
                print(f"Loaded checkpoint: step {step}")

            return client_state

        except Exception as e:
            print(f"[Rank {self.global_rank}] FAILED to load checkpoint '{tag}': {e}")
            raise

    def _download_checkpoint(self, local_checkpoint_dir: str, tag: str):
        """
        Download checkpoint files from S3.

        Args:
            local_checkpoint_dir: Local directory to save checkpoint
            tag: Checkpoint tag
        """
        os.makedirs(local_checkpoint_dir, exist_ok=True)

        if self.config.verbose:
            print(f"[Rank {self.global_rank}] Downloading checkpoint from S3...")

        download_start = time.time()
        file_count = 0

        try:
            # Determine prefix based on node configuration
            if self.num_nodes > 1:
                # Multi-node: download this node's files
                node_prefix = f"{self.config.s3_prefix}/{tag}/node_{self.node_rank}/"
            else:
                # Single-node: download all files
                node_prefix = f"{self.config.s3_prefix}/{tag}/"

            # Download node-specific files
            file_count += self._download_files_from_prefix(
                node_prefix, local_checkpoint_dir
            )

            # Multi-node: also download shared files from node 0
            if self.num_nodes > 1 and self.node_rank != 0:
                shared_prefix = f"{self.config.s3_prefix}/{tag}/node_0/"
                file_count += self._download_files_from_prefix(
                    shared_prefix, local_checkpoint_dir, skip_model_ranks=True
                )

            download_time = time.time() - download_start

            if self.config.verbose:
                print(
                    f"[Rank {self.global_rank}] "
                    f"Downloaded {file_count} files in {download_time:.1f}s"
                )

        except Exception as e:
            print(f"[Rank {self.global_rank}] FAILED to download checkpoint: {e}")
            raise

    def _download_files_from_prefix(
        self, prefix: str, local_dir: str, skip_model_ranks: bool = False
    ) -> int:
        """
        Download all files with given S3 prefix.

        Args:
            prefix: S3 prefix to download from
            local_dir: Local directory to save files
            skip_model_ranks: Skip mp_rank_ files (for shared metadata)

        Returns:
            Number of files downloaded
        """
        file_count = 0

        paginator = self.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.config.bucket_name, Prefix=prefix)

        for page in pages:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                s3_key = obj["Key"]
                filename = os.path.basename(s3_key)

                # Skip model rank files if requested (for shared metadata)
                if skip_model_ranks and filename.startswith("mp_rank_"):
                    continue

                local_path = os.path.join(local_dir, filename)

                try:
                    self.s3_client.download_file(
                        self.config.bucket_name, s3_key, local_path
                    )
                    file_count += 1

                except Exception as e:
                    print(
                        f"[Rank {self.global_rank}] "
                        f"Failed to download '{filename}': {e}"
                    )

        return file_count

    def wait_for_uploads(self):
        """
        Block until all pending uploads complete.

        Call this at the end of training or before exiting.
        """
        if self.is_local_main:
            with self._upload_lock:
                pending = len(self.active_uploads)

            if pending > 0:
                node_str = (
                    f"Node {self.node_rank}" if self.num_nodes > 1 else "Single-node"
                )
                print(
                    f"[{node_str}, Rank {self.global_rank}] "
                    f"Waiting for {pending} upload(s) to complete..."
                )

                # Poll until all uploads are done
                while True:
                    with self._upload_lock:
                        if len(self.active_uploads) == 0:
                            break
                    time.sleep(0.5)

                print(f"[{node_str}, Rank {self.global_rank}] All uploads complete.")

        # Synchronize all processes
        if dist.is_initialized():
            dist.barrier()

        if self.is_global_main and self.config.verbose:
            print("All checkpoints uploaded across all nodes.")

    def cleanup_old_checkpoints(self, keep_last_n: Optional[int] = None):
        """
        Clean up old local checkpoints, keeping only the most recent N.
        Skips checkpoints that are still pending S3 upload.

        Args:
            keep_last_n: Number of checkpoints to keep (uses config default if None)

        Example:
            >>> checkpoint_mgr.cleanup_old_checkpoints(keep_last_n=3)
        """
        if not self.is_local_main:
            return

        if keep_last_n is None:
            keep_last_n = self.config.keep_last_n_checkpoints

        try:
            # Get steps still pending upload
            with self._upload_lock:
                pending_steps = set(self.active_uploads)

            # List all checkpoint directories
            checkpoint_dirs = []
            for item in os.listdir(self.config.local_checkpoint_dir):
                item_path = os.path.join(self.config.local_checkpoint_dir, item)
                if os.path.isdir(item_path) and item.startswith("step_"):
                    try:
                        step_num = int(item.split("_")[1])
                        checkpoint_dirs.append((step_num, item))
                    except (ValueError, IndexError):
                        continue

            # Sort by step number (newest last)
            checkpoint_dirs.sort(key=lambda x: x[0])

            # Remove old checkpoints (skip pending uploads)
            if len(checkpoint_dirs) > keep_last_n:
                to_remove = checkpoint_dirs[:-keep_last_n]

                for step_num, dir_name in to_remove:
                    if step_num in pending_steps:
                        if self.config.verbose:
                            print(
                                f"[Rank {self.global_rank}] "
                                f"Skipping cleanup of {dir_name} (S3 upload pending)"
                            )
                        continue

                    dir_path = os.path.join(self.config.local_checkpoint_dir, dir_name)

                    if self.config.verbose:
                        node_str = (
                            f"Node {self.node_rank}" if self.num_nodes > 1 else "Local"
                        )
                        print(f"[{node_str}] Removing old checkpoint: {dir_name}")

                    shutil.rmtree(dir_path)

        except Exception as e:
            print(f"[Rank {self.global_rank}] Error during checkpoint cleanup: {e}")

    def list_available_checkpoints(self) -> list:
        """
        List all available checkpoints in S3.

        Returns:
            List of checkpoint tags available in S3

        Example:
            >>> checkpoints = checkpoint_mgr.list_available_checkpoints()
            >>> print(f"Available: {checkpoints}")
        """
        if not self.is_local_main:
            return []

        try:
            checkpoints = set()
            prefix = f"{self.config.s3_prefix}/"

            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=self.config.bucket_name, Prefix=prefix, Delimiter="/"
            )

            for page in pages:
                if "CommonPrefixes" in page:
                    for prefix_info in page["CommonPrefixes"]:
                        # Extract tag from prefix like "training/checkpoints/step_1000/"
                        full_prefix = prefix_info["Prefix"]
                        tag = full_prefix.rstrip("/").split("/")[-1]
                        if tag.startswith("step_"):
                            checkpoints.add(tag)

            return sorted(list(checkpoints), key=lambda x: int(x.split("_")[1]))

        except Exception as e:
            print(f"[Rank {self.global_rank}] Error listing checkpoints: {e}")
            return []

    def get_latest_checkpoint_step(self) -> Optional[int]:
        """
        Get the step number of the latest checkpoint in S3.

        Returns:
            Latest checkpoint step number, or None if no checkpoints exist

        Example:
            >>> latest_step = checkpoint_mgr.get_latest_checkpoint_step()
            >>> if latest_step:
            ...     checkpoint_mgr.load_checkpoint(model_engine, latest_step)
        """
        checkpoints = self.list_available_checkpoints()

        if not checkpoints:
            return None

        # Extract step numbers and return the maximum
        try:
            steps = [int(tag.split("_")[1]) for tag in checkpoints]
            return max(steps)
        except (ValueError, IndexError):
            return None
