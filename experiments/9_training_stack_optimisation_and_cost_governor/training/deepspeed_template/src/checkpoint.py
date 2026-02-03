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

import os
import shutil
import threading
import time
from queue import Queue
from typing import Any, Dict, Optional

import boto3
import torch
import torch.distributed as dist
from aws.config import S3Config
from botocore.exceptions import ClientError


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
        >>> from aws.config import S3Config
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
        print(f"\n{'='*70}")
        print(f"{'S3 Checkpoint Manager Configuration':^70}")
        print(f"{'='*70}")
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
        print(f"{'='*70}\n")

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
                    f"✓ S3 client initialized and connected"
                )

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                print(
                    f"[Node {self.node_rank}, Rank {self.global_rank}] "
                    f"⚠️  Bucket '{self.config.bucket_name}' not found. "
                    f"Please create it before training."
                )
            else:
                print(
                    f"[Node {self.node_rank}, Rank {self.global_rank}] "
                    f"⚠️  S3 connectivity error: {e}"
                )
            raise

        except Exception as e:
            print(
                f"[Node {self.node_rank}, Rank {self.global_rank}] "
                f"❌ Failed to initialize S3 client: {e}"
            )
            raise

    def _init_upload_thread(self):
        """Initialize background upload thread."""
        self.upload_queue = Queue()
        self.upload_thread = threading.Thread(
            target=self._upload_worker,
            daemon=True,
            name=f"S3-Uploader-Node{self.node_rank}",
        )
        self.upload_thread.start()

        if self.config.verbose:
            print(
                f"[Node {self.node_rank}, Rank {self.global_rank}] "
                f"✓ Upload thread started"
            )

    def save_checkpoint(
        self,
        model_engine,
        step: int,
        client_state: Optional[Dict[str, Any]] = None,
        tag: Optional[str] = None,
    ):
        """
        Save checkpoint locally and queue for S3 upload.

        Args:
            model_engine: DeepSpeed model engine
            step: Training step number
            client_state: Optional client state dictionary to save
            tag: Optional custom tag (defaults to f"step_{step}")

        Example:
            >>> checkpoint_mgr.save_checkpoint(
            ...     model_engine,
            ...     step=1000,
            ...     client_state={'epoch': 2, 'best_loss': 0.5}
            ... )
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
                    f"💾 Saved locally in {save_time:.2f}s: {tag}"
                )

        except Exception as e:
            print(
                f"[Rank {self.global_rank}] "
                f"❌ Failed to save checkpoint '{tag}': {e}"
            )
            raise

        # Synchronize all ranks before uploading
        if dist.is_initialized():
            dist.barrier()

        # Each node's local_rank=0 uploads
        if self.is_local_main:
            checkpoint_dir = os.path.join(self.config.local_checkpoint_dir, tag)

            with self._upload_lock:
                self.upload_queue.put((checkpoint_dir, tag, step))
                self.active_uploads.append(step)

            if self.config.verbose:
                if self.num_nodes > 1:
                    print(
                        f"[Node {self.node_rank}, Rank {self.global_rank}] "
                        f"📤 Queued for S3 upload: step {step}"
                    )
                else:
                    print(
                        f"[Rank {self.global_rank}] 📤 Queued for S3 upload: step {step}"
                    )

    def _upload_worker(self):
        """Background worker thread for uploading checkpoints to S3."""
        transfer_config = self.config.get_transfer_config()

        while True:
            try:
                checkpoint_dir, tag, step = self.upload_queue.get()

                node_str = (
                    f"Node {self.node_rank}" if self.num_nodes > 1 else "Single-node"
                )

                if self.config.verbose:
                    print(
                        f"[{node_str}, Rank {self.global_rank}] "
                        f"⬆️  Starting upload: step {step}"
                    )

                upload_start = time.time()
                file_count = 0
                total_bytes = 0
                failed_files = []

                # Walk through checkpoint directory
                for root, dirs, files in os.walk(checkpoint_dir):
                    for file in files:
                        local_path = os.path.join(root, file)

                        # Determine S3 key based on setup
                        if self.num_nodes > 1:
                            # Multi-node: organize by node
                            relative_path = os.path.relpath(local_path, checkpoint_dir)
                            s3_key = f"{self.config.s3_prefix}/{tag}/node_{self.node_rank}/{relative_path}"
                        else:
                            # Single-node: flat structure
                            relative_path = os.path.relpath(local_path, checkpoint_dir)
                            s3_key = f"{self.config.s3_prefix}/{tag}/{relative_path}"

                        # Only upload files belonging to this node
                        if not self._should_upload_file(file):
                            continue

                        # Upload file with retry logic
                        success = self._upload_file_with_retry(
                            local_path, s3_key, transfer_config
                        )

                        if success:
                            file_size = os.path.getsize(local_path)
                            file_count += 1
                            total_bytes += file_size
                        else:
                            failed_files.append(file)

                upload_time = time.time() - upload_start
                total_mb = total_bytes / (1024 * 1024)

                # Report results
                if failed_files:
                    print(
                        f"[{node_str}, Rank {self.global_rank}] "
                        f"⚠️  Uploaded step {step} with errors: "
                        f"{file_count} files succeeded, {len(failed_files)} failed"
                    )
                    print(
                        f"    Failed files: {', '.join(failed_files[:5])}"
                        f"{'...' if len(failed_files) > 5 else ''}"
                    )
                else:
                    if self.config.verbose or self.config.log_upload_progress:
                        throughput = total_mb / upload_time if upload_time > 0 else 0
                        print(
                            f"[{node_str}, Rank {self.global_rank}] "
                            f"✅ Uploaded step {step}: {file_count} files, "
                            f"{total_mb:.1f}MB in {upload_time:.1f}s "
                            f"({throughput:.1f} MB/s)"
                        )

                # Mark upload as complete
                with self._upload_lock:
                    if step in self.active_uploads:
                        self.active_uploads.remove(step)

            except Exception as e:
                print(
                    f"[Rank {self.global_rank}] "
                    f"❌ Upload worker error for step {step}: {e}"
                )

            finally:
                self.upload_queue.task_done()

    def _upload_file_with_retry(
        self, local_path: str, s3_key: str, transfer_config
    ) -> bool:
        """
        Upload a single file with retry logic.

        Args:
            local_path: Local file path
            s3_key: S3 key (destination path)
            transfer_config: Boto3 transfer configuration

        Returns:
            True if upload succeeded, False otherwise
        """
        for attempt in range(self.config.max_retries):
            try:
                self.s3_client.upload_file(
                    local_path, self.config.bucket_name, s3_key, Config=transfer_config
                )
                return True

            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    print(
                        f"[Rank {self.global_rank}] "
                        f"❌ Failed to upload '{os.path.basename(local_path)}' "
                        f"after {self.config.max_retries} attempts: {e}"
                    )
                    return False

                # Exponential backoff
                sleep_time = self.config.retry_backoff_base**attempt
                if self.config.verbose:
                    print(
                        f"[Rank {self.global_rank}] "
                        f"⚠️  Upload attempt {attempt + 1} failed for "
                        f"'{os.path.basename(local_path)}', "
                        f"retrying in {sleep_time}s..."
                    )
                time.sleep(sleep_time)

        return False

    def _should_upload_file(self, filename: str) -> bool:
        """
        Determine if this process should upload this file.

        In single-node: upload everything (only one uploader).
        In multi-node: distribute files by rank to avoid duplicate uploads.

        Args:
            filename: Name of the file

        Returns:
            True if this process should upload the file
        """
        if self.num_nodes == 1:
            # Single node: only one uploader, upload everything
            return True

        # Multi-node: distribute files by rank
        # DeepSpeed saves model partitions as mp_rank_XX_model_states.pt
        if filename.startswith("mp_rank_"):
            try:
                # Extract rank from filename like "mp_rank_00_model_states.pt"
                parts = filename.split("_")
                if len(parts) >= 3:
                    rank_str = parts[2]
                    file_rank = int(rank_str)
                    file_node = file_rank // self.gpus_per_node
                    return file_node == self.node_rank
            except (ValueError, IndexError):
                # If parsing fails, upload from node 0
                return self.node_rank == 0

        # Metadata files (latest, zero_pp_rank_*): only node 0 uploads
        return self.node_rank == 0

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
            print(f"📥 Loading checkpoint: step {step} (tag: {tag})")

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
                print(f"✓ Loaded checkpoint: step {step}")

            return client_state

        except Exception as e:
            print(
                f"[Rank {self.global_rank}] "
                f"❌ Failed to load checkpoint '{tag}': {e}"
            )
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
            print(f"[Rank {self.global_rank}] 📥 Downloading checkpoint from S3...")

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
                    f"✓ Downloaded {file_count} files in {download_time:.1f}s"
                )

        except Exception as e:
            print(
                f"[Rank {self.global_rank}] " f"❌ Failed to download checkpoint: {e}"
            )
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
                        f"⚠️  Failed to download '{filename}': {e}"
                    )

        return file_count

    def wait_for_uploads(self):
        """
        Block until all pending uploads complete.

        Call this at the end of training or before exiting.

        Example:
            >>> checkpoint_mgr.save_checkpoint(model_engine, step=1000)
            >>> checkpoint_mgr.wait_for_uploads()  # Blocks until upload finishes
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
                    f"⏳ Waiting for {pending} upload(s) to complete..."
                )

                self.upload_queue.join()

                print(
                    f"[{node_str}, Rank {self.global_rank}] " f"✓ All uploads complete!"
                )

        # Synchronize all processes
        if dist.is_initialized():
            dist.barrier()

        if self.is_global_main and self.config.verbose:
            print("✅ All checkpoints uploaded across all nodes!")

    def cleanup_old_checkpoints(self, keep_last_n: Optional[int] = None):
        """
        Clean up old local checkpoints, keeping only the most recent N.

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

            # Remove old checkpoints
            if len(checkpoint_dirs) > keep_last_n:
                to_remove = checkpoint_dirs[:-keep_last_n]

                for step_num, dir_name in to_remove:
                    dir_path = os.path.join(self.config.local_checkpoint_dir, dir_name)

                    if self.config.verbose:
                        node_str = (
                            f"Node {self.node_rank}" if self.num_nodes > 1 else "Local"
                        )
                        print(f"[{node_str}] 🗑️  Removing old checkpoint: {dir_name}")

                    shutil.rmtree(dir_path)

        except Exception as e:
            print(
                f"[Rank {self.global_rank}] "
                f"⚠️  Error during checkpoint cleanup: {e}"
            )

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
            print(f"[Rank {self.global_rank}] " f"⚠️  Error listing checkpoints: {e}")
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
