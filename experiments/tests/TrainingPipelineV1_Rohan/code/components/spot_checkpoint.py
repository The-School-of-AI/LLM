"""
Spot-aware checkpoint orchestrator for EC2 spot instance training.

Features:
- Periodic checkpointing (default: 1 hour)
- On-demand checkpoint via keyboard signal (SIGUSR1 or Ctrl+C once)
- AWS spot termination notice listener (2-min warning)
- Full checkpoint: model, optimizer, scheduler, shard state, RNG states
- Non-blocking S3 sync with shard metadata
- Log rotation + background S3 log upload
- Double Ctrl+C = hard abort (safety valve)

Usage in training loop:
    from components.spot_checkpoint import SpotCheckpointOrchestrator

    orch = SpotCheckpointOrchestrator(
        checkpoint_interval_seconds=3600,
        s3_bucket="my-bucket",
        s3_prefix="training/run_001",
        log_dir="/tmp/training_logs",
        metrics_jsonl_path="results/run/metrics.jsonl",
    )
    orch.install_signal_handlers()
    orch.start_spot_listener()
    orch.start_log_uploader()

    # In training loop, after each step:
    if orch.should_checkpoint(global_step):
        reason = orch.get_checkpoint_reason()
        orch.save_full_checkpoint(model_engine, global_step, epoch, step_in_epoch,
                                  shard_state=dataloader_state, extra_client_state={...})
"""

import json
import os
import signal
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist


def _is_rank_0() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0


def _print_r0(msg: str):
    if _is_rank_0():
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] [SpotCkpt] {msg}", flush=True)


class SpotTerminationListener:
    """
    Polls EC2 instance metadata for spot termination notice.
    AWS gives a 2-minute warning at:
      http://169.254.169.254/latest/meta-data/spot/instance-action

    Runs in a daemon thread, sets an event when termination is detected.
    """

    METADATA_URL = "http://169.254.169.254/latest/meta-data/spot/instance-action"
    TOKEN_URL = "http://169.254.169.254/latest/api/token"

    def __init__(self, poll_interval: float = 5.0):
        self.poll_interval = poll_interval
        self.termination_event = threading.Event()
        self.termination_time: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="spot-listener")
        self._thread.start()
        _print_r0(f"Spot termination listener started (poll every {self.poll_interval}s)")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _get_imds_token(self) -> Optional[str]:
        """Get IMDSv2 token."""
        import urllib.request
        try:
            req = urllib.request.Request(
                self.TOKEN_URL,
                method="PUT",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "30"},
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.read().decode()
        except Exception:
            return None

    def _check_spot_termination(self) -> Optional[dict]:
        import urllib.request
        import urllib.error
        try:
            headers = {}
            token = self._get_imds_token()
            if token:
                headers["X-aws-ec2-metadata-token"] = token
            req = urllib.request.Request(self.METADATA_URL, headers=headers)
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode())
                return data  # {"action": "terminate", "time": "2024-..."}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # No termination notice — normal
            return None
        except Exception:
            return None

    def _poll_loop(self):
        while not self._stop.is_set():
            result = self._check_spot_termination()
            if result and result.get("action") in ("terminate", "stop", "hibernate"):
                self.termination_time = result.get("time", "unknown")
                _print_r0(f"⚠️  SPOT TERMINATION NOTICE: action={result['action']}, time={self.termination_time}")
                self.termination_event.set()
                return  # Stop polling, termination is imminent
            self._stop.wait(self.poll_interval)

    @property
    def is_terminating(self) -> bool:
        return self.termination_event.is_set()


class LogRotatorAndUploader:
    """
    Rotates local JSONL log files and uploads them to S3 in the background.

    - Rotates every `max_steps_per_file` steps (or `max_bytes` size)
    - Uploads rotated files to S3 in a daemon thread
    - Keeps local log size bounded
    - Does NOT interfere with training (separate thread, separate I/O)
    """

    def __init__(
        self,
        metrics_jsonl_path: str,
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "training/logs",
        s3_region: str = "us-east-1",
        max_bytes: int = 50 * 1024 * 1024,  # 50MB per file
        check_interval: float = 60.0,  # Check every 60s
    ):
        self.metrics_jsonl_path = metrics_jsonl_path
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_region = s3_region
        self.max_bytes = max_bytes
        self.check_interval = check_interval
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._s3_client = None
        self._rotation_count = 0

    def start(self):
        if not _is_rank_0():
            return
        if self._thread is not None:
            return
        if self.s3_bucket:
            try:
                import boto3
                self._s3_client = boto3.client("s3", region_name=self.s3_region)
            except Exception as e:
                _print_r0(f"Log uploader: boto3 init failed: {e}")
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="log-uploader")
        self._thread.start()
        _print_r0(f"Log rotator/uploader started (max {self.max_bytes // (1024*1024)}MB/file, "
                   f"s3={'enabled' if self.s3_bucket else 'disabled'})")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=30)

    def force_upload_current(self):
        """Force upload the current log file (e.g., before shutdown)."""
        if not _is_rank_0() or not self._s3_client:
            return
        try:
            if os.path.exists(self.metrics_jsonl_path):
                self._upload_file(self.metrics_jsonl_path, suffix="_final")
        except Exception as e:
            _print_r0(f"Force upload failed: {e}")

    def _run_loop(self):
        while not self._stop.is_set():
            try:
                self._check_and_rotate()
            except Exception as e:
                _print_r0(f"Log rotation error: {e}")
            self._stop.wait(self.check_interval)

    def _check_and_rotate(self):
        if not os.path.exists(self.metrics_jsonl_path):
            return
        file_size = os.path.getsize(self.metrics_jsonl_path)
        if file_size >= self.max_bytes:
            self._rotate()

    def _rotate(self):
        self._rotation_count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_name = f"{self.metrics_jsonl_path}.{ts}.{self._rotation_count}"
        try:
            os.rename(self.metrics_jsonl_path, rotated_name)
            # Touch a new empty file so the training loop can keep writing
            Path(self.metrics_jsonl_path).touch()
            _print_r0(f"Log rotated: {rotated_name}")
            if self._s3_client:
                self._upload_file(rotated_name)
                # Remove local rotated file after successful upload
                try:
                    os.remove(rotated_name)
                except OSError:
                    pass
        except Exception as e:
            _print_r0(f"Log rotation failed: {e}")

    def _upload_file(self, local_path: str, suffix: str = ""):
        if not self._s3_client or not self.s3_bucket:
            return
        filename = os.path.basename(local_path) + suffix
        s3_key = f"{self.s3_prefix}/logs/{filename}"
        try:
            self._s3_client.upload_file(local_path, self.s3_bucket, s3_key)
            _print_r0(f"Log uploaded: s3://{self.s3_bucket}/{s3_key}")
        except Exception as e:
            _print_r0(f"Log upload failed: {e}")


class SpotCheckpointOrchestrator:
    """
    Central orchestrator for all checkpoint triggers:
    - Periodic (every N seconds, default 3600 = 1 hour)
    - On-demand (SIGUSR1 signal or single Ctrl+C)
    - Spot termination (EC2 metadata polling)

    Saves FULL checkpoints: model weights, optimizer, scheduler, RNG states,
    dataloader shard position, and custom client state.
    """

    def __init__(
        self,
        checkpoint_interval_seconds: int = 3600,
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "training/checkpoints",
        s3_region: str = "us-east-1",
        local_checkpoint_dir: str = "./checkpoints",
        log_dir: Optional[str] = None,
        metrics_jsonl_path: Optional[str] = None,
        spot_poll_interval: float = 5.0,
        log_max_bytes: int = 50 * 1024 * 1024,
        keep_last_n_local: int = 3,
    ):
        self.checkpoint_interval_seconds = checkpoint_interval_seconds
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_region = s3_region
        self.local_checkpoint_dir = local_checkpoint_dir
        self.keep_last_n_local = keep_last_n_local

        # Signal state
        self._on_demand_requested = threading.Event()
        self._original_sigint = None
        self._sigint_count = 0
        self._sigint_lock = threading.Lock()
        self._last_sigint_time = 0.0

        # Timing
        self._last_checkpoint_time = time.time()
        self._last_checkpoint_step = 0

        # Sub-components
        self._spot_listener = SpotTerminationListener(poll_interval=spot_poll_interval)
        self._log_uploader: Optional[LogRotatorAndUploader] = None
        if metrics_jsonl_path:
            self._log_uploader = LogRotatorAndUploader(
                metrics_jsonl_path=metrics_jsonl_path,
                s3_bucket=s3_bucket,
                s3_prefix=s3_prefix,
                s3_region=s3_region,
                max_bytes=log_max_bytes,
            )

        # S3 client for checkpoint metadata upload
        self._s3_client = None
        if s3_bucket:
            try:
                import boto3
                self._s3_client = boto3.client("s3", region_name=s3_region)
            except Exception:
                pass

        os.makedirs(local_checkpoint_dir, exist_ok=True)

    # ── Signal Handlers ──────────────────────────────────────────────────

    def install_signal_handlers(self):
        """
        Install signal handlers for on-demand checkpointing.

        - SIGUSR1: Trigger checkpoint (safe, no side effects)
        - SIGINT (Ctrl+C): First press = checkpoint, second within 5s = abort
        """
        if not _is_rank_0():
            return

        self._original_sigint = signal.getsignal(signal.SIGINT)

        def _handle_sigusr1(signum, frame):
            _print_r0("📸 SIGUSR1 received — on-demand checkpoint requested")
            self._on_demand_requested.set()

        def _handle_sigint(signum, frame):
            now = time.time()
            with self._sigint_lock:
                if now - self._last_sigint_time < 5.0:
                    self._sigint_count += 1
                else:
                    self._sigint_count = 1
                self._last_sigint_time = now

                if self._sigint_count >= 2:
                    _print_r0("🛑 Double Ctrl+C — aborting immediately")
                    # Restore original handler and re-raise
                    signal.signal(signal.SIGINT, self._original_sigint)
                    os.kill(os.getpid(), signal.SIGINT)
                    return

            _print_r0("📸 Ctrl+C received — on-demand checkpoint requested (press again within 5s to abort)")
            self._on_demand_requested.set()

        signal.signal(signal.SIGUSR1, _handle_sigusr1)
        signal.signal(signal.SIGINT, _handle_sigint)
        _print_r0("Signal handlers installed: SIGUSR1=checkpoint, Ctrl+C=checkpoint (x2=abort)")

    # ── Start/Stop ───────────────────────────────────────────────────────

    def start_spot_listener(self):
        self._spot_listener.start()

    def start_log_uploader(self):
        if self._log_uploader:
            self._log_uploader.start()

    def start_all(self):
        """Convenience: install signals + start all background threads."""
        self.install_signal_handlers()
        self.start_spot_listener()
        self.start_log_uploader()

    def shutdown(self):
        """Graceful shutdown of all background threads."""
        self._spot_listener.stop()
        if self._log_uploader:
            self._log_uploader.force_upload_current()
            self._log_uploader.stop()
        _print_r0("Orchestrator shut down")

    # ── Checkpoint Decision ──────────────────────────────────────────────

    def should_checkpoint(self, global_step: int) -> bool:
        """
        Check if a checkpoint should be saved right now.
        Call this after every training step.

        Returns True if any trigger fires:
        1. Periodic timer expired
        2. On-demand signal received
        3. Spot termination notice detected
        """
        # Spot termination — highest priority
        if self._spot_listener.is_terminating:
            return True

        # On-demand request (signal)
        if self._on_demand_requested.is_set():
            return True

        # Periodic timer
        elapsed = time.time() - self._last_checkpoint_time
        if elapsed >= self.checkpoint_interval_seconds:
            return True

        return False

    def get_checkpoint_reason(self) -> str:
        """Return human-readable reason for the checkpoint trigger."""
        if self._spot_listener.is_terminating:
            return f"spot_termination (time={self._spot_listener.termination_time})"
        if self._on_demand_requested.is_set():
            return "on_demand (signal)"
        return "periodic"

    def clear_on_demand(self):
        """Clear the on-demand flag after checkpoint is saved."""
        self._on_demand_requested.clear()

    @property
    def is_spot_terminating(self) -> bool:
        return self._spot_listener.is_terminating

    # ── Full Checkpoint Save ─────────────────────────────────────────────

    def save_full_checkpoint(
        self,
        model_engine,
        global_step: int,
        epoch: int,
        step_in_epoch: int,
        checkpoint_manager=None,
        shard_state: Optional[Dict[str, Any]] = None,
        extra_client_state: Optional[Dict[str, Any]] = None,
        training_ops=None,
    ):
        """
        Save a FULL checkpoint with all training state.

        Includes: model weights, optimizer, scheduler, RNG states,
        shard/dataloader position, and arbitrary client state.

        Args:
            model_engine: DeepSpeed model engine
            global_step: Current global step
            epoch: Current epoch
            step_in_epoch: Step within current epoch
            checkpoint_manager: S3CheckpointManager (optional, for S3 upload)
            shard_state: Dataloader shard position dict (shard_index, offset, etc.)
            extra_client_state: Any additional state to persist
            training_ops: TrainingOps instance for observability logging
        """
        reason = self.get_checkpoint_reason()
        _print_r0(f"💾 Saving FULL checkpoint: step={global_step}, reason={reason}")

        tag = f"step_{global_step}_{reason.split()[0]}"

        # Build comprehensive client state
        client_state = {
            "epoch": epoch,
            "step": step_in_epoch,
            "global_step": global_step,
            "checkpoint_reason": reason,
            "timestamp": datetime.now().isoformat(),
            "rng_states": {
                "python": __import__("random").getstate(),
                "numpy": __import__("numpy").random.get_state(),
                "torch_cpu": torch.random.get_rng_state(),
            },
        }

        # CUDA RNG states (per-device)
        if torch.cuda.is_available():
            client_state["rng_states"]["torch_cuda"] = torch.cuda.get_rng_state_all()

        # Shard/dataloader state for exact resume
        if shard_state:
            client_state["shard_state"] = shard_state

        # Merge any extra state
        if extra_client_state:
            client_state.update(extra_client_state)

        save_start = time.time()

        try:
            if checkpoint_manager:
                checkpoint_manager.save_checkpoint(
                    model_engine, step=global_step, tag=tag, client_state=client_state,
                )
            else:
                # Direct DeepSpeed save
                os.makedirs(self.local_checkpoint_dir, exist_ok=True)
                model_engine.save_checkpoint(
                    save_dir=self.local_checkpoint_dir, tag=tag, client_state=client_state,
                )

            save_time = time.time() - save_start
            _print_r0(f"✅ Checkpoint saved in {save_time:.1f}s: {tag}")

            # Upload shard metadata to S3 separately (lightweight JSON)
            self._upload_shard_metadata(global_step, tag, client_state)

            # Log to observability
            if training_ops is not None:
                try:
                    training_ops.log_checkpoint(
                        step=global_step,
                        path=self.local_checkpoint_dir,
                        loss=extra_client_state.get("loss", 0) if extra_client_state else 0,
                        tag=tag,
                    )
                except Exception:
                    pass

        except Exception as e:
            _print_r0(f"❌ Checkpoint save failed: {e}")
            raise

        # Update timing
        self._last_checkpoint_time = time.time()
        self._last_checkpoint_step = global_step
        self.clear_on_demand()

        # Cleanup old local checkpoints
        self._cleanup_old_local_checkpoints()

        # If spot termination, also force-upload logs
        if self._spot_listener.is_terminating and self._log_uploader:
            _print_r0("Spot termination: force-uploading logs before shutdown...")
            self._log_uploader.force_upload_current()

    def _upload_shard_metadata(self, global_step: int, tag: str, client_state: dict):
        """Upload lightweight shard summary to S3 for quick inspection.

        Handles two shard state formats:
        - BinIdxDataset: flat dict with total_shards, current_shard_index, etc.
        - CurriculumDatasetV2: per-pool dict with pools.{name}.current_shard_index
        """
        if not self._s3_client or not self.s3_bucket:
            return
        if not _is_rank_0():
            return
        try:
            shard_state = client_state.get("shard_state") or {}

            meta = {
                "global_step": global_step,
                "tag": tag,
                "timestamp": client_state.get("timestamp"),
                "checkpoint_reason": client_state.get("checkpoint_reason"),
                "epoch": client_state.get("epoch"),
                "step_in_epoch": client_state.get("step"),
            }

            # Detect curriculum vs bin_idx format
            if "pools" in shard_state:
                # CurriculumDatasetV2 — per-pool state
                meta["loader_type"] = "curriculum_v2"
                meta["stage"] = shard_state.get("stage")
                meta["mode"] = shard_state.get("mode")
                meta["rank"] = shard_state.get("rank")
                meta["world_size"] = shard_state.get("world_size")
                pool_summary = {}
                for pname, pstate in shard_state["pools"].items():
                    pool_summary[pname] = {
                        "total_shards": pstate.get("total_shards"),
                        "current_shard_index": pstate.get("current_shard_index"),
                        "completed_count": pstate.get("completed_count"),
                        "remaining_count": pstate.get("remaining_count"),
                        "exhausted": pstate.get("exhausted", False),
                    }
                meta["shard_progress"] = pool_summary
            else:
                # BinIdxDataset — flat state
                meta["loader_type"] = "bin_idx"
                meta["shard_progress"] = {
                    "total_shards": shard_state.get("total_shards"),
                    "completed_count": shard_state.get("completed_count"),
                    "remaining_count": shard_state.get("remaining_count"),
                    "current_shard_index": shard_state.get("current_shard_index"),
                    "current_shard_path": shard_state.get("current_shard_path"),
                }
                meta["rank"] = shard_state.get("rank")
                meta["world_size"] = shard_state.get("world_size")

            s3_key = f"{self.s3_prefix}/metadata/{tag}_meta.json"
            self._s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=json.dumps(meta, indent=2, default=str),
                ContentType="application/json",
            )
            _print_r0(f"Shard metadata uploaded: s3://{self.s3_bucket}/{s3_key}")
        except Exception as e:
            _print_r0(f"Shard metadata upload failed (non-fatal): {e}")

    def _cleanup_old_local_checkpoints(self):
        """Keep only the last N local checkpoints to save disk space."""
        if not _is_rank_0():
            return
        try:
            ckpt_dir = Path(self.local_checkpoint_dir)
            if not ckpt_dir.exists():
                return
            # Find checkpoint subdirectories (step_NNN_*)
            ckpt_dirs = sorted(
                [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
                key=lambda d: d.stat().st_mtime,
            )
            while len(ckpt_dirs) > self.keep_last_n_local:
                old = ckpt_dirs.pop(0)
                import shutil
                shutil.rmtree(old, ignore_errors=True)
                _print_r0(f"Cleaned up old checkpoint: {old.name}")
        except Exception as e:
            _print_r0(f"Checkpoint cleanup error (non-fatal): {e}")
