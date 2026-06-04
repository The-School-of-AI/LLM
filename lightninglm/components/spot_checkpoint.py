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
    from lightninglm.components.spot_checkpoint import SpotCheckpointOrchestrator

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
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="spot-listener"
        )
        self._thread.start()
        _print_r0(
            f"Spot termination listener started (poll every {self.poll_interval}s)"
        )

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
        import urllib.error
        import urllib.request

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
                _print_r0(
                    f"SPOT TERMINATION NOTICE: action={result['action']}, time={self.termination_time}"
                )
                self.termination_event.set()
                return  # Stop polling, termination is imminent
            self._stop.wait(self.poll_interval)

    @property
    def is_terminating(self) -> bool:
        return self.termination_event.is_set()


class LogRotatorAndUploader:
    """
    Rotates local JSONL log files and uploads them to S3 in the background.

    - Rotates every `max_bytes` size
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
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="log-uploader"
        )
        self._thread.start()
        _print_r0(
            f"Log rotator/uploader started (max {self.max_bytes // (1024*1024)}MB/file, "
            f"s3={'enabled' if self.s3_bucket else 'disabled'})"
        )

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
    Central orchestrator for spot-aware checkpointing.

    Manages three checkpoint triggers:
    1. Periodic timer (default: every 1 hour)
    2. On-demand via SIGUSR1 or single Ctrl+C (training continues after save)
    3. AWS spot termination notice (saves + exits)

    Double Ctrl+C within 5 seconds = hard abort (safety valve).
    """

    def __init__(
        self,
        checkpoint_interval_seconds: int = 3600,
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "training/checkpoints",
        s3_region: str = "us-east-1",
        local_checkpoint_dir: str = "./checkpoints",
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
        self.metrics_jsonl_path = metrics_jsonl_path
        self.keep_last_n_local = keep_last_n_local

        # Internal state
        self._on_demand = threading.Event()
        self._stop_after_checkpoint = threading.Event()
        self._last_checkpoint_time = time.time()
        self._last_checkpoint_step: Optional[int] = None
        self._checkpoint_reason: Optional[str] = None
        self._last_sigint_time: float = 0.0

        # Spot listener
        self._spot_listener = SpotTerminationListener(poll_interval=spot_poll_interval)

        # Log uploader
        self._log_uploader = LogRotatorAndUploader(
            metrics_jsonl_path=metrics_jsonl_path or "",
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_region=s3_region,
            max_bytes=log_max_bytes,
        )

        # S3 client for shard metadata (rank 0 only)
        self._s3_client = None
        if _is_rank_0() and s3_bucket:
            try:
                import boto3

                self._s3_client = boto3.client("s3", region_name=s3_region)
            except Exception as e:
                _print_r0(f"Shard metadata S3 client init failed: {e}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_all(self):
        """Start spot listener, log uploader, and install signal handlers."""
        self._spot_listener.start()
        self._log_uploader.start()
        self.install_signal_handlers()
        _print_r0("Spot checkpoint orchestrator started")

    def shutdown(self):
        """Graceful shutdown of all background threads."""
        self._spot_listener.stop()
        self._log_uploader.stop()
        _print_r0("Spot checkpoint orchestrator shut down")

    @property
    def stop_requested(self) -> bool:
        """True if training should stop after the current checkpoint save."""
        return self._stop_after_checkpoint.is_set()

    def install_signal_handlers(self):
        """
        Install signal handlers on ALL ranks.

        SIGUSR1: checkpoint + STOP training (sent by run.sh on Ctrl+C)
        SIGUSR2: checkpoint + CONTINUE training (manual: pkill -USR2 -f main.py)
        SIGINT:  direct Ctrl+C fallback — checkpoint + stop
                 double press within 5s = hard abort (safety valve)
        """
        _sigusr1_logged = False
        _sigusr2_logged = False

        def _sigusr1_handler(signum, frame):
            nonlocal _sigusr1_logged
            self._on_demand.set()
            self._stop_after_checkpoint.set()
            self._checkpoint_reason = "on_demand_sigusr1"
            if not _sigusr1_logged:
                _sigusr1_logged = True
                _print_r0("SIGUSR1 received — checkpoint + stop requested")

        def _sigusr2_handler(signum, frame):
            nonlocal _sigusr2_logged
            self._on_demand.set()
            self._checkpoint_reason = "on_demand_sigusr2"
            if not _sigusr2_logged:
                _sigusr2_logged = True
                _print_r0(
                    "SIGUSR2 received — checkpoint requested (training continues)"
                )

        def _sigint_handler(signum, frame):
            now = time.time()
            if now - self._last_sigint_time < 5.0:
                # Double Ctrl+C within 5s → hard abort
                _print_r0("Double Ctrl+C — aborting immediately")
                import sys

                sys.exit(1)
            self._last_sigint_time = now
            self._on_demand.set()
            self._stop_after_checkpoint.set()
            self._checkpoint_reason = "on_demand_ctrlc"
            if int(os.environ.get("RANK", "0")) == 0:
                _print_r0(
                    "Ctrl+C received — checkpoint + stop (press again within 5s to abort)"
                )

        signal.signal(signal.SIGUSR1, _sigusr1_handler)
        signal.signal(signal.SIGUSR2, _sigusr2_handler)
        signal.signal(signal.SIGINT, _sigint_handler)

    # ------------------------------------------------------------------
    # Checkpoint trigger logic
    # ------------------------------------------------------------------

    @property
    def is_spot_terminating(self) -> bool:
        return self._spot_listener.is_terminating

    def should_checkpoint(self, global_step: int) -> bool:
        """
        Check if a checkpoint should be saved now.

        CRITICAL: The decision is made on rank 0 and broadcast to all ranks.
        This prevents deadlocks caused by timer drift between ranks — if rank 0
        enters the checkpoint path (which contains dist.barrier) but rank 3
        doesn't, the barrier deadlocks against the next allreduce.

        Returns True if any trigger fires:
        1. Spot termination detected
        2. On-demand signal received (SIGUSR1 / Ctrl+C)
        3. Periodic timer expired
        """
        # Each rank computes its local decision first
        local_decision = False
        reason = "unknown"

        # 1. Spot termination — highest priority
        if self._spot_listener.is_terminating:
            reason = "spot_termination"
            local_decision = True

        # 2. On-demand signal
        elif self._on_demand.is_set():
            reason = self._checkpoint_reason or "on_demand"
            local_decision = True

        # 3. Periodic timer (only rank 0's clock matters)
        elif _is_rank_0():
            elapsed = time.time() - self._last_checkpoint_time
            if elapsed >= self.checkpoint_interval_seconds:
                reason = "periodic"
                local_decision = True

        # Broadcast rank 0's decision to all ranks so they enter/skip together
        if dist.is_available() and dist.is_initialized():
            decision_tensor = torch.tensor(
                [1 if local_decision else 0], dtype=torch.int32
            )
            if torch.cuda.is_available():
                decision_tensor = decision_tensor.to(
                    f"cuda:{dist.get_rank() % torch.cuda.device_count()}"
                )
            dist.broadcast(decision_tensor, src=0)
            final_decision = decision_tensor.item() == 1
        else:
            final_decision = local_decision

        if final_decision:
            # Spot termination and on-demand are detected per-rank (signal-based),
            # so each rank already has the right reason. For periodic, adopt rank 0's.
            if not local_decision:
                self._checkpoint_reason = "periodic"
            else:
                self._checkpoint_reason = reason

        return final_decision

    def get_checkpoint_reason(self) -> str:
        return self._checkpoint_reason or "unknown"

    def clear_on_demand(self):
        """Clear the on-demand flag after checkpoint is saved."""
        self._on_demand.clear()

    # ------------------------------------------------------------------
    # Checkpoint save
    # ------------------------------------------------------------------

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
        Save a full checkpoint: model + optimizer + scheduler + RNG + shard state.

        This is called from the training loop when should_checkpoint() returns True.
        """
        # Capture whether this is a spot exit BEFORE clearing on-demand flag
        _is_spot_exit = self._spot_listener.is_terminating
        reason = self.get_checkpoint_reason()

        tag = f"step_{global_step}_{reason}"
        _print_r0(f"Saving checkpoint: tag={tag}, reason={reason}")

        # Build client state
        client_state = {
            "epoch": epoch,
            "step": step_in_epoch,
            "global_step": global_step,
            "checkpoint_reason": reason,
        }
        if shard_state:
            client_state["shard_state"] = shard_state
        if extra_client_state:
            client_state.update(extra_client_state)

        # Save RNG states for reproducibility
        client_state["rng_states"] = {
            "python": __import__("random").getstate(),
            "numpy": __import__("numpy").random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            client_state["rng_states"]["torch_cuda"] = torch.cuda.get_rng_state()

        # Save via checkpoint manager (handles local save + S3 upload)
        if checkpoint_manager is not None:
            checkpoint_manager.save_checkpoint(
                model_engine,
                step=global_step,
                tag=tag,
                client_state=client_state,
                urgent=_is_spot_exit,
            )
        else:
            # Fallback: local-only save
            model_engine.save_checkpoint(
                save_dir=self.local_checkpoint_dir,
                tag=tag,
                client_state=client_state,
            )

        # Upload shard metadata JSON to S3 (async, non-blocking)
        if shard_state and _is_rank_0():
            self._upload_shard_metadata_async(
                global_step, tag, shard_state, reason, epoch, step_in_epoch
            )

        # Update internal state
        self._last_checkpoint_time = time.time()
        self._last_checkpoint_step = global_step
        self.clear_on_demand()

        # Log to observability
        if training_ops is not None and _is_rank_0():
            try:
                training_ops.log_checkpoint(
                    step=global_step,
                    path=self.local_checkpoint_dir,
                    loss=extra_client_state.get("loss", 0) if extra_client_state else 0,
                    tag=tag,
                )
            except Exception:
                pass

        # Spot termination: upload metrics log and wait for S3 uploads
        if _is_spot_exit:
            _print_r0(
                "Spot termination — uploading metrics log and waiting for S3 uploads..."
            )
            self._log_uploader.force_upload_current()
            if checkpoint_manager is not None:
                checkpoint_manager.wait_for_uploads()
            _print_r0("Spot termination checkpoint complete")

        # Local cleanup (keep last N) — passes checkpoint_manager so we skip
        # directories that still have pending S3 uploads
        self._cleanup_local_checkpoints(checkpoint_manager=checkpoint_manager)

        _print_r0(f"Checkpoint saved: {tag}")

    # ------------------------------------------------------------------
    # Shard metadata upload (async — fire-and-forget)
    # ------------------------------------------------------------------

    def _upload_shard_metadata_async(
        self,
        global_step: int,
        tag: str,
        shard_state: Dict[str, Any],
        reason: str,
        epoch: int,
        step_in_epoch: int,
    ):
        """
        Upload shard metadata JSON to S3 in a background daemon thread.

        This MUST NOT block rank 0 — other ranks proceed to the next training
        step immediately after dist.barrier() in save_checkpoint(). If rank 0
        blocks here, the next allreduce will hang.
        """
        if not self._s3_client or not self.s3_bucket:
            # No S3 configured — write locally only
            self._write_shard_metadata_local(
                global_step, tag, shard_state, reason, epoch, step_in_epoch
            )
            return

        # Build the metadata payload
        metadata = self._build_shard_metadata(
            global_step, tag, shard_state, reason, epoch, step_in_epoch
        )

        # Fire-and-forget in a daemon thread
        t = threading.Thread(
            target=self._do_upload_shard_metadata,
            args=(tag, metadata),
            daemon=True,
            name=f"shard-meta-upload-{global_step}",
        )
        t.start()

        # Also write locally (instant, no blocking)
        self._write_shard_metadata_local(
            global_step, tag, shard_state, reason, epoch, step_in_epoch
        )

    def _build_shard_metadata(
        self,
        global_step: int,
        tag: str,
        shard_state: Dict[str, Any],
        reason: str,
        epoch: int,
        step_in_epoch: int,
    ) -> dict:
        rank = 0
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        loader_type = "unknown"
        if shard_state:
            # Detect loader type from shard state keys
            if any(
                k in shard_state for k in ("pool_rng_state", "pool_states", "pools")
            ):
                loader_type = "curriculum_v2"
            elif "num_shards" in shard_state:
                loader_type = "bin_idx"

        metadata = {
            "global_step": global_step,
            "tag": tag,
            "timestamp": datetime.now().isoformat(),
            "checkpoint_reason": reason,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "loader_type": loader_type,
            "rank": rank,
            "world_size": world_size,
            "shard_progress": {},
        }

        # Extract per-pool progress for curriculum_v2
        # get_shard_state() stores pool data under "pools" key
        pool_states = shard_state.get("pools") or shard_state.get("pool_states") or {}
        for pool_name, ps in pool_states.items():
            metadata["shard_progress"][pool_name] = {
                "total_shards": ps.get("total_shards", 0),
                "current_shard_index": ps.get("current_shard_index", 0),
                "completed_count": ps.get("completed_count", 0),
                "remaining_count": ps.get("remaining_count", 0),
                "sequence_offset": ps.get("sequence_offset", 0),
                "exhausted": ps.get("exhausted", False),
            }

        # Extract stage/mode if present
        if "stage" in shard_state:
            metadata["stage"] = shard_state["stage"]
        if "mode" in shard_state:
            metadata["mode"] = shard_state["mode"]

        return metadata

    def _do_upload_shard_metadata(self, tag: str, metadata: dict):
        """Background thread target: upload shard metadata JSON to S3."""
        try:
            s3_key = f"{self.s3_prefix}/{tag}/shard_metadata.json"
            self._s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=json.dumps(metadata, indent=2, default=str).encode("utf-8"),
                ContentType="application/json",
            )
            _print_r0(f"Shard metadata uploaded: s3://{self.s3_bucket}/{s3_key}")
        except Exception as e:
            _print_r0(f"Shard metadata upload failed (non-fatal): {e}")

    def _write_shard_metadata_local(
        self,
        global_step: int,
        tag: str,
        shard_state: Dict[str, Any],
        reason: str,
        epoch: int,
        step_in_epoch: int,
    ):
        """Write shard metadata JSON to local checkpoint directory."""
        try:
            metadata = self._build_shard_metadata(
                global_step, tag, shard_state, reason, epoch, step_in_epoch
            )
            local_dir = os.path.join(self.local_checkpoint_dir, tag)
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, "shard_metadata.json")
            with open(local_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            _print_r0(f"Local shard metadata write failed: {e}")

    # ------------------------------------------------------------------
    # Local checkpoint cleanup
    # ------------------------------------------------------------------

    def _cleanup_local_checkpoints(self, checkpoint_manager=None):
        """Keep only the last N local checkpoints. Runs on rank 0 only.

        Skips any checkpoint directories that have pending S3 uploads
        to avoid the race condition where cleanup deletes files before
        the upload thread reads them.
        """
        if not _is_rank_0():
            return
        try:
            ckpt_dir = self.local_checkpoint_dir
            if not os.path.isdir(ckpt_dir):
                return

            # Get set of steps with pending uploads
            pending_steps = set()
            if checkpoint_manager is not None:
                with checkpoint_manager._upload_lock:
                    pending_steps = set(checkpoint_manager.active_uploads)

            # Find all step_* directories
            dirs = []
            for name in os.listdir(ckpt_dir):
                full = os.path.join(ckpt_dir, name)
                if os.path.isdir(full) and name.startswith("step_"):
                    try:
                        step_num = int(name.split("_")[1])
                        dirs.append((step_num, full, name))
                    except (ValueError, IndexError):
                        continue

            dirs.sort(key=lambda x: x[0])

            # Remove oldest, keep last N — but never delete dirs with pending uploads
            if len(dirs) > self.keep_last_n_local:
                to_remove = dirs[: len(dirs) - self.keep_last_n_local]
                for step_num, path, name in to_remove:
                    if step_num in pending_steps:
                        _print_r0(f"Skipping cleanup of {name} — upload still pending")
                        continue
                    try:
                        import shutil

                        shutil.rmtree(path, ignore_errors=True)
                        _print_r0(f"Cleaned up local checkpoint: {name}")
                    except Exception:
                        pass
        except Exception as e:
            _print_r0(f"Local checkpoint cleanup error: {e}")
