"""
TrainingOps — single entry point for P12 observability on the training instance.

Usage:
    from components.training_ops import TrainingOps

    ops = TrainingOps(run_id="run_001", rank=0)

    # In the training loop:
    ops.log_step(step, metrics={"loss": 0.5, "lr": 3e-4}, context={"epoch": 1})

    # After saving a checkpoint:
    ops.log_checkpoint(step=1000, path="/mnt/checkpoints/ckpt_1000.pt",
                       s3_key="s3://bucket/ckpt_1000.pt", loss=0.32)

    # End of training:
    ops.shutdown()

Behind the scenes this starts and manages:
    - JSONLogger          (training JSONL → Vector → ClickHouse)
    - SystemMetricsCollector (system JSONL → Vector → ClickHouse)
    - MetricsServer       (live JSON API on :8000 for dashboard/watchdog)
    - CheckpointRegistry  (ClickHouse-backed checkpoint governance)

Preflight checks:
    - Vector process running  → FATAL if missing
    - ClickHouse reachable    → WARN if down (Vector buffers until recovery)
"""

import base64
import json
import os
import shutil
import ssl
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

from components.train_logger.json_logger import JSONLogger
from components.system_metrics.collector import SystemMetricsCollector
from components.metrics_server import MetricsServer
from components.checkpoint_registry import CheckpointRegistry


class TrainingOps:
    """
    Facade that boots all P12 backend services and exposes a minimal API
    for the training team.

    Parameters
    ----------
    run_id : str
        Unique identifier for this training run.
    rank : int
        Distributed-training rank (default 0).
    log_dir : str
        Directory for JSONL files that Vector tails.
    metrics_port : int
        Port for the live metrics HTTP server.
    clickhouse_url : str | None
        ClickHouse HTTP(S) endpoint.  Falls back to CLICKHOUSE_HTTPS_ENDPOINT
        or CLICKHOUSE_HTTP_ENDPOINT env var.
    clickhouse_user : str | None
        ClickHouse user.  Falls back to CLICKHOUSE_USER env var.
    clickhouse_password : str | None
        ClickHouse password.  Falls back to CLICKHOUSE_PASSWORD env var.
    clickhouse_ca_cert : str | None
        Path to CA cert for TLS.  Falls back to CLICKHOUSE_CA_CERT env var.
    system_metrics_interval : float
        How often (seconds) to collect system metrics.
    default_context : dict | None
        Default context fields merged into every log_step call.
    """

    def __init__(
        self,
        run_id: str,
        rank: int = 0,
        log_dir: str = "/tmp/training_logs",
        metrics_port: int = 8000,
        clickhouse_url: str | None = None,
        clickhouse_user: str | None = None,
        clickhouse_password: str | None = None,
        clickhouse_ca_cert: str | None = None,
        system_metrics_interval: float = 5.0,
        default_context: dict | None = None,
        skip_vector_check: bool = False,
    ):
        self.run_id = run_id
        self.rank = rank
        self.log_dir = log_dir
        self._clickhouse_url = (
            clickhouse_url
            or os.environ.get("CLICKHOUSE_HTTPS_ENDPOINT")
            or os.environ.get("CLICKHOUSE_HTTP_ENDPOINT", "http://localhost:8123")
        )
        self._ch_user = clickhouse_user or os.environ.get("CLICKHOUSE_USER", "")
        self._ch_password = clickhouse_password or os.environ.get("CLICKHOUSE_PASSWORD", "")
        self._ch_ca_cert = clickhouse_ca_cert if clickhouse_ca_cert is not None else os.environ.get("CLICKHOUSE_CA_CERT", "")

        # Build auth header + TLS context for preflight check
        self._auth_header = ""
        if self._ch_user:
            creds = base64.b64encode(f"{self._ch_user}:{self._ch_password}".encode()).decode()
            self._auth_header = f"Basic {creds}"
        self._ssl_ctx = None
        if self._clickhouse_url.startswith("https"):
            self._ssl_ctx = ssl.create_default_context()
            if self._ch_ca_cert and os.path.isfile(self._ch_ca_cert):
                self._ssl_ctx.load_verify_locations(self._ch_ca_cert)
            else:
                self._ssl_ctx.check_hostname = False
                self._ssl_ctx.verify_mode = ssl.CERT_NONE

        print("=" * 60)
        print(f"  P12 TrainingOps — initializing for run '{run_id}'")
        print("=" * 60)

        # ---- Preflight checks ----
        if skip_vector_check:
            print("⚠ Preflight: Vector check skipped (skip_vector_check=True)")
        else:
            self._check_vector()      # fatal
        self._check_clickhouse()      # warn-only

        # ---- JSONLogger ----
        self.logger = JSONLogger(
            base_dir=log_dir,
            run_id=run_id,
            rank=rank,
            default_context=default_context or {},
        )

        # ---- SystemMetricsCollector ----
        self.system_collector = SystemMetricsCollector(
            log_dir=log_dir,
            run_id=run_id,
            rank=rank,
            interval=system_metrics_interval,
        )

        # ---- MetricsServer (live HTTP API) ----
        self.metrics_server = MetricsServer()
        self.metrics_server.config["training"]["metrics_port"] = metrics_port
        self.metrics_server.start(system_collector=self.system_collector)

        # ---- CheckpointRegistry (ClickHouse-backed) ----
        self.checkpoint_registry = CheckpointRegistry(
            clickhouse_url=self._clickhouse_url,
            user=self._ch_user,
            password=self._ch_password,
            ca_cert=self._ch_ca_cert,
        )

        print("=" * 60)
        print("  P12 TrainingOps — ready")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Preflight checks
    # ------------------------------------------------------------------

    def _check_vector(self):
        """Verify Vector is running. FATAL if not."""
        # Check for vector binary first
        vector_bin = shutil.which("vector")
        if vector_bin is None:
            # Also check common install locations
            for path in ["/usr/bin/vector", "/usr/local/bin/vector"]:
                if os.path.isfile(path):
                    vector_bin = path
                    break

        # Check if process is running
        try:
            result = subprocess.run(
                ["pgrep", "-x", "vector"],
                capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                pids = result.stdout.decode().strip().split("\n")
                print(f"✓ Preflight: Vector is running (PID: {', '.join(pids)})")
                return
        except Exception:
            pass

        # Not running — fatal
        print("=" * 60)
        print("  FATAL: Vector sidecar is not running!")
        print()
        print("  Training logs will not be shipped to ClickHouse.")
        print("  Start Vector before launching training:")
        print()
        print("    vector --config /path/to/vector.toml")
        print("=" * 60)
        sys.exit(1)

    def _check_clickhouse(self):
        """Verify ClickHouse is reachable (with auth). WARN if not (Vector buffers)."""
        try:
            url = f"{self._clickhouse_url}/?query={urllib.request.quote('SELECT 1')}"
            req = urllib.request.Request(url, method="GET")
            if self._auth_header:
                req.add_header("Authorization", self._auth_header)
            ctx = self._ssl_ctx if self._ssl_ctx else None
            with urllib.request.urlopen(req, timeout=5, context=ctx) as resp:
                resp.read()
            print(f"✓ Preflight: ClickHouse reachable at {self._clickhouse_url}")
        except Exception as e:
            print(f"⚠ Preflight: ClickHouse not reachable at {self._clickhouse_url}")
            print(f"  ({e})")
            print(f"  Vector will buffer data until ClickHouse recovers.")

    # ------------------------------------------------------------------
    # Training loop API
    # ------------------------------------------------------------------

    def log_step(
        self,
        step: int,
        metrics: dict,
        context: dict | None = None,
    ):
        """
        Log a training step.

        Writes to:
          1. JSONLogger → JSONL on disk → Vector → ClickHouse (logs + metric_points)
          2. MetricsServer in-memory gauges → live HTTP API (dashboard/watchdog)

        Parameters
        ----------
        step : int
            Global training step.
        metrics : dict
            Scalar metrics, e.g. {"loss": 0.5, "lr": 3e-4, "tokens_per_second": 12000}.
        context : dict | None
            Optional per-step context, e.g. {"epoch": 1, "phase": "warmup"}.
        """
        # 1. Durable log (JSONL → Vector → ClickHouse)
        self.logger.log_step(step=step, metrics=metrics, context=context)

        # 2. Live gauges (in-memory → HTTP API)
        loss = metrics.get("loss") or metrics.get("training_loss")
        lr = metrics.get("lr") or metrics.get("learning_rate", 0)
        tps = metrics.get("tokens_per_second") or metrics.get("tok_sec")
        grad_norm = metrics.get("gradient_norm") or metrics.get("grad_norm")

        if loss is not None:
            self.metrics_server.update_training_metrics(
                loss=loss, lr=lr, step=step, grad_norm=grad_norm,
            )
        if tps is not None:
            self.metrics_server.update_throughput(tps)

    def log_checkpoint(
        self,
        step: int,
        path: str,
        s3_key: str | None = None,
        loss: float = 0.0,
        tag: str = "temporary",
        duration_s: float = 0.0,
        size_bytes: int = 0,
        metadata: dict | None = None,
    ):
        """
        Record a checkpoint after it has been saved.

        The canonical path stored is ``s3_key`` if provided, otherwise ``path``
        (local filesystem).  Both are recorded in metadata for traceability.

        Data flow (two paths to ClickHouse ``checkpoints`` table):

          1. **Durable (guaranteed):** JSONL → Vector ``to_checkpoints``
             transform → ClickHouse ``checkpoints`` table.  Works even if
             ClickHouse is temporarily unreachable — Vector buffers & retries.
          2. **Fast path (best-effort):** Direct HTTP INSERT via
             ``CheckpointRegistry``.  Gives immediate query-ability but may
             fail if ClickHouse is down.  The durable path catches up.

        Additionally:
          3. The same JSONL line also flows through ``to_raw_logs`` →
             ClickHouse ``logs`` table (audit trail).
          4. MetricsServer counters updated (live dashboard).

        Parameters
        ----------
        step : int
            Training step at which the checkpoint was saved.
        path : str
            Local filesystem path where the checkpoint was saved.
        s3_key : str | None
            S3 URI if the checkpoint was uploaded. Preferred canonical path.
        loss : float
            Loss value at checkpoint time.
        tag : str
            Governance tag: "temporary", "growth", "lora", "release_candidate".
        duration_s : float
            How long the save took (seconds).
        size_bytes : int
            Checkpoint file size.
        metadata : dict | None
            Arbitrary extra metadata.
        """
        canonical_path = s3_key or path

        # 1. Durable path: JSONL → Vector → ClickHouse (logs + checkpoints tables)
        #    Vector's to_checkpoints transform filters on context.event == "checkpoint_saved"
        #    and maps the fields to the checkpoints table schema.
        ckpt_metrics = {
            "checkpoint_step": step,
            "checkpoint_loss": loss,
            "checkpoint_duration_s": duration_s,
            "checkpoint_size_bytes": size_bytes,
        }
        ckpt_context = {
            "event": "checkpoint_saved",
            "path": path,
            "s3_key": s3_key or "",
            "tag": tag,
            "canonical_path": canonical_path,
        }
        if metadata:
            ckpt_context["metadata"] = metadata
        self.logger.log_step(step=step, metrics=ckpt_metrics, context=ckpt_context)

        # 2. Fast path (best-effort): direct INSERT for immediate query-ability.
        #    If this fails, the durable JSONL path above guarantees delivery.
        try:
            self.checkpoint_registry.register_checkpoint(
                run_id=self.run_id,
                step=step,
                s3_key=canonical_path,
                loss=loss,
                tag=tag,
                duration_s=duration_s,
                size_bytes=size_bytes,
                metadata=metadata,
            )
        except Exception as e:
            print(f"⚠ TrainingOps: direct checkpoint registry insert failed: {e}")
            print(f"  (durable path: JSONL → Vector will deliver it)")

        # 3. Live metrics
        self.metrics_server.record_checkpoint(duration_s, success=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self):
        """Gracefully stop all P12 services."""
        print("\nP12 TrainingOps — shutting down...")
        self.logger.close()
        self.metrics_server.stop()
        # system_collector is stopped by metrics_server.stop()
        print("P12 TrainingOps — shutdown complete.")
