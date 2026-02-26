"""
System Metrics Collector for P12 Training Operations.

Collects host-level metrics (CPU, memory, disk, network, GPU) and writes them
as JSONL to disk so Vector can pick them up and push to ClickHouse via the
existing metric_points pipeline.

The output format matches what the training JSONLogger produces:
    {"timestamp": "...", "run_id": "...", "host": "...", "rank": 0, "step": <current>,
     "metrics": {"sys.cpu_percent": 42.1, ...}, "context": {"collector": "system"}}

The ``step`` field is updated via ``set_step()`` from the training loop, so system
metric samples are attributed to the correct training step.

This means the existing Vector transforms (parse_json → to_metric_points → ClickHouse)
work without any changes for the scalar fan-out path.
"""

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


class SystemMetricsCollector:
    """
    Periodically collects system metrics and writes them as JSONL for Vector.

    Parameters
    ----------
    log_dir : str
        Directory where the JSONL file is written.  Should be under the same
        tree that Vector tails (e.g. /tmp/training_logs/).
    run_id : str
        Run identifier — ties system metrics to the training run.
    rank : int
        Distributed-training rank (default 0).
    interval : float
        Collection interval in seconds (default 5).
    gpu : bool
        Attempt to collect NVIDIA GPU metrics via pynvml (default True).
        Silently disabled if pynvml is not installed or no GPUs are found.
    disk_paths : list[str] | None
        Mount points to monitor for disk usage.  Defaults to ["/"].
    net_interfaces : list[str] | None
        Network interfaces to monitor.  None = all physical interfaces.
    """

    def __init__(
        self,
        log_dir: str = "/tmp/training_logs",
        run_id: str = "unknown",
        rank: int = 0,
        interval: float = 5.0,
        gpu: bool = True,
        disk_paths: list | None = None,
        net_interfaces: list | None = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"system_metrics_rank_{rank}.jsonl"

        self.run_id = run_id
        self.rank = rank
        self.host = os.environ.get("HOSTNAME") or os.uname().nodename
        self.interval = interval
        self.disk_paths = disk_paths or ["/"]
        self.net_interfaces = net_interfaces

        self._running = False
        self._thread: threading.Thread | None = None

        # Thread-safe training step tracker (updated by training loop)
        self._current_step = 0
        self._step_lock = threading.Lock()

        # GPU support (optional)
        self._gpu_available = False
        self._gpu_count = 0
        if gpu:
            self._init_gpu()

        # Baseline for delta counters (network bytes)
        self._prev_net = self._snapshot_net()

        print("✓ SystemMetricsCollector initialized")
        print(f"  Log file : {self.log_file}")
        print(f"  Interval : {self.interval}s")
        print(f"  GPU      : {self._gpu_available} ({self._gpu_count} devices)")

    # ------------------------------------------------------------------
    # GPU helpers
    # ------------------------------------------------------------------

    def _init_gpu(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            self._gpu_count = pynvml.nvmlDeviceGetCount()
            self._gpu_available = self._gpu_count > 0
        except Exception:
            self._gpu_available = False
            self._gpu_count = 0

    def _collect_gpu_metrics(self) -> dict:
        """Return per-GPU metrics keyed as sys.gpu.<idx>.<metric>."""
        metrics = {}
        if not self._gpu_available:
            return metrics
        try:
            import pynvml

            for i in range(self._gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
                except pynvml.NVMLError:
                    power = 0.0

                prefix = f"sys.gpu.{i}"
                metrics[f"{prefix}.util_percent"] = float(util.gpu)
                metrics[f"{prefix}.mem_used_bytes"] = float(mem.used)
                metrics[f"{prefix}.mem_total_bytes"] = float(mem.total)
                metrics[f"{prefix}.mem_percent"] = (
                    round(mem.used / mem.total * 100, 2) if mem.total else 0.0
                )
                metrics[f"{prefix}.temperature_c"] = float(temp)
                metrics[f"{prefix}.power_w"] = round(power, 2)
        except Exception:
            pass
        return metrics

    # ------------------------------------------------------------------
    # Network helpers
    # ------------------------------------------------------------------

    def _snapshot_net(self) -> dict:
        """Return {iface: (bytes_sent, bytes_recv)} for tracked interfaces."""
        counters = psutil.net_io_counters(pernic=True)
        result = {}
        for iface, stats in counters.items():
            if self.net_interfaces is not None and iface not in self.net_interfaces:
                continue
            if iface.startswith("lo"):
                continue
            result[iface] = (stats.bytes_sent, stats.bytes_recv)
        return result

    def _collect_net_metrics(self) -> dict:
        """Return per-interface bytes/sec deltas."""
        current = self._snapshot_net()
        metrics = {}
        for iface, (sent, recv) in current.items():
            prev_sent, prev_recv = self._prev_net.get(iface, (sent, recv))
            delta_sent = max(0, sent - prev_sent)
            delta_recv = max(0, recv - prev_recv)
            metrics[f"sys.net.{iface}.sent_bytes_per_s"] = round(
                delta_sent / self.interval, 2
            )
            metrics[f"sys.net.{iface}.recv_bytes_per_s"] = round(
                delta_recv / self.interval, 2
            )
        self._prev_net = current
        return metrics

    # ------------------------------------------------------------------
    # Core collection
    # ------------------------------------------------------------------

    def collect_once(self) -> dict:
        """Collect all system metrics and return as a flat dict."""
        metrics: dict = {}

        # CPU
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_freq = psutil.cpu_freq()
        load1, load5, load15 = os.getloadavg()
        metrics["sys.cpu_percent"] = cpu_percent
        metrics["sys.cpu_count"] = float(psutil.cpu_count())
        metrics["sys.load_1m"] = round(load1, 2)
        metrics["sys.load_5m"] = round(load5, 2)
        metrics["sys.load_15m"] = round(load15, 2)
        if cpu_freq:
            metrics["sys.cpu_freq_mhz"] = round(cpu_freq.current, 1)

        # Memory
        vm = psutil.virtual_memory()
        metrics["sys.mem_total_bytes"] = float(vm.total)
        metrics["sys.mem_used_bytes"] = float(vm.used)
        metrics["sys.mem_available_bytes"] = float(vm.available)
        metrics["sys.mem_percent"] = vm.percent

        swap = psutil.swap_memory()
        metrics["sys.swap_used_bytes"] = float(swap.used)
        metrics["sys.swap_percent"] = swap.percent

        # Disk
        for path in self.disk_paths:
            try:
                usage = psutil.disk_usage(path)
                tag = path.replace("/", "_").strip("_") or "root"
                metrics[f"sys.disk.{tag}.total_bytes"] = float(usage.total)
                metrics[f"sys.disk.{tag}.used_bytes"] = float(usage.used)
                metrics[f"sys.disk.{tag}.free_bytes"] = float(usage.free)
                metrics[f"sys.disk.{tag}.percent"] = usage.percent
            except OSError:
                pass

        # Network
        metrics.update(self._collect_net_metrics())

        # GPU
        metrics.update(self._collect_gpu_metrics())

        return metrics

    def _build_payload(self, metrics: dict) -> dict:
        """Build a JSONL-compatible payload matching the training logger format."""
        ts = (
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        with self._step_lock:
            step = self._current_step
        return {
            "timestamp": ts,
            "run_id": self.run_id,
            "rank": self.rank,
            "host": self.host,
            "step": step,
            "metrics": metrics,
            "context": {"collector": "system"},
        }

    def _write_payload(self, payload: dict):
        """Append a single JSON line to the log file."""
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            print(f"SystemMetricsCollector write error: {e}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _loop(self):
        # Prime the CPU percent counter (first call always returns 0)
        psutil.cpu_percent(interval=None)
        while self._running:
            try:
                metrics = self.collect_once()
                payload = self._build_payload(metrics)
                self._write_payload(payload)
            except Exception as e:
                print(f"SystemMetricsCollector error: {e}")
            time.sleep(self.interval)

    def set_step(self, step: int):
        """Update the current training step (called from the training loop)."""
        with self._step_lock:
            self._current_step = step

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"✓ SystemMetricsCollector started (every {self.interval}s)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.interval + 2)
        print("✓ SystemMetricsCollector stopped")
