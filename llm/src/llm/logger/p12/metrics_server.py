"""Metrics server for P12 POC — Custom Implementation (no Prometheus)"""

import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import psutil
import yaml

# ---------------------------------------------------------------------------
# In-memory metric store
# ---------------------------------------------------------------------------


class _Gauge:
    """Thread-safe gauge (last-value) metric."""

    __slots__ = ("name", "description", "_value", "_lock")

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = threading.Lock()

    def set(self, value: float):
        with self._lock:
            self._value = float(value)

    def get(self) -> float:
        with self._lock:
            return self._value


class _Counter:
    """Thread-safe monotonically-increasing counter."""

    __slots__ = ("name", "description", "_value", "_lock")

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0
        self._lock = threading.Lock()

    def inc(self, amount: int = 1):
        with self._lock:
            self._value += amount

    def get(self) -> int:
        with self._lock:
            return self._value


class _InfoMetric:
    """Thread-safe key/value info metric."""

    __slots__ = ("name", "description", "_data", "_lock")

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._data: dict = {}
        self._lock = threading.Lock()

    def info(self, data: dict):
        with self._lock:
            self._data = dict(data)

    def get(self) -> dict:
        with self._lock:
            return dict(self._data)


# ---------------------------------------------------------------------------
# Time-series ring buffer for recent metric history
# ---------------------------------------------------------------------------


class _MetricHistory:
    """Fixed-size ring buffer that stores (timestamp, value) tuples."""

    def __init__(self, maxlen: int = 7200):
        self._buf: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, ts: float, value: float):
        with self._lock:
            self._buf.append((ts, value))

    def query(self, since: float = 0.0):
        """Return all points with timestamp >= *since*."""
        with self._lock:
            return [(t, v) for t, v in self._buf if t >= since]


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------


class _MetricsHandler(BaseHTTPRequestHandler):
    """
    Endpoints:
        GET /metrics          — current snapshot (JSON)
        GET /query?metric=X   — single metric current value
        GET /history?metric=X&since=T — time-series for metric X since epoch T
        GET /health           — liveness probe
    """

    # Suppress default stderr logging per request
    def log_message(self, format, *args):
        pass

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        server: MetricsServer = self.server._metrics_server  # type: ignore[attr-defined]

        if path == "/health":
            self._send_json({"status": "ok"})

        elif path == "/metrics":
            self._send_json(server.snapshot())

        elif path == "/query":
            metric_name = params.get("metric", [None])[0]
            if metric_name is None:
                self._send_json({"error": "missing 'metric' param"}, 400)
                return
            value = server.get_metric_value(metric_name)
            if value is None:
                self._send_json({"error": f"unknown metric '{metric_name}'"}, 404)
                return
            self._send_json({"metric": metric_name, "value": value})

        elif path == "/history":
            metric_name = params.get("metric", [None])[0]
            since = float(params.get("since", [0])[0])
            if metric_name is None:
                self._send_json({"error": "missing 'metric' param"}, 400)
                return
            points = server.get_metric_history(metric_name, since)
            if points is None:
                self._send_json({"error": f"unknown metric '{metric_name}'"}, 404)
                return
            self._send_json(
                {
                    "metric": metric_name,
                    "data": [{"timestamp": t, "value": v} for t, v in points],
                }
            )

        else:
            self._send_json({"error": "not found"}, 404)


# ---------------------------------------------------------------------------
# MetricsServer — drop-in replacement (same public API)
# ---------------------------------------------------------------------------


class MetricsServer:
    _DEFAULTS = {
        "training": {
            "metrics_port": 8000,
        }
    }

    def __init__(self, config_path=None):
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f) or {}
        else:
            self.config = {}
        # Merge defaults for any missing keys
        for section, values in self._DEFAULTS.items():
            self.config.setdefault(section, {})
            for k, v in values.items():
                self.config[section].setdefault(k, v)

        # Gauges
        self.loss = _Gauge("training_loss", "Training loss")
        self.learning_rate = _Gauge("learning_rate", "Learning rate")
        self.throughput = _Gauge("tokens_per_second", "Training throughput")
        self.global_step = _Gauge("global_step", "Training step")
        self.gradient_norm = _Gauge("gradient_norm", "Gradient norm")
        self.cpu_usage = _Gauge("cpu_usage_percent", "CPU usage")
        self.memory_usage = _Gauge("memory_usage_percent", "Memory usage")
        self.last_checkpoint_time = _Gauge(
            "last_checkpoint_timestamp", "Last checkpoint time"
        )

        # Counters
        self.checkpoint_saves = _Counter("checkpoint_saves_total", "Checkpoints saved")
        self.checkpoint_failures = _Counter(
            "checkpoint_failures_total", "Checkpoint failures"
        )

        # Info
        self.training_status = _InfoMetric("training_status", "Training status")

        # Registry for easy iteration
        self._gauges: dict[str, _Gauge] = {
            g.name: g
            for g in [
                self.loss,
                self.learning_rate,
                self.throughput,
                self.global_step,
                self.gradient_norm,
                self.cpu_usage,
                self.memory_usage,
                self.last_checkpoint_time,
            ]
        }
        self._counters: dict[str, _Counter] = {
            c.name: c for c in [self.checkpoint_saves, self.checkpoint_failures]
        }
        self._infos: dict[str, _InfoMetric] = {
            self.training_status.name: self.training_status
        }

        # Time-series history (ring buffers)
        self._history: dict[str, _MetricHistory] = {
            name: _MetricHistory() for name in self._gauges
        }

        self.running = False
        self.collection_thread = None
        self._httpd = None
        self._http_thread = None
        print("✓ MetricsServer initialized")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, system_collector=None):
        """
        Start the HTTP server and system metrics collection.

        Parameters
        ----------
        system_collector : SystemMetricsCollector | None
            If provided, the collector is started alongside this server.
            Pass one in to have system metrics written to disk for Vector.
        """
        port = self.config["training"]["metrics_port"]

        self._httpd = HTTPServer(("0.0.0.0", port), _MetricsHandler)
        self._httpd._metrics_server = self  # type: ignore[attr-defined]
        self._http_thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True
        )
        self._http_thread.start()
        print(f"✓ Metrics server started on port {port}")

        self.running = True
        self.collection_thread = threading.Thread(
            target=self._collect_system_metrics, daemon=True
        )
        self.collection_thread.start()

        # Optionally start the disk-writing system collector (for ClickHouse)
        self._system_collector = system_collector
        if self._system_collector is not None:
            self._system_collector.start()

    def _collect_system_metrics(self):
        while self.running:
            try:
                self.cpu_usage.set(psutil.cpu_percent(interval=1))
                self.memory_usage.set(psutil.virtual_memory().percent)
            except Exception:
                pass
            time.sleep(5)

    def stop(self):
        self.running = False
        if self._system_collector is not None:
            self._system_collector.stop()
        if self._httpd:
            self._httpd.shutdown()
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        print("✓ Metrics server stopped")

    # ------------------------------------------------------------------
    # Update helpers (same signatures as before)
    # ------------------------------------------------------------------

    def update_training_metrics(self, loss, lr, step, tokens=None, grad_norm=None):
        now = time.time()
        self.loss.set(loss)
        self.learning_rate.set(lr)
        self.global_step.set(step)
        if grad_norm:
            self.gradient_norm.set(grad_norm)
        # Record history
        self._history["training_loss"].append(now, float(loss))
        self._history["learning_rate"].append(now, float(lr))
        self._history["global_step"].append(now, float(step))
        if grad_norm:
            self._history["gradient_norm"].append(now, float(grad_norm))

    def update_throughput(self, tps):
        self.throughput.set(tps)
        self._history["tokens_per_second"].append(time.time(), float(tps))

    def record_checkpoint(self, duration, success=True):
        if success:
            self.checkpoint_saves.inc()
            self.last_checkpoint_time.set(time.time())
        else:
            self.checkpoint_failures.inc()

    def update_training_status(self, status, message=""):
        self.training_status.info({"status": status, "message": message})

    # ------------------------------------------------------------------
    # Query helpers (used by HTTP handler and watchdog)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a full snapshot of all current metric values."""
        data: dict = {"gauges": {}, "counters": {}, "info": {}}
        for name, g in self._gauges.items():
            data["gauges"][name] = g.get()
        for name, c in self._counters.items():
            data["counters"][name] = c.get()
        for name, i in self._infos.items():
            data["info"][name] = i.get()
        return data

    def get_metric_value(self, name: str):
        """Return the current value of a single metric, or None."""
        if name in self._gauges:
            return self._gauges[name].get()
        if name in self._counters:
            return self._counters[name].get()
        return None

    def get_metric_history(self, name: str, since: float = 0.0):
        """Return time-series points for a gauge since *since*, or None."""
        hist = self._history.get(name)
        if hist is None:
            return None
        return hist.query(since)


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_metrics_server = None


def get_metrics_server(config_path=None):
    global _metrics_server
    if _metrics_server is None:
        _metrics_server = MetricsServer(config_path)
    return _metrics_server
