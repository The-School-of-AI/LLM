import json
import os
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class JSONLogger:
    """
    A high-performance, non-blocking structured logger for training runs.
    Writes JSONL files to local NVMe storage for sidecar ingestion.
    """

    def __init__(
        self,
        base_dir: str,
        run_id: str,
        rank: int = 0,
        buffer_size: int = 100,
        default_context: dict = None,
        queue_maxsize: int = 0,
        fsync_on_flush: bool = False,
    ):
        self.base_dir = Path(base_dir)
        self.run_id = run_id
        self.rank = rank
        self.default_context = default_context or {}
        self.host = os.environ.get("HOSTNAME") or os.uname().nodename
        self.fsync_on_flush = fsync_on_flush
        self.dropped = 0

        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # In distributed training, each rank writes its own file
        # Even if on shared storage, this prevents corruption
        self.log_file = self.base_dir / f"{self.run_id}_rank_{self.rank}.jsonl"
        self.buffer_size = buffer_size

        # Async writing setup
        self.queue = (
            queue.Queue(maxsize=queue_maxsize)
            if queue_maxsize and queue_maxsize > 0
            else queue.Queue()
        )
        self.running = True

        # Buffer for batch writing
        self.buffer = []

        # Start worker thread
        self.worker_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.worker_thread.start()

        print(f"✓ JSONLogger initialized. Writing to: {self.log_file}")

    def log_step(self, step: int, metrics: dict, context: dict = None):
        """
        Log a training step. Thread-safe and non-blocking.
        """
        # Merge default context with per-step context
        merged_context = self.default_context.copy()
        if context:
            merged_context.update(context)

        ts = (
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        payload = {
            "timestamp": ts,
            "run_id": self.run_id,
            "rank": self.rank,
            "host": self.host,
            "step": step,
            "metrics": metrics,
            "context": merged_context,
        }
        try:
            self.queue.put(payload, block=False)
        except queue.Full:
            self.dropped += 1

    def _writer_loop(self):
        """
        Background thread to batch write logs to disk.
        """
        while self.running or not self.queue.empty():
            try:
                # Wait for data with timeout to allow periodic flushing
                try:
                    data = self.queue.get(timeout=1.0)
                    self.buffer.append(data)
                except queue.Empty:
                    pass

                # Flush if buffer full or timeout reached
                if len(self.buffer) >= self.buffer_size or (
                    self.buffer and self.queue.empty()
                ):
                    self._flush()

            except Exception as e:
                print(f"CRITICAL LOGGER ERROR: {e}")

    def _flush(self):
        """
        Actual I/O operation.
        """
        try:
            with open(self.log_file, "a") as f:
                for entry in self.buffer:
                    # Use a custom encoder to handle numpy types
                    f.write(json.dumps(entry, default=self._json_serializer) + "\n")
                if self.fsync_on_flush:
                    f.flush()
                    os.fsync(f.fileno())
            self.buffer = []
        except Exception as e:
            print(f"FAILED TO WRITE LOGS: {e}")

    def _json_serializer(self, obj):
        """
        Handle non-JSON types like Numpy arrays.
        """
        try:
            if hasattr(obj, "detach") and hasattr(obj, "cpu"):
                t = obj.detach().cpu()
                if (
                    hasattr(t, "numel")
                    and callable(t.numel)
                    and t.numel() == 1
                    and hasattr(t, "item")
                ):
                    return float(t.item())
                if hasattr(t, "tolist"):
                    return t.tolist()
        except Exception:
            pass
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)
        return str(obj)

    def close(self):
        """
        Graceful shutdown.
        """
        self.running = False
        self.worker_thread.join(timeout=5.0)
        # Flush any remaining items
        if self.buffer:
            self._flush()
        if self.dropped:
            print(f"! JSONLogger dropped {self.dropped} log entries due to full queue")
        print(f"✓ JSONLogger closed. Logs saved to {self.log_file}")
