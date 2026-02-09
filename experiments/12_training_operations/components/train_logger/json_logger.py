
import json
import time
import threading
import queue
from pathlib import Path
from datetime import datetime
import numpy as np
import os

class JSONLogger:
    """
    A high-performance, non-blocking structured logger for training runs.
    Writes JSONL files to local NVMe storage for sidecar ingestion.
    """
    def __init__(self, base_dir: str, run_id: str, rank: int = 0, buffer_size: int = 100, default_context: dict = None):
        self.base_dir = Path(base_dir)
        self.run_id = run_id
        self.rank = rank
        self.default_context = default_context or {}
        
        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # In distributed training, each rank writes its own file
        # Even if on shared storage, this prevents corruption
        self.log_file = self.base_dir / f"{self.run_id}_rank_{self.rank}.jsonl"
        self.buffer_size = buffer_size
        
        # Async writing setup
        self.queue = queue.Queue()
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

        payload = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "metrics": metrics,
            "context": merged_context
        }
        self.queue.put(payload)

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
                if len(self.buffer) >= self.buffer_size or (self.buffer and self.queue.empty()):
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
            self.buffer = []
        except Exception as e:
            print(f"FAILED TO WRITE LOGS: {e}")

    def _json_serializer(self, obj):
        """
        Handle non-JSON types like Numpy arrays.
        """
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
        print(f"✓ JSONLogger closed. Logs saved to {self.log_file}")
