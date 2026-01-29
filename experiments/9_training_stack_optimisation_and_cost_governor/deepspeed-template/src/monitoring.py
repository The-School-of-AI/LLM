"""
Simple metrics logger for training instrumentation.
"""

import json
from datetime import datetime, timezone


class MetricsLogger:
    def __init__(self, path, stage=1):
        self.path = path
        self.stage = stage

    def log(self, payload):
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "stage": self.stage,
        }
        record.update(payload)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
