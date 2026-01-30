"""
Dashboard Logger
================

Lightweight metrics publisher for the Team 7 MoE dashboard.
Uses Redis by default and is a no-op when disabled.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Optional


class DashboardLogger:
    """
    Minimal logger for pushing dashboard metrics.

    Usage:
        logger = DashboardLogger(enabled=True, redis_url="redis://localhost:6379")
        logger.log(metrics_dict)
    """

    def __init__(
        self,
        enabled: bool = False,
        backend: str = "redis",
        redis_url: str = "redis://localhost:6379",
        redis_key: str = "moe_metrics",
    ) -> None:
        self.enabled = enabled
        self.backend = backend
        self.redis_url = redis_url
        self.redis_key = redis_key
        self._client = None

    def _ensure_client(self) -> None:
        if not self.enabled:
            return
        if self.backend != "redis":
            raise ValueError(f"Unsupported backend: {self.backend}")
        if self._client is None:
            try:
                import redis
            except ModuleNotFoundError:
                self.enabled = False
                print(
                    "[DashboardLogger] Missing optional 'redis' package. "
                    "Install with 'pip install redis' to enable dashboard logging. "
                    "Continuing without dashboard output.",
                    file=sys.stderr,
                )
                return
            self._client = redis.from_url(self.redis_url)

    def log(self, metrics: Dict[str, Any]) -> None:
        """Push metrics to the dashboard backend."""
        if not self.enabled:
            return
        self._ensure_client()
        if not self.enabled or self._client is None:
            return
        payload = json.dumps(metrics, default=str)
        self._client.set(self.redis_key, payload)

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
