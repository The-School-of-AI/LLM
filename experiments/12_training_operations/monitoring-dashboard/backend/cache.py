from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field


@dataclass
class Point:
    step: int
    value: float
    timestamp: float


@dataclass
class RunInfo:
    run_id: str
    start_time: float
    last_event_time: float
    latest_step: int
    is_active: bool


RunData = dict[str, list[Point]]  # metric_name -> [points]
AllRuns = dict[str, RunData]       # run_id -> RunData


def _lttb_downsample(points: list[Point], target: int) -> list[Point]:
    """Largest-Triangle-Three-Buckets downsampling."""
    n = len(points)
    if n <= target:
        return points

    bucket_size = (n - 2) / (target - 2)
    result: list[Point] = [points[0]]

    a_idx = 0
    for i in range(1, target - 1):
        bucket_start = int(math.floor((i - 1) * bucket_size)) + 1
        bucket_end = int(math.floor(i * bucket_size)) + 1
        next_bucket_start = int(math.floor(i * bucket_size)) + 1
        next_bucket_end = int(math.floor((i + 1) * bucket_size)) + 1
        next_bucket_end = min(next_bucket_end, n - 1)

        # Average of next bucket
        avg_step = sum(p.step for p in points[next_bucket_start:next_bucket_end + 1]) / max(1, next_bucket_end - next_bucket_start + 1)
        avg_val = sum(p.value for p in points[next_bucket_start:next_bucket_end + 1]) / max(1, next_bucket_end - next_bucket_start + 1)

        # Find point in current bucket with largest triangle area
        best_area = -1.0
        best_idx = bucket_start
        a = points[a_idx]
        for j in range(bucket_start, min(bucket_end + 1, n)):
            area = abs(
                (a.step - avg_step) * (points[j].value - a.value)
                - (a.step - points[j].step) * (avg_val - a.value)
            ) * 0.5
            if area > best_area:
                best_area = area
                best_idx = j

        result.append(points[best_idx])
        a_idx = best_idx

    result.append(points[-1])
    return result


MAX_POINTS_PER_SERIES = 2000


class MetricsCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.version: int = 0
        self.data: AllRuns = {}
        self.delta: AllRuns = {}
        self._runs_meta: dict[str, RunInfo] = {}

    def update(self, new_points: AllRuns) -> int:
        """Merge new points into cache, compute delta, bump version."""
        with self._lock:
            self.version += 1
            self.delta = {}

            for run_id, metrics in new_points.items():
                if run_id not in self.data:
                    self.data[run_id] = {}
                for metric_name, points in metrics.items():
                    if not points:
                        continue
                    if metric_name not in self.data[run_id]:
                        self.data[run_id][metric_name] = []

                    self.data[run_id][metric_name].extend(points)

                    # Downsample if too many points
                    series = self.data[run_id][metric_name]
                    if len(series) > MAX_POINTS_PER_SERIES:
                        self.data[run_id][metric_name] = _lttb_downsample(
                            series, MAX_POINTS_PER_SERIES
                        )

                    # Record delta
                    if run_id not in self.delta:
                        self.delta[run_id] = {}
                    self.delta[run_id][metric_name] = points

            return self.version

    def update_runs_meta(self, runs: list[RunInfo]) -> None:
        """Replace run metadata from a full DB scan."""
        with self._lock:
            self._runs_meta = {r.run_id: r for r in runs}

    def all_runs(self) -> list[dict]:
        """Return metadata for every known run."""
        with self._lock:
            return [
                {
                    "run_id": r.run_id,
                    "start_time": r.start_time,
                    "last_event_time": r.last_event_time,
                    "latest_step": r.latest_step,
                    "is_active": r.is_active,
                }
                for r in self._runs_meta.values()
            ]

    def snapshot(self) -> dict:
        """Full state for new SSE clients."""
        with self._lock:
            return {
                "version": self.version,
                "runs": {
                    run_id: {
                        metric: [[p.step, p.value, p.timestamp] for p in pts]
                        for metric, pts in metrics.items()
                    }
                    for run_id, metrics in self.data.items()
                },
                "runs_meta": [
                    {
                        "run_id": r.run_id,
                        "start_time": r.start_time,
                        "last_event_time": r.last_event_time,
                        "latest_step": r.latest_step,
                        "is_active": r.is_active,
                    }
                    for r in self._runs_meta.values()
                ],
            }

    def get_delta(self) -> dict:
        """Incremental update from latest poll."""
        with self._lock:
            return {
                "version": self.version,
                "runs": {
                    run_id: {
                        metric: [[p.step, p.value, p.timestamp] for p in pts]
                        for metric, pts in metrics.items()
                    }
                    for run_id, metrics in self.delta.items()
                },
            }

cache = MetricsCache()
