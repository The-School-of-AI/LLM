from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

from cache import AllRuns, Point, RunInfo, cache
from config import settings
from db import get_client
from sse import broadcaster

logger = logging.getLogger(__name__)

_last_poll_time: datetime | None = None

RUNS_META_INTERVAL = 30.0  # seconds between full run-metadata scans


def _fetch_active_runs() -> list[str]:
    client = get_client()
    result = client.query(
        "SELECT DISTINCT run_id FROM metric_points "
        "WHERE event_time > now() - INTERVAL 5 MINUTE"
    )
    return [row[0] for row in result.result_rows]


def _fetch_all_run_ids() -> list[str]:
    client = get_client()
    result = client.query("SELECT DISTINCT run_id FROM metric_points")
    return [row[0] for row in result.result_rows]


def _fetch_metrics_since(run_id: str, since: datetime) -> list[tuple]:
    client = get_client()
    result = client.query(
        "SELECT metric, step, value, event_time "
        "FROM metric_points "
        "WHERE run_id = %(run_id)s AND event_time > %(since)s "
        "ORDER BY step",
        parameters={"run_id": run_id, "since": since},
    )
    return result.result_rows


def _fetch_all_runs_meta() -> list[RunInfo]:
    """Query aggregate metadata for every run in the DB."""
    client = get_client()
    result = client.query(
        "SELECT run_id, min(event_time), max(event_time), max(step) "
        "FROM metric_points GROUP BY run_id ORDER BY max(event_time) DESC"
    )
    now_ts = time.time()
    runs: list[RunInfo] = []
    for run_id, min_et, max_et, max_step in result.result_rows:
        start_ts = min_et.timestamp() if hasattr(min_et, "timestamp") else float(min_et)
        last_ts = max_et.timestamp() if hasattr(max_et, "timestamp") else float(max_et)
        is_active = (now_ts - last_ts) < 300  # active if data within 5 min
        runs.append(RunInfo(
            run_id=run_id,
            start_time=start_ts,
            last_event_time=last_ts,
            latest_step=int(max_step),
            is_active=is_active,
        ))
    return runs


def _poll_once() -> AllRuns:
    global _last_poll_time

    now = datetime.now(timezone.utc)
    since = _last_poll_time or datetime(2020, 1, 1, tzinfo=timezone.utc)

    # First poll: load all runs (including inactive); subsequent: only active
    if _last_poll_time is None:
        run_ids = _fetch_all_run_ids()
        logger.info("Initial poll: %d total runs", len(run_ids))
    else:
        run_ids = _fetch_active_runs()
        logger.info("Poll: %d active runs, since=%s", len(run_ids), since.isoformat())

    new_points: AllRuns = {}
    for run_id in run_ids:
        rows = _fetch_metrics_since(run_id, since)
        if not rows:
            continue
        run_data: dict[str, list[Point]] = {}
        for metric, step, value, event_time in rows:
            ts = event_time.timestamp() if hasattr(event_time, "timestamp") else float(event_time)
            if metric not in run_data:
                run_data[metric] = []
            run_data[metric].append(Point(step=int(step), value=float(value), timestamp=ts))
        new_points[run_id] = run_data

    _last_poll_time = now
    return new_points


async def poll_loop() -> None:
    logger.info("Poller started (interval=%.1fs)", settings.poll_interval_sec)
    while True:
        try:
            new_points = await asyncio.get_event_loop().run_in_executor(
                None, _poll_once
            )
            if new_points:
                version = cache.update(new_points)
                delta = cache.get_delta()
                await broadcaster.broadcast(delta)
                logger.info(
                    "Poll complete: version=%d, clients=%d",
                    version,
                    broadcaster.client_count,
                )
            else:
                logger.debug("Poll: no new data")
        except Exception:
            logger.exception("Poller error")

        await asyncio.sleep(settings.poll_interval_sec)


async def runs_meta_loop() -> None:
    """Periodically refresh run metadata and broadcast to clients."""
    logger.info("Runs-meta poller started (interval=%.0fs)", RUNS_META_INTERVAL)
    while True:
        try:
            runs = await asyncio.get_event_loop().run_in_executor(
                None, _fetch_all_runs_meta
            )
            cache.update_runs_meta(runs)
            meta_payload = {
                "runs_meta": cache.all_runs(),
            }
            await broadcaster.broadcast_event("runs_meta", meta_payload)
            logger.info("Runs-meta refresh: %d runs", len(runs))
        except Exception:
            logger.exception("Runs-meta poller error")

        await asyncio.sleep(RUNS_META_INTERVAL)
