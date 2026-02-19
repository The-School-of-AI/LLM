from typing import Optional

from fastapi import APIRouter, Query

from db import get_client

router = APIRouter()


@router.get("/metrics/{run_id}")
async def get_metrics(
    run_id: str,
    metric: str = Query(..., description="Metric name"),
    from_step: int = Query(0, description="Start step"),
    to_step: int = Query(10_000_000, description="End step"),
    from_time: Optional[float] = Query(None, description="Start epoch seconds"),
    to_time: Optional[float] = Query(None, description="End epoch seconds"),
):
    client = get_client()

    where = (
        "run_id = %(run_id)s AND metric = %(metric)s "
        "AND step >= %(from_step)s AND step <= %(to_step)s"
    )
    params: dict = {
        "run_id": run_id,
        "metric": metric,
        "from_step": from_step,
        "to_step": to_step,
    }
    if from_time is not None:
        where += " AND event_time >= fromUnixTimestamp(%(from_time)s)"
        params["from_time"] = int(from_time)
    if to_time is not None:
        where += " AND event_time <= fromUnixTimestamp(%(to_time)s)"
        params["to_time"] = int(to_time)

    result = client.query(
        f"SELECT step, value, event_time "
        f"FROM metric_points "
        f"WHERE {where} "
        f"ORDER BY step",
        parameters=params,
    )
    return {
        "run_id": run_id,
        "metric": metric,
        "points": [
            [row[0], float(row[1]), row[2].timestamp() if hasattr(row[2], "timestamp") else float(row[2])]
            for row in result.result_rows
        ],
    }
