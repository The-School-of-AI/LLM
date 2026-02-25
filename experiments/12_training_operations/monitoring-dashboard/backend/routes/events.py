from typing import Optional

from db import get_client
from fastapi import APIRouter, Query

router = APIRouter()


@router.get("/events/{run_id}")
async def get_events(
    run_id: str,
    limit: int = Query(100, description="Max number of events"),
    from_step: int = Query(0, description="Start step"),
    to_step: int = Query(10_000_000, description="End step"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
):
    client = get_client()

    where = "run_id = %(run_id)s AND step >= %(from_step)s AND step <= %(to_step)s"
    params: dict = {
        "run_id": run_id,
        "from_step": from_step,
        "to_step": to_step,
        "limit": limit,
    }
    if severity is not None:
        where += " AND severity = %(severity)s"
        params["severity"] = severity
    if event_type is not None:
        where += " AND event_type = %(event_type)s"
        params["event_type"] = event_type

    result = client.query(
        f"SELECT step, event_type, severity, message, host, rank, event_time "
        f"FROM events "
        f"WHERE {where} "
        f"ORDER BY step DESC LIMIT %(limit)s",
        parameters=params,
    )
    return {
        "run_id": run_id,
        "events": [
            {
                "step": row[0],
                "event_type": row[1],
                "severity": row[2],
                "message": row[3],
                "host": row[4],
                "rank": row[5],
                "timestamp": (
                    row[6].timestamp()
                    if hasattr(row[6], "timestamp")
                    else float(row[6])
                ),
            }
            for row in result.result_rows
        ],
    }
