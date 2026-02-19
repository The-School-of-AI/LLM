from fastapi import APIRouter

from db import get_client

router = APIRouter()


@router.get("/checkpoints/{run_id}")
async def get_checkpoints(run_id: str):
    client = get_client()
    result = client.query(
        "SELECT metric, step, value, event_time "
        "FROM metric_points "
        "WHERE run_id = %(run_id)s AND metric LIKE 'checkpoint_%%' "
        "ORDER BY step",
        parameters={"run_id": run_id},
    )
    return {
        "run_id": run_id,
        "checkpoints": [
            {
                "metric": row[0],
                "step": row[1],
                "value": float(row[2]),
                "timestamp": row[3].timestamp() if hasattr(row[3], "timestamp") else float(row[3]),
            }
            for row in result.result_rows
        ],
    }
