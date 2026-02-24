from db import get_client
from fastapi import APIRouter

router = APIRouter()


@router.get("/checkpoints/{run_id}")
async def get_checkpoints(run_id: str):
    client = get_client()
    result = client.query(
        "SELECT step, s3_key, loss, tag, is_protected, status, "
        "       duration_s, size_bytes, host, event_time "
        "FROM checkpoints FINAL "
        "WHERE run_id = %(run_id)s "
        "ORDER BY step DESC "
        "LIMIT 200",
        parameters={"run_id": run_id},
    )
    return {
        "run_id": run_id,
        "checkpoints": [
            {
                "step": row[0],
                "s3_key": row[1],
                "loss": float(row[2]) if row[2] is not None else None,
                "tag": row[3],
                "is_protected": bool(row[4]),
                "status": row[5],
                "duration_s": float(row[6]) if row[6] is not None else None,
                "size_bytes": int(row[7]) if row[7] is not None else None,
                "host": row[8],
                "timestamp": (
                    row[9].timestamp()
                    if hasattr(row[9], "timestamp")
                    else float(row[9])
                ),
            }
            for row in result.result_rows
        ],
    }
