from datetime import datetime
import uuid

def _utc_ts():
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")

def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{_utc_ts()}_{uuid.uuid4().hex[:8]}"

def new_coreset_id() -> str:
    return new_run_id(prefix="coreset")
