import uuid
from datetime import datetime, timezone


def generate_run_id(prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{prefix}_{ts}_{short}"


def generate_training_run_id() -> str:
    return generate_run_id("run")


def generate_coreset_run_id() -> str:
    return generate_run_id("coreset")
