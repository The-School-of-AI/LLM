from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class GitInfo(BaseModel):
    repo_url: str
    commit_hash: str
    dirty: bool


class RunManifest(BaseModel):
    run_id: str
    pipeline: str
    created_at: datetime

    git: GitInfo
    config_hash: str
    seed: int

    coreset_run_id: Optional[str] = None
    status: str  # STARTED / COMPLETED / FAILED
