from pydantic import BaseModel
from typing import Optional, Dict

class RunManifest(BaseModel):
    run_id: str
    type: str  # training | coreset
    git: Dict
    env: Dict
    config_hash: str
    coreset_run_id: Optional[str] = None
