"""OpusConfig dataclass for OPUS data selection."""

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class OpusConfig:
    enabled: bool = False
    selection_mode: str = "opus"  # "opus" or "random"
    candidate_multiplier: int = 2
    selection_ratio: float = 0.5
    score_seq_len: int = 512
    proxy_batch_size: int = 8
    sketch_dim: int = 8192
    temperature: float = 0.9
    sketch_seed: int = 42
    fallback_random_on_error: bool = True
    max_selector_time_s: float = 30.0
    include_embeddings: bool = False
    include_lm_head: bool = False
    track_nonfinite_stats: bool = True
    zero2_exact_global_scoring: bool = True
    strict_shard_preconditioner: bool = False
    log_selection_metrics: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OpusConfig":
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
