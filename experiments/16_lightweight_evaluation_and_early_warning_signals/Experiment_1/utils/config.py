"""Shared configuration loader."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path | None = None) -> dict:
    if config_path is None:
        config_path = ROOT / "configs" / "eval_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_root() -> Path:
    return ROOT
