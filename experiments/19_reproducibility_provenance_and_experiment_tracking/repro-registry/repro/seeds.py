import json
import random
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None


def set_all_seeds(seed: int):
    random.seed(seed)
    if np:
        np.random.seed(seed)
    if torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def capture_seeds(seed: int, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = {
        "global_seed": seed,
        "python_random": seed,
        "numpy": seed if np else None,
        "torch": seed if torch else None,
    }

    output_path.write_text(json.dumps(seeds, indent=2))
