import hashlib
from pathlib import Path
from typing import Dict

import yaml


def freeze_config(config: Dict, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=True)

    return hash_file(output_path)


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
