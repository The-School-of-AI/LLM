import hashlib
import yaml
from pathlib import Path

def freeze_config(config: dict, out_path: Path):
    out_path.write_text(yaml.dump(config, sort_keys=True))
    return sha256(out_path)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()
