import hashlib
from pathlib import Path

def compute_checksums(root: Path):
    out = {}
    for p in root.rglob("*"):
        if p.is_file():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            out[str(p.relative_to(root))] = f"sha256:{h}"
    return out
