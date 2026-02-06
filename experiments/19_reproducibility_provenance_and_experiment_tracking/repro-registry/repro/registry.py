from pathlib import Path
from .ids import new_run_id
from .checksums import compute_checksums

def start_run(root: Path):
    run_id = new_run_id()
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    return run_id, run_dir

def finalize_run(run_dir: Path):
    checksums = compute_checksums(run_dir)
    (run_dir / "checksums.json").write_text(
        __import__("json").dumps(checksums, indent=2)
    )
