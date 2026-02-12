from pathlib import Path

from repro.registry import start_run
from repro.seeds import set_seed

set_seed(42)

run_id, run_dir = start_run(Path("runs"))
(run_dir / "logs").mkdir()
(run_dir / "logs/train.log").write_text("step=1 loss=5.0\n")

print("Training run:", run_id)
