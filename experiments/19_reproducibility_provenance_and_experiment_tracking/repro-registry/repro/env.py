import json
import platform
import subprocess
from pathlib import Path


def capture_env(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "pip_freeze": _pip_freeze(),
    }

    output_path.write_text(json.dumps(env, indent=2))


def _pip_freeze():
    try:
        return subprocess.check_output(["pip", "freeze"]).decode().splitlines()
    except Exception:
        return []
