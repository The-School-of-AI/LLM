import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_doctor_passes_default_mode():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "doctor.py"), "--root", str(ROOT)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout


def test_no_deprecated_turboquant_names_outside_artifacts():
    forbidden = (
        "TQ-" + "Lo" + "RA",
        "TQ" + "Lo" + "RA",
        "T" + "Q" + "L",
        "tq_" + "lo" + "ra",
        "tq" + "lo" + "ra",
    )
    suffixes = {".py", ".sh", ".yaml", ".yml", ".md", ".json", ".txt", ".toml"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if (
            "tokenizer" in path.parts
            or "manifests" in path.parts
            or "__pycache__" in path.parts
        ):
            continue
        text = path.read_text(errors="ignore")
        assert not any(item in text for item in forbidden), str(path)
