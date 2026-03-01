from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def get_safe_filename(name: str) -> str:
    invalid = r"<>:\"/\\|?*"
    safe = "".join(("_" if ch in invalid else ch) for ch in name)
    safe = safe.replace("/", "_").replace("\\", "_").replace(":", "_")
    safe = re.sub(r"\s+", " ", safe).strip()
    return safe


def ensure_directory(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json_file(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        return None
    # lm-eval sometimes writes UTF-8 with BOM on Windows.
    text = p.read_text(encoding="utf-8-sig")
    return json.loads(text)


def write_json_file(path: str | Path, obj: Any) -> None:
    p = Path(path)
    if p.parent:
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def set_determinism_env(seed: int) -> None:
    # Best-effort determinism. Note: exact determinism on GPU can still vary by driver/cuDNN/kernel.
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Required by cuBLAS for deterministic results in some GEMM paths.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Make device ordering stable.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


@dataclass
class RunStatus:
    status: str
    updated_at_utc: str
    pid: int | None = None
    exit_code: int | None = None
    message: str | None = None


def _utc_iso() -> str:
    # time.strftime can't do sub-second; keep it simple and compatible
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def write_run_status(
    *,
    out_dir: str | Path,
    status: str,
    process_id: int | None = None,
    exit_code: int | None = None,
    message: str | None = None,
) -> None:
    if status not in {"created", "running", "succeeded", "failed"}:
        raise ValueError(f"Invalid status: {status}")

    out_dir = Path(out_dir)
    ensure_directory(out_dir)
    status_path = out_dir / "status.json"

    existing: dict[str, Any] = {}
    prev = read_json_file(status_path)
    if isinstance(prev, dict):
        existing.update(prev)

    existing["status"] = status
    existing["updated_at_utc"] = _utc_iso()
    if process_id and process_id > 0:
        existing["pid"] = process_id
    if exit_code is not None and exit_code >= 0:
        existing["exit_code"] = exit_code
    if message:
        existing["message"] = message

    write_json_file(status_path, existing)
