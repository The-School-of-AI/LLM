from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Quick CPU sanity run for IndicIFEval (HF backend).")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--task", default="indicifeval_trans_en_sanity")
    ap.add_argument("--limit", type=float, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbosity", default="INFO", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
    ap.add_argument("--detached", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    runner = repo_root / "scripts" / "run_indicifeval_hf.py"

    cmd = [
        sys.executable,
        str(runner),
        "--model",
        args.model,
        "--tasks",
        args.task,
        "--device",
        "cpu",
        "--batch_size",
        "1",
        "--limit",
        str(args.limit),
        "--seed",
        str(args.seed),
        "--verbosity",
        args.verbosity,
    ]
    if args.detached:
        cmd.append("--detached")

    return int(subprocess.call(cmd, cwd=str(repo_root)))


if __name__ == "__main__":
    raise SystemExit(main())
