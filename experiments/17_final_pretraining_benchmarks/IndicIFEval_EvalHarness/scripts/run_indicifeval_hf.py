from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from common import (
    ensure_directory,
    get_safe_filename,
    set_determinism_env,
    write_json_file,
    write_run_status,
)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _venv_python(repo_root: Path) -> str:
    if os.name == "nt":
        cand = repo_root / ".venv" / "Scripts" / "python.exe"
        if cand.exists():
            return str(cand)
    else:
        cand = repo_root / ".venv" / "bin" / "python"
        if cand.exists():
            return str(cand)
    return sys.executable


def main() -> int:
    ap = argparse.ArgumentParser(description="Run IndicIFEval via lm-eval (HF backend).")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tasks", nargs="+", default=["indicifeval_trans_hi"], help="One or more task ids")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", default="1")
    ap.add_argument("--limit", type=float, default=0, help="If <= 0, run full split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbosity", default="INFO", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--detached", action="store_true")
    ap.add_argument("--max_attempts", type=int, default=1)
    ap.add_argument("--retry_delay_sec", type=int, default=30)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    include_path = repo_root / "lm-evaluation-harness" / "custom_configs"

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        safe_model = get_safe_filename(args.model)
        out_dir = repo_root / "results" / "hf" / safe_model / _timestamp()

    if args.resume and not out_dir.exists():
        raise SystemExit(f"--resume specified but out_dir does not exist: {out_dir}")

    ensure_directory(out_dir)

    cache_path = out_dir / "lm_eval_cache.sqlite"
    meta_path = out_dir / "run_meta.json"

    set_determinism_env(args.seed)

    if not meta_path.exists():
        meta = {
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
            "runner": "run_indicifeval_hf.py",
            "model": args.model,
            "tasks": args.tasks,
            "device": args.device,
            "batch_size": args.batch_size,
            "limit": args.limit,
            "seed": args.seed,
            "verbosity": args.verbosity,
            "include_path": str(include_path),
            "cache_path": str(cache_path),
            "notes": "Uses lm-eval response cache to allow resume after interruptions.",
        }
        write_json_file(meta_path, meta)
        write_run_status(out_dir=out_dir, status="created")

    tasks_csv = ",".join(args.tasks)

    worker_script = repo_root / "scripts" / "run_indicifeval_hf_worker.py"
    py = _venv_python(repo_root)

    worker_cmd = [
        py,
        str(worker_script),
        "--model",
        args.model,
        "--tasks",
        tasks_csv,
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--limit",
        str(args.limit),
        "--seed",
        str(args.seed),
        "--verbosity",
        args.verbosity,
        "--out_dir",
        str(out_dir),
        "--cache_path",
        str(cache_path),
        "--include_path",
        str(include_path),
        "--max_attempts",
        str(args.max_attempts),
        "--retry_delay_sec",
        str(args.retry_delay_sec),
    ]

    if args.detached:
        pid_path = out_dir / "pid.txt"

        # Spawn an independent process so the job survives VS Code restarts.
        creationflags = 0
        if os.name == "nt":
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            creationflags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(
            worker_cmd,
            cwd=str(repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=(os.name != "nt"),
        )

        pid_path.write_text(str(proc.pid) + "\n", encoding="ascii")
        write_run_status(
            out_dir=out_dir,
            status="running",
            process_id=proc.pid,
            message="Detached run started",
        )

        print(f"Detached run started (pid={proc.pid}).")
        print(f"OutDir: {out_dir}")
        print(f"Log: {out_dir / 'lm_eval.log'}")
        return 0

    # Foreground run
    exit_code = subprocess.call(worker_cmd, cwd=str(repo_root))
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
