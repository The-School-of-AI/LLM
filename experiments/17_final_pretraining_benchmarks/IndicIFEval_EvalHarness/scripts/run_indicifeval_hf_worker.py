from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from common import ensure_directory, set_determinism_env, write_run_status


def _find_lm_eval_exe(repo_root: Path) -> str:
    # Prefer the workspace venv console entrypoint to avoid environment mismatch.
    if os.name == "nt":
        cand = repo_root / ".venv" / "Scripts" / "lm_eval.exe"
        if cand.exists():
            return str(cand)
    else:
        cand = repo_root / ".venv" / "bin" / "lm_eval"
        if cand.exists():
            return str(cand)

    # Fallback to PATH.
    from shutil import which

    w = which("lm_eval")
    if w:
        return w

    # Last resort: python -m lm_eval (won't exist if not installed as module).
    return ""


def _tee_subprocess(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    ensure_directory(log_path.parent)

    with log_path.open("a", encoding="utf-8", errors="replace") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_f.write(line)
            log_f.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        return proc.wait()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", required=True, help="Comma-separated task list")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", default="1")
    ap.add_argument("--limit", type=float, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbosity", default="INFO", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_path", required=True)
    ap.add_argument("--include_path", required=True)
    ap.add_argument("--max_attempts", type=int, default=1)
    ap.add_argument("--retry_delay_sec", type=int, default=30)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_directory(out_dir)
    set_determinism_env(args.seed)

    log_path = out_dir / "lm_eval.log"
    write_run_status(out_dir=out_dir, status="running", message="lm_eval started", process_id=os.getpid())

    repo_root = Path(__file__).resolve().parents[1]
    lm_eval_exe = _find_lm_eval_exe(repo_root)
    if not lm_eval_exe:
        write_run_status(out_dir=out_dir, status="failed", exit_code=1, message="lm_eval not found")
        raise SystemExit("lm_eval executable not found (expected .venv/Scripts/lm_eval.exe or lm_eval on PATH)")

    base_args = [
        lm_eval_exe,
        "--model",
        "hf",
        "--model_args",
        f"pretrained={args.model},trust_remote_code=True",
        "--device",
        args.device,
        "--include_path",
        args.include_path,
        "--tasks",
        args.tasks,
        "--num_fewshot",
        "0",
        "--batch_size",
        str(args.batch_size),
        "--output_path",
        str(out_dir),
        "--log_samples",
        "--verbosity",
        args.verbosity,
        "--seed",
        str(args.seed),
        "--use_cache",
        str(args.cache_path),
        "--cache_requests",
        "true",
    ]

    if args.limit and args.limit > 0:
        base_args += ["--limit", str(args.limit)]

    env = os.environ.copy()

    max_attempts = max(1, int(args.max_attempts))
    for attempt in range(1, max_attempts + 1):
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        header = (
            f"[{stamp}] attempt={attempt} model={args.model} tasks={args.tasks} "
            f"device={args.device} batch_size={args.batch_size} limit={args.limit} seed={args.seed}\n"
        )
        with log_path.open("a", encoding="utf-8", errors="replace") as f:
            f.write(header)

        exit_code = _tee_subprocess(base_args, log_path=log_path, env=env)
        if exit_code == 0:
            write_run_status(out_dir=out_dir, status="succeeded", exit_code=0, message="lm_eval finished successfully")
            return 0

        write_run_status(out_dir=out_dir, status="failed", exit_code=exit_code, message=f"lm_eval failed (attempt {attempt})")
        if attempt < max_attempts:
            retry_msg = f"Retrying in {args.retry_delay_sec}s (resume via cache: {args.cache_path})\n"
            with log_path.open("a", encoding="utf-8", errors="replace") as f:
                f.write(retry_msg)
            sys.stdout.write(retry_msg)
            sys.stdout.flush()
            time.sleep(args.retry_delay_sec)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
