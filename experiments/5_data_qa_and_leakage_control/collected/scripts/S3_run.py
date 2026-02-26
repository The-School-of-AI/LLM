#!/usr/bin/env python3
"""
One-command runner for the S3 contamination scan flow.

Reads project config from ``S3_scan.json`` (and optional ``S3_aws.json``),
ensures benchmarks exist, then runs ``scripts/scan_from_s3.py``.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def run_python_snippet(snippet: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )


def detect_cuda_gpu(force_cpu: bool) -> bool:
    """Detect an NVIDIA CUDA-capable environment for GPU semantic path."""
    if force_cpu:
        return False

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            text=True,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except FileNotFoundError:
        return False


def probe_semantic_runtime() -> dict[str, bool]:
    """Check whether sentence-transformers and FAISS are importable, and FAISS GPU support exists."""
    snippet = r"""
import importlib.util, json
out = {
  "sentence_transformers": importlib.util.find_spec("sentence_transformers") is not None,
  "faiss": importlib.util.find_spec("faiss") is not None,
  "faiss_gpu": False,
}
if out["faiss"]:
    try:
        import faiss
        out["faiss_gpu"] = hasattr(faiss, "StandardGpuResources")
    except Exception:
        out["faiss"] = False
print(json.dumps(out))
"""
    result = run_python_snippet(snippet)
    if result.returncode != 0 or not result.stdout.strip():
        return {"sentence_transformers": False, "faiss": False, "faiss_gpu": False}
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return {"sentence_transformers": False, "faiss": False, "faiss_gpu": False}


def pip_install(packages: list[str], env: dict[str, str]) -> None:
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def pip_uninstall(packages: list[str], env: dict[str, str]) -> None:
    cmd = [sys.executable, "-m", "pip", "uninstall", "-y", *packages]
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=False)


def ensure_semantic_runtime(env: dict[str, str], force_cpu: bool) -> str:
    """Auto-install semantic dependencies. Prefer GPU FAISS on CUDA hosts."""
    prefer_gpu = detect_cuda_gpu(force_cpu)
    if force_cpu:
        print("CPU override enabled: forcing semantic CPU mode")
    else:
        print(f"GPU detected for semantic path: {'yes' if prefer_gpu else 'no'}")

    status = probe_semantic_runtime()
    has_sentence = status["sentence_transformers"]
    has_faiss = status["faiss"]
    has_faiss_gpu = status["faiss_gpu"]

    target = "cpu"
    if prefer_gpu:
        target = "gpu"
        if has_sentence and has_faiss_gpu:
            print("Semantic runtime already available (sentence-transformers + FAISS GPU)")
            return "cuda"
    else:
        if has_sentence and has_faiss:
            print("Semantic runtime already available (sentence-transformers + FAISS)")
            return "cpu"

    print("Installing semantic runtime dependencies...")
    if prefer_gpu:
        # Remove CPU FAISS first to avoid package conflicts, then try GPU FAISS.
        pip_uninstall(["faiss-cpu"], env)
        try:
            pip_install(["sentence-transformers", "faiss-gpu"], env)
            status = probe_semantic_runtime()
            if status["sentence_transformers"] and status["faiss_gpu"]:
                print("Semantic runtime ready with FAISS GPU")
                return "cuda"
            print("FAISS GPU install did not expose GPU APIs; falling back to CPU FAISS")
        except subprocess.CalledProcessError as exc:
            print(f"FAISS GPU install failed ({exc}); falling back to CPU FAISS")

        pip_uninstall(["faiss-gpu"], env)
        pip_install(["sentence-transformers", "faiss-cpu"], env)
        return "cuda"

    # CPU path
    if not has_sentence or not has_faiss:
        pip_install(["sentence-transformers", "faiss-cpu"], env)
    return "cpu"


def merge_aws_env(env: dict[str, str], aws_cfg: dict) -> dict[str, str]:
    updated = dict(env)

    mapping = {
        "access_key_id": "AWS_ACCESS_KEY_ID",
        "secret_access_key": "AWS_SECRET_ACCESS_KEY",
        "session_token": "AWS_SESSION_TOKEN",
        "region": "AWS_DEFAULT_REGION",
        "profile": "AWS_PROFILE",
    }
    for src, dst in mapping.items():
        value = aws_cfg.get(src)
        if value:
            updated[dst] = str(value)
    return updated


def ensure_benchmarks(config: dict, env: dict[str, str]) -> None:
    benchmarks_dir = PROJECT_ROOT / config.get("benchmarks_dir", "benchmarks")
    auto_download = bool(config.get("auto_download_benchmarks", True))

    has_benchmarks = benchmarks_dir.exists() and any(benchmarks_dir.glob("*_test.jsonl"))
    if has_benchmarks:
        return

    if not auto_download:
        raise FileNotFoundError(
            f"Benchmarks not found in {benchmarks_dir}. "
            "Enable auto_download_benchmarks or run download_benchmarks.py first."
        )

    print(f"Benchmarks not found in {benchmarks_dir}. Downloading...")
    cmd = [sys.executable, "scripts/download_benchmarks.py"]
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def build_scan_command(config: dict, semantic_device: str) -> list[str]:
    required = ["s3_uri", "team_name", "batch_name"]
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"Missing required fields in S3_scan.json: {', '.join(missing)}")

    cmd = [
        sys.executable,
        "scripts/scan_from_s3.py",
        str(config["s3_uri"]),
        str(config["team_name"]),
        str(config["batch_name"]),
    ]

    cmd.extend(["--benchmarks-dir", str(config.get("benchmarks_dir", "benchmarks"))])
    cmd.extend(["--reports-dir", str(config.get("reports_dir", "reports"))])
    cmd.extend(["--semantic-device", semantic_device])
    return cmd


def main() -> None:
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run S3 scan from S3_scan.json")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force semantic detector to use CPU (disables GPU use even if available)",
    )
    args = parser.parse_args()

    scan_cfg_path = PROJECT_ROOT / "S3_scan.json"
    aws_cfg_path = PROJECT_ROOT / "S3_aws.json"

    try:
        config = load_json(scan_cfg_path)
    except Exception as exc:
        print(f"Error loading S3_scan.json: {exc}")
        sys.exit(1)

    env = dict(os.environ)

    # Non-secret AWS settings can live in S3_scan.json
    if config.get("aws_region"):
        env["AWS_DEFAULT_REGION"] = str(config["aws_region"])
    if config.get("aws_profile"):
        env["AWS_PROFILE"] = str(config["aws_profile"])

    force_cpu = args.cpu or os.environ.get("S3_FORCE_CPU", "").strip() in {"1", "true", "TRUE"}

    # Optional local credentials file (keep local; avoid committing real secrets)
    if aws_cfg_path.exists():
        try:
            aws_cfg = load_json(aws_cfg_path)
            env = merge_aws_env(env, aws_cfg)
            print("Loaded AWS settings from S3_aws.json")
        except Exception as exc:
            print(f"Error loading S3_aws.json: {exc}")
            sys.exit(1)

    try:
        ensure_benchmarks(config, env)
        ensure_semantic_runtime(env, force_cpu=force_cpu)
        semantic_device = "cpu" if force_cpu else "auto"
        cmd = build_scan_command(config, semantic_device)
    except Exception as exc:
        print(f"Configuration error: {exc}")
        sys.exit(1)

    print("Running S3 scan")
    print(f"  S3 URI:   {config['s3_uri']}")
    print(f"  Team:     {config['team_name']}")
    print(f"  Batch:    {config['batch_name']}")
    print()

    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    elapsed = time.perf_counter() - start_time
    print()
    print(f"Total runtime: {format_duration(elapsed)} ({elapsed:.1f}s)")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
