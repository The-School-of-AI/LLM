#!/usr/bin/env python3
"""
Start Both Inference Servers
============================
Launches base and SFT inference servers as background processes and keeps
them running until Ctrl-C. Writes PID files so validate_option_b.py can
check server health.

Usage
-----
  python scripts/start_servers.py
  python scripts/start_servers.py --device cuda
  python scripts/start_servers.py --base-port 8001 --sft-port 8002
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
SCRIPTS   = ROOT / "scripts"
PID_DIR   = ROOT / "scripts" / ".pids"

BASE_MODEL_PATH = str(ROOT / "models" / "base")
SFT_MODEL_PATH  = str(ROOT / "models" / "sft")
BASE_PORT = 8001
SFT_PORT  = 8002

PYTHON = sys.executable


def start_server(
    model_path: str,
    port: int,
    device: str,
    load_in_4bit: bool,
    label: str,
) -> subprocess.Popen:
    cmd = [
        PYTHON,
        str(SCRIPTS / "inference_server.py"),
        "--model-path", model_path,
        "--port", str(port),
        "--device", device,
    ]
    if load_in_4bit:
        cmd.append("--load-in-4bit")

    log_path = PID_DIR / f"{label}_server.log"
    pid_path = PID_DIR / f"{label}.pid"

    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)

    pid_path.write_text(str(proc.pid), encoding="utf-8")
    print(f"  [{label.upper()}] PID {proc.pid}  port {port}  log → {log_path}")
    return proc


def wait_for_health(port: int, label: str, timeout: int = 120) -> bool:
    """Poll /health until ready or timeout."""
    import urllib.request
    import urllib.error

    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    dots = 0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    print(f"\r  [{label.upper()}] ✓ Ready on port {port}          ")
                    return True
        except Exception:
            pass
        print(f"\r  [{label.upper()}] Waiting for port {port} {'.' * (dots % 6 + 1)}   ", end="", flush=True)
        dots += 1
        time.sleep(3)
    print(f"\r  [{label.upper()}] ✗ Timed out waiting for port {port}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start base and SFT inference servers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-model-path", default=BASE_MODEL_PATH)
    parser.add_argument("--sft-model-path",  default=SFT_MODEL_PATH)
    parser.add_argument("--base-port", type=int, default=BASE_PORT)
    parser.add_argument("--sft-port",  type=int, default=SFT_PORT)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    PID_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Starting SFT Experiment Inference Servers")
    print("=" * 60)

    base_proc = start_server(args.base_model_path, args.base_port, args.device, args.load_in_4bit, "base")
    sft_proc  = start_server(args.sft_model_path,  args.sft_port,  args.device, args.load_in_4bit, "sft")

    print("\n  Waiting for servers to load models (this may take several minutes)…")

    base_ok = wait_for_health(args.base_port, "base")
    sft_ok  = wait_for_health(args.sft_port,  "sft")

    if not (base_ok and sft_ok):
        print("\n  ✗ One or both servers failed to start. Check log files in scripts/.pids/")
        base_proc.terminate()
        sft_proc.terminate()
        sys.exit(1)

    print(f"""
  ✓ Both servers are ready.

  Base : http://127.0.0.1:{args.base_port}/v1
  SFT  : http://127.0.0.1:{args.sft_port}/v1

  Press Ctrl-C to stop both servers.
  Log files: scripts/.pids/base_server.log
             scripts/.pids/sft_server.log
""")

    # Keep running until interrupted
    def _shutdown(sig, frame):
        print("\n\n  Stopping servers…")
        base_proc.terminate()
        sft_proc.terminate()
        print("  Done.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        # Restart any crashed processes
        if base_proc.poll() is not None:
            print("  [BASE] Server exited unexpectedly — restarting…")
            base_proc = start_server(
                args.base_model_path, args.base_port, args.device, args.load_in_4bit, "base"
            )
        if sft_proc.poll() is not None:
            print("  [SFT]  Server exited unexpectedly — restarting…")
            sft_proc = start_server(
                args.sft_model_path, args.sft_port, args.device, args.load_in_4bit, "sft"
            )
        time.sleep(5)


if __name__ == "__main__":
    main()
