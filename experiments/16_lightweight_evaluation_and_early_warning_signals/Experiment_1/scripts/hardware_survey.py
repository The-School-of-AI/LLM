#!/usr/bin/env python3
"""
Phase 1 — Hardware survey tool.

Detects available hardware and prints a summary that team members can share.
Also outputs a JSON snippet to paste into the team hardware register.

Usage:
    python scripts/hardware_survey.py
"""
from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from datetime import datetime


def detect_gpu() -> list[dict]:
    gpus = []

    # NVIDIA via nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                gpus.append({"vendor": "NVIDIA", "name": parts[0],
                             "vram_mb": int(parts[1]) if parts[1].isdigit() else parts[1],
                             "driver": parts[2] if len(parts) > 2 else "?"})
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Apple Metal via system_profiler
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            data = json.loads(out)
            displays = data.get("SPDisplaysDataType", [])
            for d in displays:
                gpu_name = d.get("sppci_model", d.get("_name", "Apple GPU"))
                vram = d.get("spdisplays_vram", "shared")
                gpus.append({"vendor": "Apple", "name": gpu_name, "vram": vram, "metal": True})
        except Exception:
            pass

    return gpus


def detect_ram() -> int:
    """Return total RAM in GB."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3))
    except ImportError:
        pass

    # macOS fallback
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "hw.memsize"], text=True)
            bytes_ram = int(out.split(":")[1].strip())
            return round(bytes_ram / (1024 ** 3))
        except Exception:
            pass

    return -1


def detect_cpu() -> dict:
    info: dict = {
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
        "cores": None,
    }
    try:
        import psutil
        info["cores"] = psutil.cpu_count(logical=False)
        info["threads"] = psutil.cpu_count(logical=True)
    except ImportError:
        pass
    return info


def can_run_int4(ram_gb: int, gpus: list[dict]) -> str:
    """Rough estimate of whether this machine can run INT4 7B inference."""
    # INT4 7B needs ~4-5 GB VRAM or ~6-8 GB RAM
    if any("NVIDIA" in g.get("vendor", "") for g in gpus):
        vram = max((g.get("vram_mb", 0) for g in gpus if g.get("vendor") == "NVIDIA"), default=0)
        if isinstance(vram, int) and vram >= 6000:
            return "YES (NVIDIA GPU)"
    if any("Apple" in g.get("vendor", "") for g in gpus):
        if ram_gb >= 16:
            return "YES (Apple Metal, unified memory)"
    if ram_gb >= 16:
        return "YES (CPU-only, will be slow)"
    return "UNCERTAIN (low RAM)"


def main() -> None:
    print("=" * 60)
    print("  Team 16 Early Warning — Hardware Survey")
    print("=" * 60)

    hostname = socket.gethostname()
    os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
    cpu = detect_cpu()
    ram_gb = detect_ram()
    gpus = detect_gpu()
    int4_capable = can_run_int4(ram_gb, gpus)

    print(f"\n  Hostname : {hostname}")
    print(f"  OS       : {os_info}")
    print(f"  CPU      : {cpu['processor']}")
    print(f"  Cores    : {cpu.get('cores', '?')} physical / {cpu.get('threads', '?')} logical")
    print(f"  RAM      : {ram_gb} GB")
    print(f"  INT4 cap.: {int4_capable}")

    if gpus:
        print(f"  GPUs     :")
        for g in gpus:
            vram = g.get("vram_mb") or g.get("vram", "?")
            print(f"    - {g.get('vendor')} {g.get('name')} | VRAM: {vram}")
    else:
        print("  GPUs     : None detected")

    survey = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hostname": hostname,
        "os": os_info,
        "cpu": cpu,
        "ram_gb": ram_gb,
        "gpus": gpus,
        "int4_capable": int4_capable,
        "recommended_backend": (
            "llama_cpp" if platform.system() == "Darwin" and ram_gb >= 16
            else "bitsandbytes" if gpus else "llama_cpp"
        ),
        "recommended_quant": (
            "int4" if ram_gb >= 8 else "int8"
        ),
    }

    print("\n  === JSON snippet for hardware register ===")
    print(json.dumps(survey, indent=2))
    print("\nShare the JSON above with your team lead (Teams_9_12 coordination).")


if __name__ == "__main__":
    main()
