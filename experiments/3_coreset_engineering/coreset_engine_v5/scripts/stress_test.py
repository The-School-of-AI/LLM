#!/usr/bin/env python3
"""
NOT TO BE USED IN PRODUCTION
stress_test.py — Simulate CPU + I/O + Memory load to test monitoring.

Usage:
    # Start monitors first
    nohup scripts/monitor.sh &

    # Run this stress test (~60s)
    python3 scripts/stress_test.py

    # Stop monitors
    kill $(cat /mnt/nvme/logs/monitor.pid)

    # Check reports
    ./scripts/monitor_report.sh
    python3 scripts/monitor_report.py
"""

import math
import os
import time
from multiprocessing import Pool, cpu_count


def cpu_burn(duration_sec: int) -> str:
    """Burn CPU for N seconds with math ops."""
    end = time.time() + duration_sec
    total = 0.0
    i = 0
    while time.time() < end:
        total += math.sqrt(i) * math.sin(i) * math.cos(i)
        i += 1
    return f"pid={os.getpid()} iters={i:,}"


def memory_allocate(mb: int) -> list:
    """Allocate N MB of memory."""
    blocks = []
    for _ in range(mb):
        blocks.append(bytearray(1024 * 1024))  # 1 MB each
    return blocks


def disk_write(path: str, mb: int):
    """Write N MB of data to disk."""
    chunk = b"X" * (1024 * 1024)  # 1 MB
    with open(path, "wb") as f:
        for _ in range(mb):
            f.write(chunk)
    os.remove(path)


def main():
    n_cores = min(cpu_count(), 8)  # Use up to 8 cores
    scratch = os.environ.get("SCRATCH_DIR", "/mnt/nvme")
    duration_hrs = float(os.environ.get("DURATION", "1"))
    duration = int(duration_hrs * 3600)

    print("🔥 Stress test starting")
    print(f"   Cores to burn : {n_cores}")
    print(f"   Duration      : {duration_hrs}h ({duration}s)")
    print(f"   Scratch dir   : {scratch}")
    print()

    # Phase 1: CPU burn (parallel)
    print(f"[1/3] CPU burn ({n_cores} cores × {duration}s)...")
    t0 = time.time()
    with Pool(n_cores) as pool:
        results = pool.starmap(cpu_burn, [(duration,)] * n_cores)
    print(f"      Done in {time.time()-t0:.1f}s")
    for r in results:
        print(f"      {r}")

    # Phase 2: Memory allocation
    alloc_mb = 512
    print(f"\n[2/3] Allocating {alloc_mb} MB RAM...")
    t0 = time.time()
    blocks = memory_allocate(alloc_mb)
    print(f"      Done in {time.time()-t0:.1f}s " f"({len(blocks)} blocks)")
    time.sleep(5)  # Hold for monitors to capture
    del blocks

    # Phase 3: Disk I/O
    write_mb = 256
    test_file = os.path.join(scratch, "stress_test.bin")
    print(f"\n[3/3] Writing {write_mb} MB to {test_file}...")
    t0 = time.time()
    try:
        disk_write(test_file, write_mb)
        print(f"      Done in {time.time()-t0:.1f}s")
    except (PermissionError, FileNotFoundError) as e:
        print(f"      Skipped disk test: {e}")
        print("      (Set SCRATCH_DIR to a writable path)")

    print("\n✅ Stress test complete!")
    print("   Now run: ./scripts/monitor_report.sh")


if __name__ == "__main__":
    main()
