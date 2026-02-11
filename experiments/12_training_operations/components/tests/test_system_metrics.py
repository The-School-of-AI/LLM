
import json
import time
import shutil
import os

def test_system_metrics_collector():
    """Verify SystemMetricsCollector writes valid JSONL that Vector can ingest."""

    log_dir = "/tmp/test_system_metrics_logs"
    run_id = "test_sys_run"
    rank = 0

    # Cleanup
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    from components.system_metrics.collector import SystemMetricsCollector

    collector = SystemMetricsCollector(
        log_dir=log_dir,
        run_id=run_id,
        rank=rank,
        interval=1.0,
        gpu=False,  # skip GPU for CI/test environments
    )

    # ---- Test collect_once() returns expected keys ----
    print("--- Test: collect_once() ---")
    metrics = collector.collect_once()
    assert "sys.cpu_percent" in metrics, "Missing sys.cpu_percent"
    assert "sys.mem_percent" in metrics, "Missing sys.mem_percent"
    assert "sys.mem_total_bytes" in metrics, "Missing sys.mem_total_bytes"
    assert "sys.mem_used_bytes" in metrics, "Missing sys.mem_used_bytes"
    assert "sys.load_1m" in metrics, "Missing sys.load_1m"
    assert "sys.swap_percent" in metrics, "Missing sys.swap_percent"

    # Disk metrics (default path "/")
    assert "sys.disk.root.percent" in metrics, "Missing sys.disk.root.percent"
    assert "sys.disk.root.total_bytes" in metrics, "Missing sys.disk.root.total_bytes"

    # All values should be numeric
    for k, v in metrics.items():
        assert isinstance(v, (int, float)), f"{k} is not numeric: {type(v)}"

    print(f"✓ collect_once() returned {len(metrics)} metrics, all numeric")

    # ---- Test start/stop writes JSONL to disk ----
    print("\n--- Test: start/stop lifecycle ---")
    collector.start()
    time.sleep(2.5)  # let it write at least 2 samples
    collector.stop()

    log_file = os.path.join(log_dir, f"system_metrics_rank_{rank}.jsonl")
    assert os.path.exists(log_file), f"Log file not created: {log_file}"

    with open(log_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) >= 2, f"Expected >= 2 lines, got {len(lines)}"
    print(f"✓ Wrote {len(lines)} JSONL lines")

    # ---- Test JSONL format matches training logger schema ----
    print("\n--- Test: JSONL schema compatibility ---")
    for i, line in enumerate(lines):
        record = json.loads(line)

        # Required top-level fields (same as JSONLogger output)
        assert "timestamp" in record, f"Line {i}: missing timestamp"
        assert "run_id" in record, f"Line {i}: missing run_id"
        assert "host" in record, f"Line {i}: missing host"
        assert "rank" in record, f"Line {i}: missing rank"
        assert "step" in record, f"Line {i}: missing step"
        assert "metrics" in record, f"Line {i}: missing metrics"
        assert "context" in record, f"Line {i}: missing context"

        assert record["run_id"] == run_id
        assert record["rank"] == rank
        assert record["step"] == 0  # system metrics always step=0
        assert record["context"]["collector"] == "system"

        # Metrics should be a dict of numeric values
        m = record["metrics"]
        assert isinstance(m, dict), f"Line {i}: metrics is not a dict"
        assert len(m) > 0, f"Line {i}: metrics is empty"
        for k, v in m.items():
            assert isinstance(v, (int, float)), f"Line {i}: {k} = {v} is not numeric"

    print(f"✓ All {len(lines)} lines have valid schema")

    # ---- Test timestamp is ISO 8601 UTC ----
    print("\n--- Test: timestamp format ---")
    first = json.loads(lines[0])
    ts = first["timestamp"]
    assert ts.endswith("Z"), f"Timestamp not UTC: {ts}"
    assert "T" in ts, f"Timestamp not ISO format: {ts}"
    print(f"✓ Timestamp format OK: {ts}")

    # ---- Test network delta metrics exist ----
    print("\n--- Test: network metrics ---")
    last = json.loads(lines[-1])
    net_keys = [k for k in last["metrics"] if k.startswith("sys.net.")]
    # At least one interface should be present (unless running in a very bare container)
    print(f"  Network metric keys: {net_keys}")
    if net_keys:
        print(f"✓ Found {len(net_keys)} network metrics")
    else:
        print("⚠ No network metrics (may be expected in some environments)")

    # Cleanup
    shutil.rmtree(log_dir)
    print("\n✅ All SystemMetricsCollector tests passed!")


if __name__ == "__main__":
    test_system_metrics_collector()
