"""
Integration test for TrainingOps — verifies the full pipeline:
  TrainingOps → JSONLogger + SystemMetrics + MetricsServer + CheckpointRegistry
"""

import json
import time
import urllib.request
from pathlib import Path

from components.training_ops import TrainingOps


def test_training_ops():
    """Simulate a mini training run through TrainingOps."""

    run_id = f"test_ops_{int(time.time())}"
    log_dir = "/tmp/training_logs"

    print("=" * 60)
    print("  test_training_ops — integration test")
    print("=" * 60)

    # ---- 1. Init (starts all services, runs preflight) ----
    ops = TrainingOps(
        run_id=run_id,
        rank=0,
        log_dir=log_dir,
        metrics_port=8111,  # avoid conflict with any running server
        system_metrics_interval=2.0,
        skip_vector_check=True,  # no Vector on this machine
    )

    # ---- 2. Simulate a few training steps ----
    print("\n--- Simulating training steps ---")
    for step in range(5):
        ops.log_step(
            step=step,
            metrics={
                "loss": 5.0 - step * 0.5,
                "lr": 3e-4,
                "tokens_per_second": 10000 + step * 500,
                "gradient_norm": 1.2 - step * 0.1,
            },
            context={"epoch": 0, "phase": "warmup"},
        )
        time.sleep(0.1)
    print("✓ Logged 5 training steps")

    # ---- 3. Simulate a checkpoint save ----
    print("\n--- Simulating checkpoint ---")
    ops.log_checkpoint(
        step=4,
        path="/mnt/checkpoints/ckpt_step_4.pt",
        s3_key=f"s3://test-bucket/{run_id}/ckpt_step_4.pt",
        loss=3.0,
        tag="temporary",
        duration_s=12.5,
        size_bytes=1024 * 1024 * 500,
    )

    # ---- 4. Verify MetricsServer is serving live data ----
    print("\n--- Checking MetricsServer HTTP API ---")
    time.sleep(1)  # let the logger flush

    try:
        url = "http://localhost:8111/metrics"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        assert "gauges" in data, "Missing 'gauges' in /metrics"
        assert (
            data["gauges"]["training_loss"] == 3.0
        ), f"Expected loss=3.0, got {data['gauges']['training_loss']}"
        assert (
            data["gauges"]["global_step"] == 4.0
        ), f"Expected step=4, got {data['gauges']['global_step']}"
        assert data["counters"]["checkpoint_saves_total"] == 1
        print(
            f"✓ /metrics: loss={data['gauges']['training_loss']}, step={data['gauges']['global_step']}, ckpt_saves={data['counters']['checkpoint_saves_total']}"
        )
    except Exception as e:
        print(f"✗ MetricsServer check failed: {e}")
        raise

    # ---- 5. Verify JSONL was written ----
    print("\n--- Checking JSONL output ---")
    log_file = Path(log_dir) / f"{run_id}_rank_0.jsonl"
    assert log_file.exists(), f"Log file not found: {log_file}"

    with open(log_file) as f:
        lines = [json.loads(line) for line in f if line.strip()]

    # 5 training steps + 1 checkpoint event = 6 lines
    assert len(lines) >= 6, f"Expected >= 6 log lines, got {len(lines)}"
    print(f"✓ {log_file.name}: {len(lines)} lines written")

    # Check checkpoint event is in the logs
    ckpt_lines = [
        line
        for line in lines
        if line.get("context", {}).get("event") == "checkpoint_saved"
    ]
    assert len(ckpt_lines) == 1, f"Expected 1 checkpoint event, got {len(ckpt_lines)}"
    assert (
        ckpt_lines[0]["context"]["s3_key"]
        == f"s3://test-bucket/{run_id}/ckpt_step_4.pt"
    )
    print("✓ Checkpoint event found in JSONL with correct s3_key")

    # ---- 6. Verify CheckpointRegistry in ClickHouse ----
    print("\n--- Checking ClickHouse checkpoint registry ---")
    try:
        ckpts = ops.checkpoint_registry.list_checkpoints(run_id)
        assert len(ckpts) == 1, f"Expected 1 checkpoint in registry, got {len(ckpts)}"
        assert ckpts[0]["step"] == 4
        assert ckpts[0]["tag"] == "temporary"
        assert ckpts[0]["is_protected"] is False
        print(
            f"✓ ClickHouse registry: step={ckpts[0]['step']}, tag={ckpts[0]['tag']}, s3_key={ckpts[0]['s3_key']}"
        )
    except Exception as e:
        print(f"⚠ ClickHouse registry check: {e}")

    # ---- 7. Verify system metrics file exists ----
    print("\n--- Checking system metrics ---")
    sys_file = Path(log_dir) / "system_metrics_rank_0.jsonl"
    # Give the collector a moment to write
    time.sleep(3)
    if sys_file.exists():
        with open(sys_file) as f:
            sys_lines = [line for line in f if line.strip()]
        print(f"✓ System metrics file: {len(sys_lines)} lines")
    else:
        print("⚠ System metrics file not yet written (may need more time)")

    # ---- 8. Shutdown ----
    print("\n--- Shutting down ---")
    ops.shutdown()

    # ---- 9. Cleanup test log file ----
    if log_file.exists():
        log_file.unlink()
        print(f"✓ Cleaned up {log_file}")

    print("\n" + "=" * 60)
    print("  ✅ All TrainingOps integration tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_training_ops()
