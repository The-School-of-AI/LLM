import json
import os
import tempfile
import time
import urllib.parse
import urllib.request


def test_metrics_server():
    """End-to-end test for the custom metrics server (no Prometheus dependency)."""

    # Write a minimal config to a temp file
    config_fd, config_path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(config_fd, "w") as f:
        # yaml-compatible JSON
        f.write("training:\n  metrics_port: 18999\n")

    try:
        from components.metrics_server import MetricsServer

        server = MetricsServer(config_path)
        server.start()
        time.sleep(0.5)  # let HTTP server bind

        base = "http://localhost:18999"

        # ---- /health ----
        print("--- Test: /health ---")
        resp = _get(f"{base}/health")
        assert resp["status"] == "ok", f"Expected ok, got {resp}"
        print("✓ /health OK")

        # ---- /metrics (empty snapshot) ----
        print("\n--- Test: /metrics (initial) ---")
        resp = _get(f"{base}/metrics")
        assert "gauges" in resp and "counters" in resp and "info" in resp
        assert resp["gauges"]["training_loss"] == 0.0
        print(f"✓ /metrics snapshot OK (loss={resp['gauges']['training_loss']})")

        # ---- Push some training metrics ----
        print("\n--- Test: update_training_metrics ---")
        server.update_training_metrics(loss=2.5, lr=0.001, step=100, grad_norm=1.2)
        time.sleep(0.05)

        resp = _get(f"{base}/query?metric=training_loss")
        assert resp["value"] == 2.5, f"Expected 2.5, got {resp['value']}"
        print(f"✓ /query training_loss = {resp['value']}")

        resp = _get(f"{base}/query?metric=learning_rate")
        assert resp["value"] == 0.001
        print(f"✓ /query learning_rate = {resp['value']}")

        # ---- Push more and check history ----
        print("\n--- Test: /history ---")
        since_ts = time.time() - 1
        server.update_training_metrics(loss=2.3, lr=0.001, step=200)
        server.update_training_metrics(loss=2.1, lr=0.001, step=300)
        time.sleep(0.05)

        resp = _get(f"{base}/history?metric=training_loss&since={since_ts}")
        assert len(resp["data"]) >= 3, f"Expected >=3 points, got {len(resp['data'])}"
        print(f"✓ /history returned {len(resp['data'])} points")

        # ---- Counters ----
        print("\n--- Test: record_checkpoint ---")
        server.record_checkpoint(duration=5.0, success=True)
        server.record_checkpoint(duration=0, success=False)
        resp = _get(f"{base}/metrics")
        assert resp["counters"]["checkpoint_saves_total"] == 1
        assert resp["counters"]["checkpoint_failures_total"] == 1
        print("✓ Counters OK")

        # ---- Info metric ----
        print("\n--- Test: training_status info ---")
        server.update_training_status("running", "step 300")
        resp = _get(f"{base}/metrics")
        assert resp["info"]["training_status"]["status"] == "running"
        print("✓ Info metric OK")

        # ---- Unknown metric returns 404 ----
        print("\n--- Test: unknown metric ---")
        code, resp = _get_with_code(f"{base}/query?metric=nonexistent")
        assert code == 404, f"Expected 404, got {code}"
        print("✓ Unknown metric returns 404")

        # ---- Throughput ----
        print("\n--- Test: update_throughput ---")
        server.update_throughput(50000.0)
        resp = _get(f"{base}/query?metric=tokens_per_second")
        assert resp["value"] == 50000.0
        print(f"✓ tokens_per_second = {resp['value']}")

        server.stop()
        print("\n✅ All metrics server tests passed!")

    finally:
        os.unlink(config_path)


def _get(url: str) -> dict:
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def _get_with_code(url: str):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


if __name__ == "__main__":
    test_metrics_server()
