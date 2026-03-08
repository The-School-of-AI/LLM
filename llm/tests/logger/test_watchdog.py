import json
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

from llm.logger.p12.watchdog.watchdog import Watchdog


def test_watchdog_tracks_core_signals():
    with tempfile.TemporaryDirectory() as td:
        tracking_path = f"{td}/watchdog_metrics.jsonl"
        wd = Watchdog(
            metrics_url="http://unused",
            poll_interval=1,
            tracking_log_path=tracking_path,
            loss_window_size=10,
            throughput_window_size=10,
            min_window_points=2,
        )

        snapshots = [
            {"training_loss": 2.0, "tokens_per_second": 100.0, "global_step": 1},
            {"training_loss": 2.0, "tokens_per_second": 100.0, "global_step": 2},
            {"training_loss": 5.0, "tokens_per_second": 70.0, "global_step": 3},
            {"training_loss": float("nan"), "tokens_per_second": 100.0, "global_step": 4},
        ]

        def fake_fetch():
            if not snapshots:
                return 0, {}
            payload = snapshots.pop(0)
            return 1, {"gauges": payload}

        wd._fetch_metrics_snapshot = fake_fetch  # type: ignore[method-assign]
        wd._collect_gpu_mem_max_percent = lambda: (96.0, 2, "mock")  # type: ignore[method-assign]
        wd._collect_disk_min_free_percent = lambda: 5.0  # type: ignore[method-assign]
        wd._update_process_state = lambda: {  # type: ignore[method-assign]
            "training_process_known": 1,
            "training_process_pid": 9999,
            "training_process_up": 1,
            "signal_training_process_crash": 0,
            "training_process_crash_events_total": 0,
            "training_process_restart_events_total": 0,
        }

        r1 = wd.collect_tracking_metrics()
        r2 = wd.collect_tracking_metrics()
        r3 = wd.collect_tracking_metrics()
        r4 = wd.collect_tracking_metrics()

        # Base metrics present
        assert r1["metrics_server_up"] == 1
        assert r1["signal_metrics_server_down"] == 0
        assert r1["training_loss"] == 2.0
        assert r1["tokens_per_second"] == 100.0

        # GPU and disk tracking
        assert r1["gpu_mem_max_percent"] == 96.0
        assert r1["signal_gpu_oom_risk"] == 1
        assert r1["disk_min_free_percent"] == 5.0
        assert r1["signal_disk_low_space"] == 1

        # Rolling signals
        assert r2["signal_loss_spike"] == 0
        assert r2["signal_throughput_drop"] == 0
        assert r3["signal_loss_spike"] == 1
        assert r3["signal_throughput_drop"] == 1

        # NaN detection
        assert r4["signal_nan_loss"] == 1

        with open(tracking_path, "r") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 4


def test_watchdog_process_crash_and_restart_counters():
    with tempfile.TemporaryDirectory() as td:
        tracking_path = f"{td}/watchdog_metrics.jsonl"
        wd = Watchdog(
            metrics_url="http://unused",
            poll_interval=1,
            tracking_log_path=tracking_path,
            training_pid=1234,
            min_window_points=1,
        )

        identities = [
            (1234, 10.0),  # up
            None,  # crash (down)
            (1234, 20.0),  # restart with new identity
        ]

        wd._fetch_metrics_snapshot = lambda: (0, {})  # type: ignore[method-assign]
        wd._collect_gpu_mem_max_percent = lambda: (None, 0, "unavailable")  # type: ignore[method-assign]
        wd._collect_disk_min_free_percent = lambda: None  # type: ignore[method-assign]
        wd._resolve_training_pid = lambda: 1234  # type: ignore[method-assign]
        wd._get_process_identity = lambda pid: identities.pop(0)  # type: ignore[method-assign]

        r1 = wd.collect_tracking_metrics()
        r2 = wd.collect_tracking_metrics()
        r3 = wd.collect_tracking_metrics()

        assert r1["training_process_up"] == 1
        assert r1["training_process_crash_events_total"] == 0
        assert r1["training_process_restart_events_total"] == 0

        assert r2["training_process_up"] == 0
        assert r2["signal_training_process_crash"] == 1
        assert r2["training_process_crash_events_total"] == 1
        assert r2["training_process_restart_events_total"] == 0

        assert r3["training_process_up"] == 1
        assert r3["training_process_crash_events_total"] == 1
        assert r3["training_process_restart_events_total"] == 1

        # metrics server down should be explicitly signaled
        assert r1["metrics_server_up"] == 0
        assert r1["signal_metrics_server_down"] == 1


def test_watchdog_nvidia_smi_fallback_parser():
    with tempfile.TemporaryDirectory() as td:
        tracking_path = f"{td}/watchdog_metrics.jsonl"
        wd = Watchdog(
            metrics_url="http://unused",
            poll_interval=1,
            tracking_log_path=tracking_path,
        )
        wd._gpu_probe_mode = "nvidia-smi"

        fake_result = SimpleNamespace(
            returncode=0,
            stdout="9500, 10000\n1000, 4000\n",
            stderr="",
        )
        with patch("subprocess.run", return_value=fake_result):
            max_percent, device_count, source = wd._collect_gpu_mem_max_percent()

        assert max_percent == 95.0
        assert device_count == 2
        assert source == "nvidia-smi"


def test_watchdog_trigger_pause_retrigger_updates_file():
    with tempfile.TemporaryDirectory() as td:
        control_path = f"{td}/training_control.flag"
        tracking_path = f"{td}/watchdog_metrics.jsonl"
        wd = Watchdog(
            metrics_url="http://unused",
            poll_interval=1,
            tracking_log_path=tracking_path,
            control_file_path=control_path,
        )

        wd.trigger_pause("first crash")
        with open(control_path, "r") as f:
            payload1 = json.load(f)
        assert payload1["retriggered"] is False
        assert payload1["trigger_count"] == 1
        assert payload1["reason"] == "first crash"

        wd.trigger_pause("second crash")
        with open(control_path, "r") as f:
            payload2 = json.load(f)
        assert payload2["retriggered"] is True
        assert payload2["trigger_count"] == 2
        assert payload2["reason"] == "second crash"
        assert payload2["previous_reason"] == "first crash"


def test_watchdog_retries_pynvml_after_fallback_interval():
    with tempfile.TemporaryDirectory() as td:
        tracking_path = f"{td}/watchdog_metrics.jsonl"
        wd = Watchdog(
            metrics_url="http://unused",
            poll_interval=1,
            tracking_log_path=tracking_path,
            pynvml_retry_interval_s=5,
        )
        wd._gpu_probe_mode = "nvidia-smi"
        wd._last_pynvml_attempt_ts = time.time() - 100

        wd._try_collect_gpu_mem_via_pynvml = MagicMock(return_value=(88.0, 1, "pynvml"))  # type: ignore[method-assign]
        wd._try_collect_gpu_mem_via_nvidia_smi = MagicMock(return_value=(70.0, 1, "nvidia-smi"))  # type: ignore[method-assign]

        max_percent, device_count, source = wd._collect_gpu_mem_max_percent()
        assert max_percent == 88.0
        assert device_count == 1
        assert source == "pynvml"
        assert wd._try_collect_gpu_mem_via_pynvml.called


def test_watchdog_poll_once_dispatches_alerter_events():
    with tempfile.TemporaryDirectory() as td:
        tracking_path = f"{td}/watchdog_metrics.jsonl"
        wd = Watchdog(
            metrics_url="http://unused",
            poll_interval=1,
            tracking_log_path=tracking_path,
        )

        wd._fetch_metrics_snapshot = lambda: (1, {"gauges": {}})  # type: ignore[method-assign]
        wd._collect_gpu_mem_max_percent = lambda: (None, 0, "unavailable")  # type: ignore[method-assign]
        wd._collect_disk_min_free_percent = lambda: None  # type: ignore[method-assign]
        wd._update_process_state = lambda: {  # type: ignore[method-assign]
            "training_process_known": 0,
            "training_process_pid": None,
            "training_process_up": None,
            "signal_training_process_crash": 0,
            "training_process_crash_events_total": 0,
            "training_process_restart_events_total": 0,
        }

        fake_alerter = MagicMock()
        fake_alerter.evaluate.return_value = [{"signal_key": "signal_nan_loss"}]
        wd._alerter = fake_alerter

        row = wd.poll_once()

        fake_alerter.evaluate.assert_called_once_with(row)
        assert wd.last_alert_events == [{"signal_key": "signal_nan_loss"}]


def test_watchdog_alert_resend_interval_floor_matches_alerter():
    with tempfile.TemporaryDirectory() as td:
        tracking_path = f"{td}/watchdog_metrics.jsonl"
        wd = Watchdog(
            metrics_url="http://unused",
            poll_interval=1,
            tracking_log_path=tracking_path,
            alert_resend_interval_s=30,
        )
        assert wd.alert_resend_interval_s == 60


if __name__ == "__main__":
    test_watchdog_tracks_core_signals()
    test_watchdog_process_crash_and_restart_counters()
    test_watchdog_nvidia_smi_fallback_parser()
    test_watchdog_trigger_pause_retrigger_updates_file()
    test_watchdog_retries_pynvml_after_fallback_interval()
    test_watchdog_poll_once_dispatches_alerter_events()
    test_watchdog_alert_resend_interval_floor_matches_alerter()
    print("\n✅ All Watchdog tracking tests passed!")
