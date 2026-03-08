import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from llm.logger.p12.watchdog.alerter import Alerter


def test_alerter_transition_and_resend_state_machine():
    with tempfile.TemporaryDirectory() as td:
        alert_log_path = f"{td}/watchdog_alerts.jsonl"
        a = Alerter(
            telegram_bot_token="token",
            telegram_chat_id="chat",
            resend_interval_s=300,
            alert_log_path=alert_log_path,
        )

        sent_messages = []

        def fake_send(text: str):
            sent_messages.append(text)
            return True, None, 200

        a._send_telegram = fake_send  # type: ignore[method-assign]

        # INACTIVE -> FIRING
        row1 = {"timestamp": 1000, "signal_nan_loss": 1, "training_loss": "nan"}
        e1 = a.evaluate(row1)
        assert len(e1) == 1
        assert e1[0]["state"] == "NEED_ATTENTION"
        assert "NaN Loss" in e1[0]["signal_label"]
        assert len(sent_messages) == 1

        # FIRING -> FIRING (within resend interval): no event
        row2 = {"timestamp": 1100, "signal_nan_loss": 1, "training_loss": "nan"}
        e2 = a.evaluate(row2)
        assert e2 == []
        assert len(sent_messages) == 1

        # FIRING -> FIRING (after resend interval): resend
        row3 = {"timestamp": 1400, "signal_nan_loss": 1, "training_loss": "nan"}
        e3 = a.evaluate(row3)
        assert len(e3) == 1
        assert e3[0]["state"] == "NEED_ATTENTION_REMINDER"
        assert "(reminder)" in sent_messages[1]
        assert len(sent_messages) == 2

        # FIRING -> INACTIVE: resolved immediately
        row4 = {"timestamp": 1410, "signal_nan_loss": 0, "training_loss": 2.0}
        e4 = a.evaluate(row4)
        assert len(e4) == 1
        assert e4[0]["state"] == "RESOLVED"
        assert len(sent_messages) == 3

        with open(alert_log_path, "r") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 3


def test_alerter_none_signal_treated_as_inactive():
    with tempfile.TemporaryDirectory() as td:
        a = Alerter(
            telegram_bot_token="token",
            telegram_chat_id="chat",
            resend_interval_s=300,
            alert_log_path=f"{td}/watchdog_alerts.jsonl",
        )
        a._send_telegram = lambda text: (True, None, 200)  # type: ignore[method-assign]

        events = a.evaluate(
            {
                "timestamp": 1000,
                "signal_loss_spike": None,
                "loss_spike_ratio": 3.2,
            }
        )
        assert events == []


def test_alerter_telegram_failure_never_crashes_and_logs():
    with tempfile.TemporaryDirectory() as td:
        alert_log_path = f"{td}/watchdog_alerts.jsonl"
        a = Alerter(
            telegram_bot_token="token",
            telegram_chat_id="chat",
            resend_interval_s=300,
            alert_log_path=alert_log_path,
        )

        def failing_send(text: str):
            return False, "network down", None

        a._send_telegram = failing_send  # type: ignore[method-assign]

        events = a.evaluate(
            {
                "timestamp": 1000,
                "signal_metrics_server_down": 1,
                "metrics_server_up": 0,
            }
        )
        assert len(events) == 1
        assert events[0]["state"] == "NEED_ATTENTION"
        assert events[0]["telegram_sent"] is False
        assert events[0]["telegram_error"] == "network down"

        with open(alert_log_path, "r") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 1
        assert lines[0]["telegram_sent"] is False


def test_alerter_independent_signal_states():
    with tempfile.TemporaryDirectory() as td:
        a = Alerter(
            telegram_bot_token="token",
            telegram_chat_id="chat",
            resend_interval_s=300,
            alert_log_path=f"{td}/watchdog_alerts.jsonl",
        )
        a._send_telegram = lambda text: (True, None, 200)  # type: ignore[method-assign]

        events = a.evaluate(
            {
                "timestamp": 1000,
                "signal_nan_loss": 1,
                "training_loss": "nan",
                "signal_disk_low_space": 1,
                "disk_min_free_percent": 7.5,
            }
        )
        assert len(events) == 2
        keys = {e["signal_key"] for e in events}
        assert "signal_nan_loss" in keys
        assert "signal_disk_low_space" in keys


def test_alerter_persists_last_sent_state_across_restart():
    with tempfile.TemporaryDirectory() as td:
        alert_log_path = f"{td}/watchdog_alerts.jsonl"
        state_path = f"{td}/watchdog_alert_state.json"

        a1 = Alerter(
            telegram_bot_token="token",
            telegram_chat_id="chat",
            resend_interval_s=300,
            alert_log_path=alert_log_path,
            state_path=state_path,
        )
        a1._send_telegram = lambda text: (True, None, 200)  # type: ignore[method-assign]

        first = a1.evaluate(
            {"timestamp": 1000, "signal_nan_loss": 1, "training_loss": "nan"}
        )
        assert len(first) == 1
        assert first[0]["state"] == "NEED_ATTENTION"

        # Simulate process restart: new instance should restore firing + last_sent_ts.
        a2 = Alerter(
            telegram_bot_token="token",
            telegram_chat_id="chat",
            resend_interval_s=300,
            alert_log_path=alert_log_path,
            state_path=state_path,
        )
        a2._send_telegram = lambda text: (True, None, 200)  # type: ignore[method-assign]

        # 10 seconds later: should NOT resend due to restored last_sent_ts.
        no_resend = a2.evaluate(
            {"timestamp": 1010, "signal_nan_loss": 1, "training_loss": "nan"}
        )
        assert no_resend == []

        # After interval: reminder should fire.
        resend = a2.evaluate(
            {"timestamp": 1301, "signal_nan_loss": 1, "training_loss": "nan"}
        )
        assert len(resend) == 1
        assert resend[0]["state"] == "NEED_ATTENTION_REMINDER"


def test_alerter_enforces_resend_floor_and_timeout_split():
    with tempfile.TemporaryDirectory() as td:
        a = Alerter(
            telegram_bot_token="token",
            telegram_chat_id="chat",
            resend_interval_s=5,
            alert_log_path=f"{td}/watchdog_alerts.jsonl",
            state_path=f"{td}/watchdog_alert_state.json",
            telegram_connect_timeout_s=1.0,
            telegram_read_timeout_s=2.5,
        )
        assert a.resend_interval_s == 60
        with patch(
            "llm.logger.p12.watchdog.alerter.requests.post",
            return_value=SimpleNamespace(status_code=200),
        ) as mocked_post:
            sent, error, code = a._send_telegram("hello")
        assert sent is True
        assert error is None
        assert code == 200
        assert mocked_post.call_count == 1
        assert mocked_post.call_args.kwargs["timeout"] == (1.0, 2.5)


def test_alerter_save_failure_records_internal_event():
    with tempfile.TemporaryDirectory() as td:
        alert_log_path = f"{td}/watchdog_alerts.jsonl"
        state_path = f"{td}/watchdog_alert_state.json"
        a = Alerter(
            telegram_bot_token="token",
            telegram_chat_id="chat",
            resend_interval_s=300,
            alert_log_path=alert_log_path,
            state_path=state_path,
        )
        with patch("llm.logger.p12.watchdog.alerter.os.replace", side_effect=OSError("disk full")):
            saved = a._save_state()
        assert saved is False
        assert a.state_save_failures_total == 1

        with open(alert_log_path, "r") as f:
            records = [json.loads(line) for line in f if line.strip()]
        assert len(records) >= 1
        last = records[-1]
        assert last["event_type"] == "ALERTER_STATE_ERROR"
        assert last["operation"] == "save"
        assert "disk full" in last["error"]


def test_alerter_load_failure_records_internal_event():
    with tempfile.TemporaryDirectory() as td:
        alert_log_path = f"{td}/watchdog_alerts.jsonl"
        state_path = f"{td}/watchdog_alert_state.json"
        with open(state_path, "w") as f:
            f.write("{ this is not json")

        a = Alerter(
            telegram_bot_token="token",
            telegram_chat_id="chat",
            resend_interval_s=300,
            alert_log_path=alert_log_path,
            state_path=state_path,
        )
        assert a.state_load_failures_total == 1

        with open(alert_log_path, "r") as f:
            records = [json.loads(line) for line in f if line.strip()]
        assert len(records) >= 1
        first = records[0]
        assert first["event_type"] == "ALERTER_STATE_ERROR"
        assert first["operation"] == "load"


def test_alerter_state_error_warning_is_throttled():
    with tempfile.TemporaryDirectory() as td:
        a = Alerter(
            telegram_bot_token="token",
            telegram_chat_id="chat",
            resend_interval_s=300,
            alert_log_path=f"{td}/watchdog_alerts.jsonl",
            state_path=f"{td}/watchdog_alert_state.json",
        )
        with patch("builtins.print") as mock_print:
            a._record_state_error("save", "first")
            a._record_state_error("save", "second")
        assert mock_print.call_count == 1


def test_alerter_writes_backend_events_in_vector_shape():
    with tempfile.TemporaryDirectory() as td:
        alert_log_path = f"{td}/watchdog_alerts.jsonl"
        backend_events_path = f"{td}/training_logs/watchdog/watchdog_events.jsonl"
        a = Alerter(
            telegram_bot_token="token",
            telegram_chat_id="chat",
            resend_interval_s=300,
            alert_log_path=alert_log_path,
            state_path=f"{td}/watchdog_alert_state.json",
            backend_events_log_path=backend_events_path,
            run_id="run_test_123",
            rank=7,
        )
        a._send_telegram = lambda text: (True, None, 200)  # type: ignore[method-assign]

        events = a.evaluate(
            {
                "timestamp": 1000,
                "global_step": 42,
                "signal_nan_loss": 1,
                "training_loss": "nan",
            }
        )
        assert len(events) == 1

        with open(backend_events_path, "r") as f:
            records = [json.loads(line) for line in f if line.strip()]
        assert len(records) == 1
        rec = records[0]
        assert rec["run_id"] == "run_test_123"
        assert rec["rank"] == 7
        assert rec["step"] == 42
        assert rec["context"]["event"] == "event"
        assert rec["context"]["event_type"] == "watchdog_alert"
        assert rec["context"]["payload"]["signal_key"] == "signal_nan_loss"


if __name__ == "__main__":
    test_alerter_transition_and_resend_state_machine()
    test_alerter_none_signal_treated_as_inactive()
    test_alerter_telegram_failure_never_crashes_and_logs()
    test_alerter_independent_signal_states()
    test_alerter_persists_last_sent_state_across_restart()
    test_alerter_enforces_resend_floor_and_timeout_split()
    test_alerter_save_failure_records_internal_event()
    test_alerter_load_failure_records_internal_event()
    test_alerter_state_error_warning_is_throttled()
    test_alerter_writes_backend_events_in_vector_shape()
    print("\n✅ All alerter tests passed!")
