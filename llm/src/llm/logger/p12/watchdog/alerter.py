import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


class Alerter:
    """
    Stateless input + stateful per-signal alert transitions.

    Public API:
        evaluate(row: dict) -> list[dict]
    """

    _STATE_NEED_ATTENTION = "NEED_ATTENTION"
    _STATE_NEED_ATTENTION_REMINDER = "NEED_ATTENTION_REMINDER"
    _STATE_RESOLVED = "RESOLVED"
    _STATE_ERROR_WARN_INTERVAL_S = 60.0
    _BACKEND_EVENT_TYPE = "watchdog_alert"

    _SIGNAL_LABELS = {
        "signal_nan_loss": "NaN Loss",
        "signal_loss_spike": "Loss Spike > 2x Rolling Average",
        "signal_throughput_drop": "Throughput Drop > 20%",
        "signal_gpu_oom_risk": "GPU Memory > 95% (OOM Imminent)",
        "signal_disk_low_space": "Disk Space < 10%",
        "signal_training_process_crash": "Training Process Crash",
        "signal_metrics_server_down": "Metrics Server Down",
    }

    _SIGNAL_VALUE_KEYS = {
        "signal_nan_loss": "training_loss",
        "signal_loss_spike": "loss_spike_ratio",
        "signal_throughput_drop": "throughput_drop_ratio",
        "signal_gpu_oom_risk": "gpu_mem_max_percent",
        "signal_disk_low_space": "disk_min_free_percent",
        "signal_training_process_crash": "training_process_crash_events_total",
        "signal_metrics_server_down": None,
    }

    def __init__(
        self,
        telegram_bot_token: str,
        telegram_chat_id: str,
        resend_interval_s: int = 300,
        alert_log_path: str = "/tmp/watchdog_alerts.jsonl",
        state_path: str = "/tmp/watchdog_alert_state.json",
        telegram_connect_timeout_s: float = 1.0,
        telegram_read_timeout_s: float = 2.5,
        backend_events_enabled: bool = True,
        backend_events_log_path: str = "/tmp/training_logs/watchdog/watchdog_events.jsonl",
        run_id: str | None = None,
        rank: int | None = None,
    ):
        self.telegram_bot_token = (telegram_bot_token or "").strip()
        self.telegram_chat_id = (telegram_chat_id or "").strip()
        try:
            requested_resend_interval = int(resend_interval_s)
        except (TypeError, ValueError):
            requested_resend_interval = 300
        if requested_resend_interval < 60:
            print(
                "⚠ Alerter resend_interval_s below 60s; clamping to 60s to prevent alert spam."
            )
        self.resend_interval_s = max(60, requested_resend_interval)
        self.telegram_connect_timeout_s = self._coerce_positive_float(
            telegram_connect_timeout_s,
            default=1.0,
            min_value=0.1,
        )
        self.telegram_read_timeout_s = self._coerce_positive_float(
            telegram_read_timeout_s,
            default=2.5,
            min_value=0.1,
        )
        self.alert_log_path = Path(alert_log_path)
        self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.backend_events_enabled = bool(backend_events_enabled)
        self.backend_events_log_path = Path(backend_events_log_path)
        if self.backend_events_enabled:
            self.backend_events_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = (
            (run_id or "").strip()
            or os.environ.get("WATCHDOG_RUN_ID", "").strip()
            or os.environ.get("RUN_ID", "").strip()
            or "watchdog"
        )
        self.host = socket.gethostname()
        self.rank = self._coerce_non_negative_int(
            rank if rank is not None else os.environ.get("RANK"),
            default=0,
        )

        # signal_key -> {"firing": bool, "last_sent_ts": float | None}
        self._signal_state: dict[str, dict] = {
            key: {"firing": False, "last_sent_ts": None}
            for key in self._SIGNAL_LABELS
        }
        self.state_save_failures_total = 0
        self.state_load_failures_total = 0
        self._last_state_error_warn_ts: dict[str, float] = {
            "load": 0.0,
            "save": 0.0,
        }
        self._load_state()

    @staticmethod
    def _utc_iso_now() -> str:
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

    @staticmethod
    def _to_float(value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        if isinstance(value, (int, float)):
            return int(value) != 0
        return False

    @staticmethod
    def _coerce_positive_float(value, default: float, min_value: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = default
        return max(min_value, parsed)

    @staticmethod
    def _coerce_non_negative_int(value, default: int = 0) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return int(default)
        return max(0, parsed)

    @staticmethod
    def _coerce_ts(value) -> float:
        try:
            if value is None:
                return time.time()
            return float(value)
        except (TypeError, ValueError):
            return time.time()

    def _warn_state_error(self, operation: str, error_message: str):
        now_ts = time.time()
        last_warn_ts = self._last_state_error_warn_ts.get(operation, 0.0)
        if now_ts - last_warn_ts < self._STATE_ERROR_WARN_INTERVAL_S:
            return
        self._last_state_error_warn_ts[operation] = now_ts
        print(f"⚠ Alerter state {operation} failed: {error_message}")

    def _record_state_error(self, operation: str, error: Exception | str):
        error_message = str(error)
        if operation == "load":
            self.state_load_failures_total += 1
        else:
            self.state_save_failures_total += 1
        self._warn_state_error(operation, error_message)

        record = {
            "timestamp": time.time(),
            "timestamp_iso": self._utc_iso_now(),
            "event_type": "ALERTER_STATE_ERROR",
            "operation": operation,
            "error": error_message,
            "state_load_failures_total": self.state_load_failures_total,
            "state_save_failures_total": self.state_save_failures_total,
        }
        self._append_alert_log(record)

    def _load_state(self) -> bool:
        try:
            if not self.state_path.exists():
                return True
            with open(self.state_path, "r") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                self._record_state_error(
                    "load",
                    "invalid state payload: expected JSON object at top level",
                )
                return False
            signals = payload.get("signals", {})
            if not isinstance(signals, dict):
                self._record_state_error(
                    "load",
                    "invalid state payload: 'signals' must be an object",
                )
                return False

            for signal_key in self._SIGNAL_LABELS:
                signal_state = signals.get(signal_key)
                if not isinstance(signal_state, dict):
                    continue
                firing = self._to_bool(signal_state.get("firing", False))
                last_sent_ts = self._to_float(signal_state.get("last_sent_ts"))
                self._signal_state[signal_key]["firing"] = firing
                self._signal_state[signal_key]["last_sent_ts"] = last_sent_ts
            return True
        except Exception as e:
            # Never crash on state load issues.
            self._record_state_error("load", e)
            return False

    def _save_state(self) -> bool:
        payload = {
            "timestamp": time.time(),
            "timestamp_iso": self._utc_iso_now(),
            "signals": self._signal_state,
        }
        tmp_path = self.state_path.with_suffix(f"{self.state_path.suffix}.tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(payload, f)
            os.replace(tmp_path, self.state_path)
            return True
        except Exception as e:
            # Never crash on state persistence issues.
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            self._record_state_error("save", e)
            return False

    @staticmethod
    def _is_active_signal(value) -> bool:
        """
        Only active when value explicitly represents 1/true.
        None is always INACTIVE.
        """
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return int(value) == 1
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "active", "firing"}:
                return True
            return False
        return False

    @staticmethod
    def _format_value(value) -> str:
        if value is None:
            return "unavailable"
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if value != value:  # NaN
                return "nan"
            if value == float("inf"):
                return "inf"
            if value == float("-inf"):
                return "-inf"
            return f"{value:.6g}"
        return str(value)

    def _detail_line(self, signal_key: str, row: dict) -> str:
        value_key = self._SIGNAL_VALUE_KEYS.get(signal_key)
        if value_key is None:
            metrics_server_up = row.get("metrics_server_up")
            return f"metrics_server_up={self._format_value(metrics_server_up)}"

        value = row.get(value_key)
        if signal_key == "signal_training_process_crash":
            up = row.get("training_process_up")
            return (
                f"{value_key}={self._format_value(value)} "
                f"training_process_up={self._format_value(up)}"
            )
        return f"{value_key}={self._format_value(value)}"

    def _build_message(
        self,
        event_state: str,
        signal_label: str,
        detail_line: str,
    ) -> str:
        if event_state == self._STATE_NEED_ATTENTION:
            status_line = "🔴 NEED ATTENTION"
        elif event_state == self._STATE_NEED_ATTENTION_REMINDER:
            status_line = "🔴 NEED ATTENTION (reminder)"
        else:
            status_line = "✅ RESOLVED"
        return f"{status_line}\n{signal_label}\n{detail_line}"

    def _send_telegram(self, text: str) -> tuple[bool, str | None, int | None]:
        """
        Returns: (sent, error, status_code)
        Never raises.
        """
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return False, "missing telegram_bot_token or telegram_chat_id", None

        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": text,
        }
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=(self.telegram_connect_timeout_s, self.telegram_read_timeout_s),
            )
            if 200 <= response.status_code < 300:
                return True, None, int(response.status_code)
            return (
                False,
                f"telegram_http_{response.status_code}",
                int(response.status_code),
            )
        except Exception as e:
            return False, str(e), None

    def _append_alert_log(self, record: dict):
        try:
            with open(self.alert_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            # Never crash watchdog on alert-log write issues.
            pass

    @staticmethod
    def _severity_for_state(event_state: str) -> str:
        if event_state == Alerter._STATE_NEED_ATTENTION:
            return "critical"
        if event_state == Alerter._STATE_NEED_ATTENTION_REMINDER:
            return "warning"
        return "info"

    def _build_backend_event_record(self, alert_event: dict, row: dict) -> dict:
        step = self._coerce_non_negative_int(
            self._to_float(row.get("global_step")),
            default=0,
        )
        severity = self._severity_for_state(str(alert_event.get("state", "")))

        payload = {
            "source": "watchdog_alerter",
            "signal_key": alert_event.get("signal_key"),
            "signal_label": alert_event.get("signal_label"),
            "state": alert_event.get("state"),
            "detail": alert_event.get("detail"),
            "value_key": alert_event.get("value_key"),
            "value": alert_event.get("value"),
            "telegram_sent": alert_event.get("telegram_sent"),
            "telegram_status_code": alert_event.get("telegram_status_code"),
            "telegram_error": alert_event.get("telegram_error"),
        }
        message = (
            f"{alert_event.get('state', 'UNKNOWN')} "
            f"{alert_event.get('signal_label', 'Unknown Signal')}: "
            f"{alert_event.get('detail', '')}"
        )
        return {
            "timestamp": self._utc_iso_now(),
            "run_id": self.run_id,
            "rank": self.rank,
            "host": self.host,
            "step": step,
            "metrics": {},
            "context": {
                "event": "event",
                "event_type": self._BACKEND_EVENT_TYPE,
                "severity": severity,
                "message": message.strip(),
                "device": 65535,
                "payload": payload,
            },
        }

    def _append_backend_event(self, alert_event: dict, row: dict):
        if not self.backend_events_enabled:
            return
        try:
            record = self._build_backend_event_record(alert_event, row)
            with open(self.backend_events_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            # Never crash watchdog on backend-event write issues.
            pass

    def _fire_event(
        self,
        *,
        now_ts: float,
        signal_key: str,
        signal_label: str,
        row: dict,
        event_state: str,
    ) -> dict:
        detail = self._detail_line(signal_key, row)
        message = self._build_message(
            event_state=event_state,
            signal_label=signal_label,
            detail_line=detail,
        )
        sent, error, status_code = self._send_telegram(message)

        value_key = self._SIGNAL_VALUE_KEYS.get(signal_key)
        value = row.get(value_key) if value_key is not None else None
        event = {
            "timestamp": now_ts,
            "timestamp_iso": self._utc_iso_now(),
            "signal_key": signal_key,
            "signal_label": signal_label,
            "state": event_state,
            "detail": detail,
            "value_key": value_key,
            "value": value,
            "telegram_sent": sent,
            "telegram_status_code": status_code,
            "telegram_error": error,
        }
        self._append_alert_log(event)
        self._append_backend_event(event, row)
        return event

    def evaluate(self, row: dict) -> list[dict]:
        """
        Evaluate all signals and emit alert events for transitions/resends.
        Returns list of event dicts fired in this evaluation cycle.

        Never raises.
        """
        events: list[dict] = []
        try:
            if not isinstance(row, dict):
                row = {}
            now_ts = self._coerce_ts(row.get("timestamp"))

            for signal_key, signal_label in self._SIGNAL_LABELS.items():
                try:
                    signal_value = row.get(signal_key)
                    is_active = self._is_active_signal(signal_value)
                    state = self._signal_state[signal_key]
                    is_firing = bool(state["firing"])
                    last_sent_ts = state["last_sent_ts"]

                    if not is_active:
                        # INACTIVE -> INACTIVE: do nothing
                        if not is_firing:
                            continue

                        # FIRING -> INACTIVE: resolved immediately
                        event = self._fire_event(
                            now_ts=now_ts,
                            signal_key=signal_key,
                            signal_label=signal_label,
                            row=row,
                            event_state=self._STATE_RESOLVED,
                        )
                        events.append(event)
                        state["firing"] = False
                        state["last_sent_ts"] = now_ts
                        self._save_state()
                        continue

                    # active signal path
                    if not is_firing:
                        # INACTIVE -> FIRING: immediate attention
                        event = self._fire_event(
                            now_ts=now_ts,
                            signal_key=signal_key,
                            signal_label=signal_label,
                            row=row,
                            event_state=self._STATE_NEED_ATTENTION,
                        )
                        events.append(event)
                        state["firing"] = True
                        state["last_sent_ts"] = now_ts
                        self._save_state()
                        continue

                    # FIRING -> FIRING: resend on interval only
                    elapsed = None
                    if last_sent_ts is not None:
                        elapsed = now_ts - float(last_sent_ts)
                    if last_sent_ts is None or (
                        elapsed is not None and elapsed >= self.resend_interval_s
                    ):
                        event = self._fire_event(
                            now_ts=now_ts,
                            signal_key=signal_key,
                            signal_label=signal_label,
                            row=row,
                            event_state=self._STATE_NEED_ATTENTION_REMINDER,
                        )
                        events.append(event)
                        state["last_sent_ts"] = now_ts
                        self._save_state()
                except Exception:
                    # isolate each signal; never break whole evaluation cycle
                    continue
        except Exception:
            return events

        return events
