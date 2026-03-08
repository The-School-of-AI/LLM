import json
import math
import os
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import psutil
import requests

Alerter = None
_ALERTER_IMPORT_ERROR = None
try:
    from .alerter import Alerter
except Exception as relative_import_error:
    try:
        from llm.logger.p12.watchdog.alerter import Alerter
    except Exception as absolute_import_error:
        Alerter = None
        _ALERTER_IMPORT_ERROR = (
            f"relative import failed: {relative_import_error}; "
            f"absolute import failed: {absolute_import_error}"
        )


class Watchdog:
    """
    Tracking plane for training reliability signals.

    This class always computes and records signal inputs.
    It can optionally dispatch alerts via the Alerter when configured.
    """

    def __init__(
        self,
        metrics_url: str = "http://localhost:8000",
        control_file_path: str = "/tmp/training_control.flag",
        poll_interval: int = 5,
        loss_threshold: float = 10.0,
        tracking_log_path: str = "/tmp/watchdog_metrics.jsonl",
        loss_window_size: int = 60,
        throughput_window_size: int = 60,
        min_window_points: int = 5,
        training_pid: int | None = None,
        training_pid_file_path: str | None = None,
        disk_paths: list[str] | None = None,
        request_timeout_s: float = 2.0,
        pynvml_retry_interval_s: float = 60.0,
        enable_alerter: bool | None = None,
        telegram_bot_token: str | None = None,
        telegram_chat_id: str | None = None,
        alert_resend_interval_s: int = 300,
        alert_log_path: str = "/tmp/watchdog_alerts.jsonl",
    ):
        self.metrics_url = metrics_url
        self.control_file = Path(control_file_path)
        self.poll_interval = poll_interval
        self.loss_threshold = loss_threshold
        self.tracking_log = Path(tracking_log_path)
        self.min_window_points = max(1, int(min_window_points))
        self.loss_history: deque[float] = deque(maxlen=max(1, int(loss_window_size)))
        self.throughput_history: deque[float] = deque(
            maxlen=max(1, int(throughput_window_size))
        )
        self.training_pid = training_pid
        self.training_pid_file_path = (
            Path(training_pid_file_path) if training_pid_file_path else None
        )
        self.disk_paths = self._resolve_disk_paths(disk_paths)
        self.request_timeout_s = request_timeout_s
        self.pynvml_retry_interval_s = max(5.0, float(pynvml_retry_interval_s))
        try:
            requested_alert_resend_interval = int(alert_resend_interval_s)
        except (TypeError, ValueError):
            requested_alert_resend_interval = 300
        if requested_alert_resend_interval < 60:
            print(
                "⚠ Watchdog alert_resend_interval_s below 60s; clamping to 60s for consistency."
            )
        self.alert_resend_interval_s = max(60, requested_alert_resend_interval)

        self.training_process_crash_events_total = 0
        self.training_process_restart_events_total = 0
        self._tracked_identity: tuple[int, float] | None = None
        self._last_training_up: bool | None = None
        self.pause_trigger_count = 0
        self.last_alert_events: list[dict] = []

        self._gpu_probe_mode = "auto"  # auto -> pynvml -> nvidia-smi -> unavailable
        self._last_pynvml_attempt_ts = 0.0
        self._pynvml = None
        self._pynvml_device_count = 0
        self._alerter = None
        self.running = True

        self.tracking_log.parent.mkdir(parents=True, exist_ok=True)
        self._setup_alerter(
            enable_alerter=enable_alerter,
            telegram_bot_token=telegram_bot_token,
            telegram_chat_id=telegram_chat_id,
            alert_log_path=alert_log_path,
        )

        print("✓ Watchdog Initialized")
        print(f"  Monitoring: {self.metrics_url}")
        print(f"  Tracking Log: {self.tracking_log}")
        print(f"  Control File: {self.control_file}")
        print(f"  Loss Threshold: {self.loss_threshold}")
        print(f"  Disk Paths: {self.disk_paths}")
        if self.training_pid is not None:
            print(f"  Training PID: {self.training_pid}")
        elif self.training_pid_file_path is not None:
            print(f"  Training PID File: {self.training_pid_file_path}")
        if self._alerter is not None:
            print("  Alerter: enabled")
            print(f"  Alert Log: {alert_log_path}")
            print(f"  Alert Resend Interval: {self.alert_resend_interval_s}s")
        else:
            print("  Alerter: disabled")

    def _setup_alerter(
        self,
        *,
        enable_alerter: bool | None,
        telegram_bot_token: str | None,
        telegram_chat_id: str | None,
        alert_log_path: str,
    ):
        token = (
            (telegram_bot_token or "").strip()
            or os.environ.get("WATCHDOG_TELEGRAM_BOT_TOKEN", "").strip()
            or os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        )
        chat_id = (
            (telegram_chat_id or "").strip()
            or os.environ.get("WATCHDOG_TELEGRAM_CHAT_ID", "").strip()
            or os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        )

        if enable_alerter is None:
            should_enable = bool(token and chat_id)
        else:
            should_enable = bool(enable_alerter)

        if not should_enable:
            return

        if not token or not chat_id:
            print("⚠ Watchdog alerter enabled but missing Telegram token/chat_id; disabled.")
            return
        if Alerter is None:
            if enable_alerter is True:
                print("⚠ Watchdog alerter explicitly enabled but unavailable (import failed).")
            else:
                print("⚠ Watchdog alerter unavailable (import failed).")
            if _ALERTER_IMPORT_ERROR:
                print(f"  Import error: {_ALERTER_IMPORT_ERROR}")
            print("  Alerter disabled.")
            return

        try:
            self._alerter = Alerter(
                telegram_bot_token=token,
                telegram_chat_id=chat_id,
                resend_interval_s=self.alert_resend_interval_s,
                alert_log_path=alert_log_path,
            )
        except Exception as e:
            self._alerter = None
            print(f"⚠ Watchdog alerter initialization failed: {e}")

    @staticmethod
    def _unique(items: list[str]) -> list[str]:
        seen = set()
        ordered = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered

    def _resolve_disk_paths(self, disk_paths: list[str] | None) -> list[str]:
        """
        Build disk paths to monitor, preferring checkpoint-related mounts.
        """
        candidates: list[str] = []
        if disk_paths:
            candidates.extend(p for p in disk_paths if p)

        env_list = os.environ.get("WATCHDOG_DISK_PATHS", "")
        if env_list:
            candidates.extend(p.strip() for p in env_list.split(",") if p.strip())

        for key in (
            "CHECKPOINT_DIR",
            "CHECKPOINTS_DIR",
            "CHECKPOINT_PATH",
            "CHECKPOINT_OUTPUT_DIR",
            "OUTPUT_DIR",
        ):
            value = os.environ.get(key)
            if value:
                candidates.append(value.strip())

        normalized: list[str] = []
        for path in candidates:
            if not path:
                continue
            try:
                normalized.append(str(Path(path).resolve()))
            except Exception:
                normalized.append(path)

        normalized = self._unique(normalized)
        existing = [p for p in normalized if Path(p).exists()]
        if existing:
            return existing

        print("⚠ Watchdog disk_paths not provided/found; falling back to ['/']")
        print("  Set disk_paths=[...] or WATCHDOG_DISK_PATHS for checkpoint volume checks.")
        return ["/"]

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
    def _mean(values: deque[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    def _append_tracking_row(self, row: dict):
        with open(self.tracking_log, "a") as f:
            f.write(json.dumps(row) + "\n")

    def _fetch_metrics_snapshot(self) -> tuple[int, dict]:
        """
        Return (metrics_server_up, payload). metrics_server_up is 1/0.
        """
        try:
            response = requests.get(
                f"{self.metrics_url}/metrics",
                timeout=self.request_timeout_s,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                return 0, {}
            return 1, data
        except Exception:
            return 0, {}

    def _resolve_training_pid(self) -> int | None:
        if self.training_pid is not None:
            return int(self.training_pid)
        if self.training_pid_file_path is None or not self.training_pid_file_path.exists():
            return None
        try:
            raw = self.training_pid_file_path.read_text().strip()
            if not raw:
                return None
            return int(raw.splitlines()[0].strip())
        except Exception:
            return None

    @staticmethod
    def _get_process_identity(pid: int | None) -> tuple[int, float] | None:
        if pid is None:
            return None
        try:
            proc = psutil.Process(pid)
            return (pid, float(proc.create_time()))
        except Exception:
            return None

    def _update_process_state(self) -> dict:
        pid = self._resolve_training_pid()
        process_known = pid is not None
        current_identity = self._get_process_identity(pid)
        process_up = current_identity is not None

        crash_event = 0
        restart_event = 0

        if process_known:
            if self._tracked_identity is None and current_identity is not None:
                self._tracked_identity = current_identity
            elif self._tracked_identity is not None:
                if current_identity is None:
                    if self._last_training_up is True:
                        crash_event = 1
                elif current_identity != self._tracked_identity:
                    restart_event = 1
                    # If we never observed a down transition, count this as a missed crash.
                    if self._last_training_up is True:
                        crash_event = 1
                    self._tracked_identity = current_identity

        if crash_event:
            self.training_process_crash_events_total += 1
        if restart_event:
            self.training_process_restart_events_total += 1

        self._last_training_up = process_up if process_known else None

        return {
            "training_process_known": 1 if process_known else 0,
            "training_process_pid": pid,
            "training_process_up": (
                1 if process_up else 0 if process_known else None
            ),
            "signal_training_process_crash": 1 if (process_known and not process_up) else 0,
            "training_process_crash_events_total": self.training_process_crash_events_total,
            "training_process_restart_events_total": self.training_process_restart_events_total,
        }

    def _try_collect_gpu_mem_via_pynvml(self) -> tuple[float | None, int, str]:
        if self._gpu_probe_mode not in ("auto", "pynvml"):
            return None, 0, "unavailable"
        self._last_pynvml_attempt_ts = time.time()
        try:
            if self._pynvml is None:
                import pynvml

                pynvml.nvmlInit()
                self._pynvml = pynvml
                self._pynvml_device_count = pynvml.nvmlDeviceGetCount()

            if self._pynvml_device_count <= 0:
                self._gpu_probe_mode = "nvidia-smi"
                return None, 0, "unavailable"

            max_percent = None
            for i in range(self._pynvml_device_count):
                handle = self._pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
                if int(mem.total) <= 0:
                    continue
                percent = float(mem.used) / float(mem.total) * 100.0
                max_percent = percent if max_percent is None else max(max_percent, percent)
            self._gpu_probe_mode = "pynvml"
            return (
                round(max_percent, 2) if max_percent is not None else None,
                int(self._pynvml_device_count),
                "pynvml",
            )
        except Exception:
            self._gpu_probe_mode = "nvidia-smi"
            return None, 0, "unavailable"

    def _try_collect_gpu_mem_via_nvidia_smi(self) -> tuple[float | None, int, str]:
        if self._gpu_probe_mode not in ("auto", "nvidia-smi"):
            return None, 0, "unavailable"
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=self.request_timeout_s,
            )
            if result.returncode != 0:
                self._gpu_probe_mode = "unavailable"
                return None, 0, "unavailable"

            max_percent = None
            device_count = 0
            for line in result.stdout.splitlines():
                raw = line.strip()
                if not raw:
                    continue
                parts = [p.strip() for p in raw.split(",")]
                if len(parts) != 2:
                    continue
                used = self._to_float(parts[0])
                total = self._to_float(parts[1])
                if used is None or total is None or total <= 0:
                    continue
                device_count += 1
                percent = (used / total) * 100.0
                max_percent = percent if max_percent is None else max(max_percent, percent)

            if device_count == 0:
                self._gpu_probe_mode = "unavailable"
                return None, 0, "unavailable"

            self._gpu_probe_mode = "nvidia-smi"
            return round(max_percent, 2), device_count, "nvidia-smi"
        except Exception:
            self._gpu_probe_mode = "unavailable"
            return None, 0, "unavailable"

    def _collect_gpu_mem_max_percent(self) -> tuple[float | None, int, str]:
        # Retry NVML periodically even after a fallback lock-in.
        now = time.time()
        if self._gpu_probe_mode in ("nvidia-smi", "unavailable"):
            elapsed = now - self._last_pynvml_attempt_ts
            if elapsed >= self.pynvml_retry_interval_s:
                self._gpu_probe_mode = "auto"

        if self._gpu_probe_mode in ("auto", "pynvml"):
            max_percent, device_count, source = self._try_collect_gpu_mem_via_pynvml()
            if source != "unavailable":
                return max_percent, device_count, source
        if self._gpu_probe_mode in ("auto", "nvidia-smi", "unavailable"):
            max_percent, device_count, source = self._try_collect_gpu_mem_via_nvidia_smi()
            if source != "unavailable":
                return max_percent, device_count, source
        return None, 0, "unavailable"

    def _collect_disk_min_free_percent(self) -> float | None:
        min_free = None
        for path in self.disk_paths:
            try:
                usage = psutil.disk_usage(path)
                if usage.total <= 0:
                    continue
                free_percent = float(usage.free) / float(usage.total) * 100.0
                min_free = free_percent if min_free is None else min(min_free, free_percent)
            except Exception:
                continue
        return round(min_free, 2) if min_free is not None else None

    def collect_tracking_metrics(self) -> dict:
        """
        Collect all tracking metrics/signals and append one JSONL record.
        """
        poll_start = time.time()
        metrics_server_up, snapshot = self._fetch_metrics_snapshot()
        gauges = snapshot.get("gauges", {}) if isinstance(snapshot, dict) else {}
        if not isinstance(gauges, dict):
            gauges = {}

        loss = self._to_float(gauges.get("training_loss"))
        tps = self._to_float(gauges.get("tokens_per_second"))
        global_step = self._to_float(gauges.get("global_step"))

        nan_loss_signal = 1 if (loss is not None and not math.isfinite(loss)) else 0

        loss_rolling_avg = None
        loss_spike_ratio = None
        loss_spike_signal = 0
        if loss is not None and math.isfinite(loss):
            if len(self.loss_history) >= self.min_window_points:
                loss_rolling_avg = self._mean(self.loss_history)
                if loss_rolling_avg is not None and loss_rolling_avg > 0:
                    loss_spike_ratio = loss / loss_rolling_avg
                    loss_spike_signal = 1 if loss_spike_ratio > 2.0 else 0
            self.loss_history.append(loss)

        throughput_rolling_avg = None
        throughput_drop_ratio = None
        throughput_drop_signal = 0
        if tps is not None and math.isfinite(tps):
            if len(self.throughput_history) >= self.min_window_points:
                throughput_rolling_avg = self._mean(self.throughput_history)
                if throughput_rolling_avg is not None and throughput_rolling_avg > 0:
                    throughput_drop_ratio = tps / throughput_rolling_avg
                    throughput_drop_signal = 1 if throughput_drop_ratio < 0.8 else 0
            self.throughput_history.append(tps)

        gpu_mem_max_percent, gpu_device_count, gpu_metrics_source = (
            self._collect_gpu_mem_max_percent()
        )
        gpu_oom_signal = (
            1 if (gpu_mem_max_percent is not None and gpu_mem_max_percent >= 95.0) else 0
        )

        disk_min_free_percent = self._collect_disk_min_free_percent()
        disk_low_space_signal = (
            1 if (disk_min_free_percent is not None and disk_min_free_percent < 10.0) else 0
        )

        process_state = self._update_process_state()

        row = {
            "timestamp": time.time(),
            "timestamp_iso": self._utc_iso_now(),
            "metrics_server_up": metrics_server_up,
            "signal_metrics_server_down": 0 if metrics_server_up else 1,
            "global_step": global_step,
            "training_loss": loss,
            "tokens_per_second": tps,
            "loss_rolling_avg": loss_rolling_avg,
            "loss_spike_ratio": loss_spike_ratio,
            "throughput_rolling_avg": throughput_rolling_avg,
            "throughput_drop_ratio": throughput_drop_ratio,
            "gpu_mem_max_percent": gpu_mem_max_percent,
            "gpu_device_count": gpu_device_count,
            "gpu_metrics_source": gpu_metrics_source,
            "disk_min_free_percent": disk_min_free_percent,
            "signal_nan_loss": nan_loss_signal,
            "signal_loss_spike": loss_spike_signal,
            "signal_throughput_drop": throughput_drop_signal,
            "signal_gpu_oom_risk": gpu_oom_signal,
            "signal_disk_low_space": disk_low_space_signal,
            "poll_latency_ms": round((time.time() - poll_start) * 1000.0, 2),
        }
        row.update(process_state)

        self._append_tracking_row(row)
        return row

    def _dispatch_alerts(self, row: dict) -> list[dict]:
        self.last_alert_events = []
        if self._alerter is None:
            return self.last_alert_events
        try:
            events = self._alerter.evaluate(row)
            if isinstance(events, list):
                self.last_alert_events = events
        except Exception as e:
            print(f"⚠ Watchdog alerter evaluate failed: {e}")
        return self.last_alert_events

    def poll_once(self) -> dict:
        row = self.collect_tracking_metrics()
        self._dispatch_alerts(row)
        return row

    def check_alerts(self):
        """
        Backward-compatible name.
        Collect/tracks metrics/signals and dispatches alerts if alerter is enabled.
        """
        return self.poll_once()

    def trigger_pause(self, reason: str):
        """
        Write the Control Flag to pause training.
        """
        existing_payload = None
        if self.control_file.exists():
            try:
                with open(self.control_file, "r") as f:
                    existing_payload = json.load(f)
            except Exception:
                existing_payload = None

        self.pause_trigger_count += 1
        payload = {
            "action": "PAUSE",
            "reason": reason,
            "timestamp": time.time(),
            "timestamp_iso": self._utc_iso_now(),
            "trigger_count": self.pause_trigger_count,
            "retriggered": bool(existing_payload),
        }
        if isinstance(existing_payload, dict):
            payload["previous_reason"] = existing_payload.get("reason")
            payload["previous_timestamp"] = existing_payload.get("timestamp")

        with open(self.control_file, "w") as f:
            json.dump(payload, f)

        if payload["retriggered"]:
            print(f"⛔ PAUSE RE-TRIGGERED: {reason}")
            print(f"   Control flag updated at {self.control_file}")
        else:
            print(f"⛔ PAUSE TRIGGERED: {reason}")
            print(f"   Control flag written to {self.control_file}")

    def run(self):
        print("Watchdog Service Running...")
        try:
            while self.running:
                self.poll_once()
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            print("Watchdog Stopped.")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_optional_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return None


def _env_optional_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _main():
    watchdog = Watchdog(
        metrics_url=os.environ.get("WATCHDOG_METRICS_URL", "http://localhost:8000"),
        control_file_path=os.environ.get(
            "WATCHDOG_CONTROL_FILE_PATH",
            "/tmp/training_control.flag",
        ),
        poll_interval=_env_int("WATCHDOG_POLL_INTERVAL_S", 5),
        loss_threshold=_env_float("WATCHDOG_LOSS_THRESHOLD", 10.0),
        tracking_log_path=os.environ.get(
            "WATCHDOG_TRACKING_LOG_PATH",
            "/tmp/watchdog_metrics.jsonl",
        ),
        loss_window_size=_env_int("WATCHDOG_LOSS_WINDOW_SIZE", 60),
        throughput_window_size=_env_int("WATCHDOG_THROUGHPUT_WINDOW_SIZE", 60),
        min_window_points=_env_int("WATCHDOG_MIN_WINDOW_POINTS", 5),
        training_pid=_env_optional_int("WATCHDOG_TRAINING_PID"),
        training_pid_file_path=os.environ.get("WATCHDOG_TRAINING_PID_FILE"),
        request_timeout_s=_env_float("WATCHDOG_REQUEST_TIMEOUT_S", 2.0),
        pynvml_retry_interval_s=_env_float("WATCHDOG_PYNVML_RETRY_INTERVAL_S", 60.0),
        enable_alerter=_env_optional_bool("WATCHDOG_ENABLE_ALERTER"),
        telegram_bot_token=os.environ.get("WATCHDOG_TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=os.environ.get("WATCHDOG_TELEGRAM_CHAT_ID"),
        alert_resend_interval_s=_env_int("WATCHDOG_ALERT_RESEND_INTERVAL_S", 300),
        alert_log_path=os.environ.get(
            "WATCHDOG_ALERT_LOG_PATH",
            "/tmp/watchdog_alerts.jsonl",
        ),
    )
    watchdog.run()


if __name__ == "__main__":
    _main()
