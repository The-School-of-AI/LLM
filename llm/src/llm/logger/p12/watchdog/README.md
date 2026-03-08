# Watchdog + Alerter

This document explains what was added, how the watchdog/alerter now works, and how to enable Telegram alerts.

## What Changed

The watchdog stack now has 3 layers:

1. `watchdog.py`  
   Collects training/system reliability signals every poll cycle.
2. `alerter.py`  
   Evaluates per-signal state transitions and sends Telegram alerts.
3. Backend event bridge  
   Every fired alert is also written as a P12 event JSON record for Vector -> ClickHouse `events`.

In addition, `llm/run.sh` now starts/stops watchdog automatically during training.

## Signals Tracked

The watchdog emits these signal fields per row:

- `signal_nan_loss`
- `signal_loss_spike`
- `signal_throughput_drop`
- `signal_gpu_oom_risk`
- `signal_disk_low_space`
- `signal_training_process_crash`
- `signal_metrics_server_down`

The tracking rows are written to:

- `/tmp/watchdog_metrics.jsonl` (default)

## Alert Behavior

Per signal, the alerter does:

- `INACTIVE -> FIRING`: send `NEED_ATTENTION` immediately
- `FIRING -> FIRING`: resend every `resend_interval_s`
- `FIRING -> INACTIVE`: send `RESOLVED` immediately
- `INACTIVE -> INACTIVE`: no action

Other behavior:

- `None` signal value is always treated as inactive
- state is persisted to `/tmp/watchdog_alert_state.json` (default)
- Telegram failures do not crash watchdog
- alert records are written to `/tmp/watchdog_alerts.jsonl` (default)
- state load/save failures are tracked and logged as internal `ALERTER_STATE_ERROR`

## Backend Events Integration

Each fired alert is also appended to:

- `/tmp/training_logs/watchdog/watchdog_events.jsonl` (default)

Record shape is compatible with existing P12 Vector pipeline:

- top-level: `timestamp`, `run_id`, `host`, `rank`, `step`, `metrics`, `context`
- `context.event = "event"`
- `context.event_type = "watchdog_alert"`

This means alerts are shipped to ClickHouse `events` via existing Vector config.

## Automatic Startup from `run.sh`

`llm/run.sh` now:

- starts watchdog in background
- writes training launcher PID to a file for crash tracking
- stops watchdog when training exits
- fails fast if watchdog cannot start

Relevant runtime log:

- `<results>/run/watchdog.log`

## Prerequisites

1. Python dependencies available in your environment (`requests`, `psutil`, `nvidia-ml-py`, etc.).
2. For full metric coverage, set training observability backend to `p12` in config:

```yaml
observability:
  backend: p12
  p12:
    log_dir: /tmp/training_logs
    metrics_port: 8000
    system_metrics_interval: 5.0
    vector_service_name: p12-vector.service
    skip_vector_check: false
```

3. Vector sidecar must be running if you want backend shipping to ClickHouse.

## Telegram Setup

### 1) Create bot and get token

- In Telegram, message `@BotFather`
- Run `/newbot`
- Copy the bot token

### 2) Get chat ID

- Send a message to your bot (or target group/channel)
- Call:

```bash
curl -s "https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates"
```

- Find `chat.id` in the response

### 3) Export env vars before running

```bash
export WATCHDOG_TELEGRAM_BOT_TOKEN="<YOUR_TOKEN>"
export WATCHDOG_TELEGRAM_CHAT_ID="<YOUR_CHAT_ID>"
```

Optional:

```bash
export WATCHDOG_ENABLE_ALERTER=1
export WATCHDOG_ALERT_RESEND_INTERVAL_S=300
```

Notes:

- resend interval has a minimum of `60` seconds
- Telegram timeout uses connect/read split (default `1.0s` / `2.5s`)

## Running

From project root:

```bash
cd llm
./run.sh
```

Or run watchdog manually:

```bash
cd llm
uv run python -m llm.logger.p12.watchdog.watchdog
```

## Watchdog Environment Variables

Core:

- `WATCHDOG_ENABLED` (run.sh toggle, default `1`)
- `WATCHDOG_METRICS_URL` (default `http://localhost:8000`)
- `WATCHDOG_POLL_INTERVAL_S` (default `5`)
- `WATCHDOG_TRACKING_LOG_PATH` (default `/tmp/watchdog_metrics.jsonl`)
- `WATCHDOG_CONTROL_FILE_PATH` (default `/tmp/training_control.flag`)
- `WATCHDOG_TRAINING_PID_FILE` (set by run.sh)

Alerter:

- `WATCHDOG_ENABLE_ALERTER` (default `1` in run.sh)
- `WATCHDOG_TELEGRAM_BOT_TOKEN`
- `WATCHDOG_TELEGRAM_CHAT_ID`
- `WATCHDOG_ALERT_RESEND_INTERVAL_S` (min `60`)
- `WATCHDOG_ALERT_LOG_PATH` (default `/tmp/watchdog_alerts.jsonl`)
- `WATCHDOG_RUN_ID` (optional override for backend event run ID)

Disk/GPU tuning:

- `WATCHDOG_DISK_PATHS` (comma-separated disk paths to monitor)
- `WATCHDOG_REQUEST_TIMEOUT_S`
- `WATCHDOG_PYNVML_RETRY_INTERVAL_S`

## Troubleshooting

1. Watchdog starts but `signal_metrics_server_down=1`
- Ensure observability backend is `p12`
- Ensure metrics server is running on `metrics_port` (default `8000`)

2. No Telegram alerts
- Check token/chat id env vars
- Check `/tmp/watchdog_alerts.jsonl` for `telegram_error`

3. Alerts not reaching backend events
- Check `/tmp/training_logs/watchdog/watchdog_events.jsonl`
- Verify Vector source includes `/tmp/training_logs/**/*.jsonl`

4. GPU metrics unavailable
- If NVML import fails, watchdog falls back to `nvidia-smi`
- Ensure `nvidia-smi` is available on host

