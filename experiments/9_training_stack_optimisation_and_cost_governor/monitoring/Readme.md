# Training Monitoring Dashboard

**A universal dashboard that visualizes metrics from ANY machine learning training framework**

---

---

## What Is This?

A web-based dashboard that:
- ✅ **Automatically discovers** all metrics in your training logs
- ✅ **Works with ANY framework** (PyTorch, TensorFlow, JAX, custom code)
- ✅ **No configuration needed** - just point it at your logs
- ✅ **Team-friendly** - anyone can view anyone's training runs

### Key Features

- **Auto-Discovery**: Scans your logs and finds all metrics automatically (no need to tell it what to look for)
- **Dynamic Charts**: You choose which metrics to visualize (not hardcoded)
- **Framework Agnostic**: Works with logs from any training code
- **Zero Setup**: No configuration files, no metric definitions required

---

## How It Works


## 🏗️ Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    ANY TRAINING FRAMEWORK                        │
│         (PyTorch, TF, JAX, Custom, etc.)                        │
│                                                                   │
│  Team writes logs in JSONL format:                              │
│  {"step": 100, "metric": "train/loss", "value": 2.5}           │
│  {"step": 100, "metric": "custom_score", "value": 0.87}        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ Upload/Store
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STORAGE LAYER                                 │
│                                                                   │
│  Option 1: Local filesystem (logs/)                             │
│  Option 2: S3 bucket (s3://bucket/logs/)                        │
│  Option 3: Network share (NFS, etc.)                            │
│                                                                   │
│  Structure: logs/run_id/metrics/data.jsonl                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ Read on demand
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              FLEXIBLE DASHBOARD BACKEND                          │
│                (dashboard_server.py)                             │
│                                                                   │
│  1. Scan all log files                                          │
│  2. Extract ALL metric names (auto-discovery)                   │
│  3. Group metrics by category                                   │
│  4. Serve data via REST API                                     │
│                                                                   │
│  API Endpoints:                                                  │
│  • GET /api/runs              → List all runs                   │
│  • GET /api/runs/{id}/metrics → List all metrics in run        │
│  • GET /api/runs/{id}/data    → Get data for selected metrics  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ HTTP/JSON
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              FLEXIBLE DASHBOARD UI                               │
│                (dashboard/index.html)                            │
│                                                                   │
│  ┌─────────────────────────────────────────────────┐           │
│  │  Step 1: Select Run                             │           │
│  │  Dropdown: [run_20260204_*, run_20260203_*, …] │           │
│  └─────────────────────────────────────────────────┘           │
│                      ↓                                           │
│  ┌─────────────────────────────────────────────────┐           │
│  │  Step 2: Discovered Metrics                     │           │
│  │                                                  │           │
│  │  Training (12 metrics)                          │           │
│  │  ☑ train/loss                                   │           │
│  │  ☑ train/accuracy                               │           │
│  │  ☐ train/learning_rate                          │           │
│  │                                                  │           │
│  │  GPU (5 metrics)                                │           │
│  │  ☑ gpu/utilization                              │           │
│  │  ☐ gpu/temperature                              │           │
│  │                                                  │           │
│  │  Custom (8 metrics)                             │           │
│  │  ☑ my_custom_metric                             │           │
│  │  ☐ another_score                                │           │
│  └─────────────────────────────────────────────────┘           │
│                      ↓                                           │
│  ┌─────────────────────────────────────────────────┐           │
│  │  Step 3: Auto-Generated Charts                  │           │
│  │                                                  │           │
│  │  [Chart: train/loss]     [Chart: train/acc]    │           │
│  │  [Chart: gpu/util]       [Chart: custom_met]   │           │
│  └─────────────────────────────────────────────────┘           │
│                                                                   │
│  User can:                                                       │
│  • Select/deselect metrics                                      │
│  • Rearrange charts (drag & drop)                              │
│  • Export data                                                  │
│  • Compare multiple runs                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

**1. Log Writer (User's Side)**
- ANY code that writes JSONL files
- Must follow minimal format (see below)
- Can be Python, C++, Rust, whatever

**2. Storage Layer**
- Just files in JSONL format
- Can be local, S3, NFS, anywhere
- Dashboard reads, never writes

**3. Backend Server**
- Python Flask app
- Scans logs, extracts metric names
- Serves data via REST API
- No database needed

**4. Frontend UI**
- Single HTML file + JavaScript
- Dynamic chart generation
- Uses Chart.js for visualization
- Responsive, mobile-friendly

### The Simple Version

```
1. You train your model → Logs get saved as JSON files
2. You start the dashboard → It scans those logs
3. You open browser → Pick what you want to see
4. Dashboard shows charts → Updates automatically
```

### The Technical Version

```
Training Code (any framework)
    ↓ writes logs in JSONL format
Log Files (logs/run_name/metrics/data.jsonl)
    ↓ dashboard reads
Dashboard Server (Python Flask)
    ↓ discovers metrics automatically
    ↓ serves data via REST API
Web Browser
    ↓ user selects metrics
    ↓ generates charts dynamically
```

---

## Quick Start Guide

### Prerequisites

- Python 3.8 or higher
- Training logs in JSONL format (see [Log Format](#log-format-requirements))

### Installation

**Step 1: Navigate to monitoring folder**
```bash
cd monitoring
```

**Step 2: Create virtual environment** (recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

Expected output:
```
Successfully installed flask-2.3.0 flask-cors-4.0.0 ...
```

### Running the Dashboard

**Basic usage** (logs in default location):
```bash
python dashboard_server.py
```

**With custom log location**:
```bash
python dashboard_server.py --log-dir /path/to/your/logs
```

**With custom port** (if port 5000 is busy):
```bash
python dashboard_server.py --port 8080
```

**Example**:
```bash
python dashboard_server.py --log-dir ../training/deepspeed_template/logs --port 8080
```

### Accessing the Dashboard

1. Open your web browser
2. Go to: `http://localhost:5000` (or whatever port you specified)
3. You should see the dashboard interface

---

## For Users: Viewing Dashboards

### Step 1: Select a Training Run

![Select Run](docs/images/select-run.png) _(if image available)_

- Open the dashboard in your browser
- Click the "Select Run" dropdown
- Choose the training run you want to view

### Step 2: Discover Metrics

The dashboard will automatically scan the logs and show you ALL available metrics grouped by category:

**Example:**
```
Training (12 metrics)
  ☐ train/loss
  ☐ train/accuracy
  ☐ train/learning_rate

GPU (5 metrics)
  ☐ gpu/utilization
  ☐ gpu/temperature

Custom (8 metrics)
  ☐ my_custom_metric
  ☐ another_score
```

### Step 3: Select Metrics to Visualize

- Check the boxes next to metrics you want to see
- You can select as many or as few as you want
- Different runs may have different metrics (that's OK!)

### Step 4: Generate Charts

- Click the "Generate Charts" button
- Charts will appear below
- Each metric gets its own chart
- Charts update automatically if training is still running

### Tips

- **Compare metrics**: Select multiple related metrics to see patterns
- **Refresh data**: Click the refresh button to get latest values
- **Switch runs**: Use dropdown to compare different experiments
- **No limit**: Select as many metrics as you want

---

## For Teams: Sharing Your Logs

### Option 1: Shared Network Drive

**Best for**: Teams working on same network

1. Save your logs to a shared location (e.g., NFS, shared drive)
2. Everyone points dashboard to that location:
   ```bash
   python dashboard_server.py --log-dir /shared/team/logs
   ```
3. Everyone can view everyone's experiments

### Option 2: Cloud Storage (S3, GCS)

**Best for**: Distributed teams

1. Upload logs to cloud storage after training:
   ```bash
   aws s3 cp logs/my_run/ s3://team-bucket/logs/my_run/ --recursive
   ```

2. Team members download and view:
   ```bash
   aws s3 cp s3://team-bucket/logs/ logs/ --recursive
   python dashboard_server.py --log-dir logs/
   ```

### Option 3: Direct File Sharing

**Best for**: Quick one-off sharing

1. Zip your logs:
   ```bash
   tar -czf my_training_logs.tar.gz logs/run_20260206_140522/
   ```

2. Share the file (email, Slack, etc.)

3. Recipient extracts and views:
   ```bash
   tar -xzf my_training_logs.tar.gz
   python dashboard_server.py --log-dir logs/
   ```

---

## Log Format Requirements

### File Structure

Your logs must follow this structure:

```
logs/
└── run_id/                    # Any name (e.g., run_20260206_140522)
    └── metrics/
        └── data.jsonl         # JSON Lines format
```

### JSONL Format

Each line in `data.jsonl` must be a valid JSON object with these fields:

**Required Fields:**
```json
{"step": 100, "metric": "train/loss", "value": 2.5}
```

**Optional Fields:**
```json
{
  "step": 100,
  "metric": "train/loss",
  "value": 2.5,
  "timestamp": 1675434622.123,
  "type": "scalar",
  "tags": {"gpu": 0, "batch_size": 32}
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `step` | integer | ✅ Yes | Training step/iteration number |
| `metric` | string | ✅ Yes | Metric name (can be anything) |
| `value` | number or array | ✅ Yes | Metric value |
| `timestamp` | float | ❌ No | Unix timestamp |
| `type` | string | ❌ No | "scalar", "histogram", or "text" |
| `tags` | object | ❌ No | Additional metadata |

### Examples

**Scalar metric** (most common):
```json
{"step": 100, "metric": "train/loss", "value": 2.5}
```

**Array metric** (e.g., expert counts):
```json
{"step": 100, "metric": "moe/expert_counts", "value": [10, 12, 8, 15, 9, 11, 14, 10]}
```

**With metadata**:
```json
{"step": 100, "metric": "gpu/temp", "value": 65, "timestamp": 1675434622.123, "tags": {"gpu_id": 0}}
```

### Metric Naming Convention

**Recommended format**: `category/metric_name`

**Examples:**
- `train/loss`
- `train/accuracy`
- `gpu/temperature`
- `gpu/utilization`
- `moe/expert_balance`
- `custom/my_metric`

**Why use this format?**
- Metrics get grouped automatically in the UI
- Easier to find related metrics
- More organized display

**But you can use any names!**
- `loss` ✅
- `training_loss` ✅
- `my_super_custom_metric_v2` ✅

---

## Troubleshooting

### Problem: "No runs found"

**Cause**: Dashboard can't find log files

**Solutions:**
1. Check log directory exists:
   ```bash
   ls -la logs/
   ```

2. Verify directory structure:
   ```bash
   logs/
   └── run_name/
       └── metrics/
           └── data.jsonl
   ```

3. Check path in command:
   ```bash
   python dashboard_server.py --log-dir /correct/path/to/logs
   ```

### Problem: "No metrics found in this run"

**Cause**: JSONL file is empty or invalid

**Solutions:**
1. Check file exists:
   ```bash
   ls -la logs/run_name/metrics/
   ```

2. Check file has content:
   ```bash
   head logs/run_name/metrics/data.jsonl
   ```

3. Verify JSON format is valid:
   ```bash
   # Each line should be valid JSON
   head -n 1 logs/run_name/metrics/data.jsonl | python -m json.tool
   ```

### Problem: "Port 5000 is in use"

**Cause**: Another service using port 5000 (often AirPlay on macOS)

**Solution**: Use different port
```bash
python dashboard_server.py --port 8080
```

Then access at: `http://localhost:8080`

### Problem: Dashboard shows old data

**Cause**: Browser cache or server cache

**Solutions:**
1. Click the "Refresh" button in dashboard
2. Hard refresh browser: `Ctrl+Shift+R` (or `Cmd+Shift+R` on Mac)
3. Restart dashboard server

### Problem: Charts not appearing

**Cause**: No data for selected metrics

**Solutions:**
1. Verify metrics exist in logs:
   ```bash
   grep "train/loss" logs/run_name/metrics/data.jsonl | head
   ```

2. Check browser console for errors (F12)

3. Try selecting different metrics

### Problem: "Module not found" errors

**Cause**: Dependencies not installed

**Solution**:
```bash
pip install -r requirements.txt
```

---

## Advanced Usage

### Custom Port

```bash
python dashboard_server.py --port 8080
```

### Debug Mode

```bash
python dashboard_server.py --debug
```

Shows detailed error messages (useful for troubleshooting).

### Multiple Dashboards

You can run multiple dashboard instances for different log directories:

**Terminal 1:**
```bash
python dashboard_server.py --log-dir /project_a/logs --port 5000
```

**Terminal 2:**
```bash
python dashboard_server.py --log-dir /project_b/logs --port 5001
```

### Remote Access

**Scenario**: Dashboard running on remote server, you want to view locally

**Solution**: SSH port forwarding

```bash
ssh -L 5000:localhost:5000 user@remote-server
```

Then access `http://localhost:5000` on your local machine.

### Automated Testing

Test your setup before production use:

```bash
python test_system.py
```

Should show all tests passing.

---

## FAQ

### Q: Do I need to change my training code?

**A:** Only if you're not already writing JSONL logs. If you are, no changes needed!

### Q: What if my metrics have different names than the examples?

**A:** That's fine! Dashboard discovers metrics automatically. Use any names you want.

### Q: Can I use this with TensorFlow/JAX/custom framework?

**A:** Yes! As long as you write logs in JSONL format, any framework works.

### Q: How much disk space do logs use?

**A:** Approximately 200 bytes per metric per step. For 10,000 steps and 50 metrics, that's about 100MB.

### Q: Can multiple people view the same dashboard?

**A:** Yes! Multiple browsers can connect to the same dashboard server.

### Q: What if I have 100+ metrics?

**A:** No problem! Dashboard handles it. You just select which ones to view.

### Q: Can I compare runs from different frameworks?

**A:** Yes, as long as they log the same metric names (e.g., both log "loss").

### Q: Does training slow down while logging?

**A:** Minimal impact (<1%). Logging is buffered and written asynchronously.

### Q: Can I export data from the dashboard?

**A:** Data is already in JSONL format, which is easy to read with any tool (Python, pandas, Excel).

### Q: What if my training crashes - do I lose logs?

**A:** Logs are written incrementally, so you keep everything up to the crash point.

### Q: Can I delete old runs?

**A:** Yes, just delete the folder: `rm -rf logs/old_run_name/`

---

## Support

### Getting Help

1. **Check this README** - Most questions answered here
2. **Run test script** - `python test_system.py` to verify setup
3. **Check browser console** - F12 → Console tab for errors
4. **Contact team** - Reach out to ML infrastructure team

### Reporting Issues

When reporting problems, include:
- Error message (full text)
- Command you ran
- Output of `python test_system.py`
- Sample of your log file (first 10 lines)

### Contributing

Found a bug? Have a feature request? Open an issue or submit a PR.

---

## Summary

### For Quick Reference

**Start dashboard:**
```bash
python dashboard_server.py --log-dir /path/to/logs --port 8080
```

**Log format:**
```json
{"step": 100, "metric": "train/loss", "value": 2.5}
```

**File structure:**
```
logs/run_name/metrics/data.jsonl
```

**Access:**
```
http://localhost:8080
```

---
