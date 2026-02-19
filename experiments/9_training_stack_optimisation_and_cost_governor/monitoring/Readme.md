# train/watch — Training Observability Dashboard

Real-time monitoring dashboard for ML training runs, backed by ClickHouse. Multi-run overlay, zoomable charts, histogram views, event log, and a summary tab with heatmaps and sparklines.

---

## Project Structure

```
monitoring/
├── dashboard_server.py       # Flask backend — serves API + static files
├── requirements.txt          # Python dependencies
└── dashboard/
    ├── index.html            # App shell (pure markup, no inline JS/CSS)
    ├── style.css             # All styles (responsive: desktop + mobile)
    └── js/
        ├── app.js            # Entry point — init, refresh, event wiring
        ├── state.js          # Shared mutable state
        ├── constants.js      # Colors, icons, Chart.js defaults
        ├── utils.js          # Pure helpers: fmt, stats, heatColor, showToast
        ├── api.js            # All fetch calls to the server
        ├── runs.js           # Run dropdown, chips, selection logic
        ├── metrics.js        # Metric sidebar, search, category select
        ├── charts.js         # Chart creation, zoom/pan, toolbar
        ├── events.js         # Events tab
        ├── summary.js        # Summary tab: KPIs, pies, stats table, heatmap
        └── tabs.js           # Tab switching
```

---

## For Teammates / New Contributors

```bash
# 1. Clone
git clone <repo-url>
cd monitoring

# 2. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Get credentials from a teammate (never shared via git)
cp .env.example .env
# Fill in .env with the real values

# 4. Run (local dev)
./start.sh
```

Then open `http://localhost:5050`.

---

## Local Setup

### 1. Clone / navigate to the directory

```bash
cd monitoring
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set credentials via environment variables

Do not hardcode credentials. Export these before running:

```bash
export CH_HOST=<your-clickhouse-host>
export CH_PORT=8443
export CH_USER=<your-user>
export CH_PASSWORD=<your-password>
export CH_DB=training_observability
```

Or create a `.env` file and load it (never commit the `.env` file):

```bash
# .env
CH_HOST=54.174.194.76
CH_PORT=8443
CH_USER=p12_reader
CH_PASSWORD=your_password_here
CH_DB=training_observability
```

```bash
export $(cat .env | xargs)
```

### 5. Start the server

```bash
./start.sh
```

Expected output:

```
📋 Discovered 5 tables:
   events                         → events
   metric_arrays                  → array
   metric_points                  → scalar
   runs                           → runs
   checkpoints                    → generic

🚀 ClickHouse Dashboard Server Running
   Host:     <host>:8443
   Database: training_observability
   Tables:   5 active

 * Running on http://0.0.0.0:5050
```

### 6. Open the dashboard

```
http://localhost:5050
```

---

## Deploying on EC2 (Production)

Runs as a systemd service with gunicorn — survives crashes, reboots, and handles concurrent traffic automatically.

### 1. Launch the instance

- AMI: **Ubuntu 24.04 LTS ARM64** — t4g instances are Graviton2 (ARM), so pick the ARM64 image in the AMI selector, not the default x86
- Type: **t4g.small** (2 vCPU, 2 GB RAM)
- Security Group inbound rules:

  | Type | Port | Source |
  |------|------|--------|
  | SSH | 22 | Your IP |
  | HTTP | 80 | 0.0.0.0/0 |

  > Port 5050 does NOT need to be open — gunicorn only listens on localhost, Nginx is the public entry point.

### 2. SSH in

```bash
ssh -i your-key.pem ubuntu@<ec2-public-ip>
```

### 3. Install system dependencies

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git
```

### 4. Clone the repo

```bash
git clone <your-repo-url> /home/ubuntu/monitoring
cd /home/ubuntu/monitoring
```

### 5. Set up Python environment

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 6. Configure credentials

```bash
cp .env.example .env
nano .env   # fill in CH_HOST, CH_USER, CH_PASSWORD, etc.
```

### 7. Add swap (safety net for memory spikes)

```bash
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 8. Install and configure Nginx

```bash
sudo apt install -y nginx

# Copy the site config
sudo cp monitoring_nginx.conf /etc/nginx/sites-available/monitoring

# Enable it and disable the default site
sudo ln -s /etc/nginx/sites-available/monitoring /etc/nginx/sites-enabled/monitoring
sudo rm -f /etc/nginx/sites-enabled/default

# Test config and reload
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl start nginx
```

### 9. Install and start the systemd service

```bash
# Copy the service file
sudo cp monitoring.service /etc/systemd/system/

# Reload systemd and enable the service (starts on every boot)
sudo systemctl daemon-reload
sudo systemctl enable monitoring

# Start it now
sudo systemctl start monitoring

# Confirm it's running
sudo systemctl status monitoring
```

Expected output:
```
● monitoring.service - Training Dashboard Server
     Loaded: loaded (/etc/systemd/system/monitoring.service; enabled)
     Active: active (running) since ...
```

### 10. Access the dashboard

```
http://<ec2-public-ip>
```

No port needed — Nginx listens on port 80 (standard HTTP).

---

### Everyday commands

```bash
# Live logs
journalctl -u monitoring -f

# Restart (e.g. after pulling new code)
sudo systemctl restart monitoring

# Stop
sudo systemctl stop monitoring

# Check status
sudo systemctl status monitoring
```

### Deploying updates

```bash
cd /home/ubuntu/monitoring
git pull
sudo systemctl restart monitoring
```

---

## Using the Dashboard

| Step | Action |
|------|--------|
| 1 | Open the **Training Runs** dropdown in the sidebar and select one or more runs |
| 2 | Check metrics in the **Metrics** section to add them to the chart view |
| 3 | Switch between **Charts / Summary / Events** tabs |
| 4 | Click the `‹` button to collapse the sidebar for more chart space |
| 5 | Set **auto-refresh** interval in the header (default: 30s) |

### Chart toolbar

| Button | Action |
|--------|--------|
| Magnifier | Box-zoom — drag to zoom into a region |
| Home | Reset zoom to full range |
| Expand | Toggle full-width expanded view |
| Download | Save chart as PNG |

---

## API Reference

| Endpoint | Description |
|----------|-------------|
| `GET /api/runs` | List all training runs |
| `GET /api/runs/<run_id>/metrics` | Metrics for a run, grouped by category |
| `GET /api/runs/<run_id>/data?metrics=a,b` | Time-series data for requested metrics |
| `GET /api/runs/<run_id>/events` | Training event log |
| `GET /api/runs/<run_id>/current` | Latest value per metric |
| `GET /api/tables` | All discovered tables and their roles |
| `GET /api/tables/refresh` | Re-discover tables without restarting the server |
| `GET /api/health` | ClickHouse connectivity check |

---

## How Table Discovery Works

At startup the server scans `system.columns` in ClickHouse and assigns each table a role automatically — no config needed.

| Role | Criteria | Used For |
|------|----------|----------|
| `runs` | Has `run_id`, no `step`/`metric` | Run metadata list |
| `scalar` | Has `run_id` + `step` + `metric` + float column | Line charts |
| `array` | Has `run_id` + `step` + `metric` + `Array(Float)` | Histogram charts |
| `events` | Has `run_id` + `step` + `event_type`/`message` | Event log tab |
| `generic` | Has `run_id` + `step`, unknown structure | Surfaced as metrics |
| `ignored` | No `run_id` column | Skipped entirely |

After adding a new table to ClickHouse, no server restart needed — just hit:

```bash
curl http://localhost:5050/api/tables/refresh
```

---

## Troubleshooting

**Port 5050 already in use**
```bash
lsof -i :5050 | awk 'NR>1 {print $2}' | xargs kill
python3 dashboard_server.py
```

**ClickHouse connection error**
```bash
curl -k "https://<host>:8443/?query=SELECT+1"
```

**No metrics showing for a run**
```bash
curl http://localhost:5050/api/runs/<run_id>/metrics | python3 -m json.tool
```

**New ClickHouse table not appearing**
```bash
curl http://localhost:5050/api/tables/refresh
```

**Health check**
```bash
curl http://localhost:5050/api/health
```
