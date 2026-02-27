# Monitoring Dashboard — Troubleshooting Guide

## Architecture Overview

```
Browser → nginx (:80) → uvicorn/FastAPI (:8000) → ClickHouse (<IP ADD>:<PORT>)
                ↓
        frontend/dist/   (static files served by nginx)
```

| Component  | Process     | Port | Path                                  |
|------------|-------------|------|---------------------------------------|
| Frontend   | nginx       | 80   | `frontend/dist/` (static build)       |
| Backend    | uvicorn     | 8000 | `backend/app.py`                      |
| Database   | ClickHouse  | Port | Remote — IP Address.                  |

---

## 1. Check Status of Everything

```bash
# All at once — processes, ports, nginx
ps aux | grep -E 'uvicorn|nginx' | grep -v grep
ss -tlnp | grep -E '80|8000'

# Quick health check
curl -s http://localhost:8000/health        # backend direct
curl -s http://localhost:80/health          # backend through nginx
curl -s http://localhost:8000/runs | python3 -m json.tool | head -20

# Backend logs (live)
tail -f /tmp/uvicorn.log

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

---

## 2. Restart the Backend

```bash
# Step 1 — kill existing uvicorn (find PID first)
pgrep -a uvicorn
kill $(pgrep -f uvicorn)

# Step 2 — wait for port to free
sleep 2 && ss -tlnp | grep 8000   # should be empty

# Step 3 — start fresh (background, logs to file)
cd /home/ubuntu/monitoring-observability/backend
nohup .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 > /tmp/uvicorn.log 2>&1 &
echo "Backend PID: $!"

# Step 4 — verify it's up
sleep 3 && curl -s http://localhost:8000/health
```

---

## 3. Restart the Frontend (rebuild + nginx)

The frontend is a static build served by nginx. Restart needed only after source changes.

```bash
# Rebuild frontend
cd /home/ubuntu/monitoring-observability/frontend
npm run build

# Reload nginx to pick up any config changes (no downtime)
sudo nginx -t && sudo nginx -s reload

# Or full restart
sudo systemctl restart nginx
sudo systemctl status nginx
```

---

## 4. Restart nginx Only

```bash
sudo nginx -t                    # test config first
sudo nginx -s reload             # graceful reload (zero downtime)
sudo systemctl restart nginx     # full restart
sudo systemctl status nginx
```

---

## 5. Monitor Process Health

### Watch processes live
```bash
watch -n 2 'ps aux | grep -E "uvicorn|nginx" | grep -v grep'
```

### Watch backend logs live
```bash
tail -f /tmp/uvicorn.log
```

### Watch all ports
```bash
watch -n 2 'ss -tlnp | grep -E "80|8000"'
```

### Check if backend is responding
```bash
# One-liner health loop — alerts if backend goes down
while true; do
  if ! curl -sf --max-time 3 http://localhost:8000/health > /dev/null; then
    echo "$(date) — BACKEND DOWN"
  else
    echo "$(date) — ok"
  fi
  sleep 10
done
```

---

## 6. Diagnose a Stuck Backend

Symptom: `curl http://localhost:8000/health` hangs forever (port is listening but no HTTP response).

```bash
# Confirm port is open but not responding
ss -tlnp | grep 8000                        # should show uvicorn listening
curl -v --max-time 5 http://localhost:8000/health   # hangs = event loop stuck

# Check process threads and state
pgrep -a uvicorn
cat /proc/$(pgrep -f 'uvicorn app')/status | grep -E 'State|Threads'

# Check what threads are waiting on
ls /proc/$(pgrep -f 'uvicorn app')/task/ | while read tid; do
  echo "TID $tid: $(cat /proc/$(pgrep -f 'uvicorn app')/task/$tid/wchan 2>/dev/null)"
done

# Check open network connections from the backend process
ss -tnp | grep $(pgrep -f 'uvicorn app')
```

**Fix:** Restart the backend (see Section 2).

---

## 7. Force Kill a Stuck Backend

```bash
# Graceful kill first
kill $(pgrep -f uvicorn)
sleep 2

# If still running, force kill
kill -9 $(pgrep -f uvicorn)

# If port 8000 still in use after kill
fuser -k 8000/tcp
sleep 1
ss -tlnp | grep 8000   # confirm empty
```

---

## 8. Check ClickHouse Connectivity

```bash
# Simple reachability test
curl -k --max-time 5 https://<IP ADDRESS>:<PORT>

# Full Python connection test
cd /home/ubuntu/monitoring-observability/backend
.venv/bin/python -c "
from db import get_client
c = get_client()
print('Connected:', c.query('SELECT 1').result_rows)
"

# Query run count
.venv/bin/python -c "
from db import get_client
c = get_client()
r = c.query('SELECT count() FROM metric_points')
print('Total metric points:', r.result_rows[0][0])
"
```

---

## 9. Common Issues & Fixes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `/stream` returns 499 in nginx logs | Backend not responding | Restart backend (Section 2) |
| Backend curl hangs (timeout) | Event loop frozen | Force kill + restart (Sections 6–7) |
| `address already in use` on startup | Old process still alive | `fuser -k 8000/tcp`, then restart |
| Frontend shows blank / old data | Stale build | `npm run build` in `frontend/`, reload nginx |
| nginx 403 on frontend files | Permissions on `dist/` | `chmod -R 755 frontend/dist/` |
| nginx 502 Bad Gateway | Backend is down | Check backend, restart if needed |
| ClickHouse queries hang | Network / TLS issue | Test connectivity (Section 8) |

---

## 10. One-Shot Full Restart

Restarts everything cleanly in order:

```bash
# 1. Kill backend
kill -9 $(pgrep -f uvicorn) 2>/dev/null; sleep 2

# 2. Start backend
cd /home/ubuntu/monitoring-observability/backend
nohup .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 > /tmp/uvicorn.log 2>&1 &
echo "Backend PID: $!"

# 3. Reload nginx
sudo nginx -s reload

# 4. Verify
sleep 3
curl -s http://localhost:8000/health
curl -s http://localhost:80/health
echo "Backend PID: $(pgrep -f uvicorn)"
echo "Nginx PIDs:  $(pgrep nginx | tr '\n' ' ')"
```
