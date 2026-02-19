"""
gunicorn.conf.py
Production config for the Training Dashboard Server.
"""
import multiprocessing

# ── Binding ────────────────────────────────────────────────────────────────────
bind = "127.0.0.1:5050"  # localhost only — nginx is the public-facing entry point

# ── Workers ────────────────────────────────────────────────────────────────────
# gthread: each worker runs N threads — plays well with our thread-local
#          ClickHouse clients (one connection per thread, reused across requests)
worker_class = "gthread"
workers      = multiprocessing.cpu_count() * 2 + 1
threads      = 4          # concurrent requests per worker

# ── Timeouts ───────────────────────────────────────────────────────────────────
timeout      = 120        # kill worker if silent for 2 min (stuck query guard)
keepalive    = 5          # seconds to hold idle connections open

# ── Logging ────────────────────────────────────────────────────────────────────
accesslog = "-"           # stdout → captured by journald / supervisor
errorlog  = "-"
loglevel  = "info"

# ── Stability ──────────────────────────────────────────────────────────────────
max_requests      = 1000  # recycle workers after N requests (guards memory leaks)
max_requests_jitter = 100 # randomise so all workers don't restart simultaneously
