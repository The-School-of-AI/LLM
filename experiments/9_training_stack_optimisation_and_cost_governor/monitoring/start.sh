#!/usr/bin/env bash
# start.sh — run gunicorn directly (useful for testing outside systemd)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec .venv/bin/gunicorn -c gunicorn.conf.py dashboard_server:app
