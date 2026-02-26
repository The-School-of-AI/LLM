#!/usr/bin/env bash
set -euo pipefail

CLICKHOUSE_HOST="${CLICKHOUSE_HOST:-127.0.0.1}"
CLICKHOUSE_PORT="${CLICKHOUSE_PORT:-9000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

clickhouse-client \
  --host "${CLICKHOUSE_HOST}" \
  --port "${CLICKHOUSE_PORT}" \
  --multiquery \
  --queries-file "${SCRIPT_DIR}/initdb.d/001_database.sql"

clickhouse-client \
  --host "${CLICKHOUSE_HOST}" \
  --port "${CLICKHOUSE_PORT}" \
  --multiquery \
  --queries-file "${SCRIPT_DIR}/initdb.d/002_logs_table.sql"

clickhouse-client \
  --host "${CLICKHOUSE_HOST}" \
  --port "${CLICKHOUSE_PORT}" \
  --multiquery \
  --queries-file "${SCRIPT_DIR}/initdb.d/003_typed_tables.sql"
