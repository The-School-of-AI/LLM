#!/usr/bin/env bash
# =============================================================================
# P12 ClickHouse Health Check — pushes custom metrics to CloudWatch
# Runs via cron every minute on the DB instance.
#
# Metrics published to namespace P12/ClickHouse:
#   Alive              — 1 if ClickHouse responds to SELECT 1, else 0
#   UptimeSeconds      — server uptime in seconds
#   DiskUsedPercent    — /data/clickhouse EBS volume usage percentage
#   LogsRowCount       — total rows in the logs table
#   MetricPointsRowCount — total rows in the metric_points table
#   LastInsertAgeSeconds — seconds since the most recent insert into logs
# =============================================================================

set -uo pipefail

REGION="${AWS_DEFAULT_REGION:-us-east-1}"
NAMESPACE="P12/ClickHouse"
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")

push_metric() {
  local name="$1" value="$2" unit="${3:-None}"
  aws cloudwatch put-metric-data \
    --namespace "$NAMESPACE" \
    --metric-name "$name" \
    --value "$value" \
    --unit "$unit" \
    --dimensions "InstanceId=$INSTANCE_ID" \
    --region "$REGION" 2>/dev/null
}

# ---- 1. Alive check ----
if sudo docker exec p12-clickhouse clickhouse-client --query "SELECT 1" &>/dev/null; then
  push_metric "Alive" 1
else
  push_metric "Alive" 0
  echo "CRITICAL: ClickHouse not responding"
  exit 1
fi

# ---- 2. Uptime ----
UPTIME=$(sudo docker exec p12-clickhouse clickhouse-client \
  --query "SELECT uptime()" 2>/dev/null || echo "0")
push_metric "UptimeSeconds" "$UPTIME" "Seconds"

# ---- 3. Disk usage (EBS volume) ----
DISK_PERCENT=$(df /data/clickhouse | tail -1 | awk '{print $5}' | tr -d '%')
push_metric "DiskUsedPercent" "$DISK_PERCENT" "Percent"

# ---- 4. Row counts ----
LOGS_COUNT=$(sudo docker exec p12-clickhouse clickhouse-client \
  --query "SELECT count() FROM training_observability.logs" 2>/dev/null || echo "0")
push_metric "LogsRowCount" "$LOGS_COUNT" "Count"

METRIC_COUNT=$(sudo docker exec p12-clickhouse clickhouse-client \
  --query "SELECT count() FROM training_observability.metric_points" 2>/dev/null || echo "0")
push_metric "MetricPointsRowCount" "$METRIC_COUNT" "Count"

# ---- 5. Ingestion freshness ----
LAST_INSERT_AGE=$(sudo docker exec p12-clickhouse clickhouse-client \
  --query "SELECT dateDiff('second', max(event_time), now64(3)) FROM training_observability.logs" \
  2>/dev/null || echo "99999")

# If table is empty, report 0 (no alarm — training hasn't started yet)
if [ "$LOGS_COUNT" -eq 0 ]; then
  LAST_INSERT_AGE=0
fi
push_metric "LastInsertAgeSeconds" "$LAST_INSERT_AGE" "Seconds"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) OK: alive=1 uptime=${UPTIME}s disk=${DISK_PERCENT}% logs=${LOGS_COUNT} metrics=${METRIC_COUNT} freshness=${LAST_INSERT_AGE}s"
