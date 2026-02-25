# INFRA_OBSERVABILITY – Runtime Monitoring

This document defines the runtime metrics to capture
on the **EC2 `c7gd.16xlarge`** Ubuntu instance **while
the coreset pipeline is running**. All tools are
available in default Ubuntu repos via `apt`. No
external agents are required.

---

## 1. Prerequisites

```bash
sudo apt-get update
sudo apt-get install -y sysstat iotop nload htop dstat
```

Enable `sysstat` collection:

```bash
sudo sed -i 's/ENABLED="false"/ENABLED="true"/' \
  /etc/default/sysstat
sudo systemctl restart sysstat
```

---

## 2. CPU Utilization

Sustained CPU usage > 80%, steal < 1%.

Live view:

```bash
mpstat -P ALL 5
```

Key columns — `%usr` + `%sys` = effective utilization,
`%steal` = hypervisor contention,
`%iowait` = blocked on disk (should be < 5%).

Background logger:

```bash
mpstat -P ALL 10 \
  | ts '[%Y-%m-%d %H:%M:%S]' \
  >> /mnt/nvme/logs/cpu_mpstat.log &
```

CPU alert thresholds:

| Metric    | Healthy | Warning | Critical |
|-----------|---------|---------|----------|
| `%usr`    | > 70%   | < 50%   | < 30%    |
| `%steal`  | < 0.5%  | > 1%    | > 3%     |
| `%iowait` | < 3%    | > 5%    | > 15%    |
| `%idle`   | < 20%   | > 40%   | > 60%    |

---

## 3. Memory Utilization

RAM occupied > 80%, zero swap usage.

Live view:

```bash
free -h -s 10
```

Background logger:

```bash
vmstat 10 \
  | ts '[%Y-%m-%d %H:%M:%S]' \
  >> /mnt/nvme/logs/mem_vmstat.log &
```

Key `vmstat` fields — `si` / `so` (swap in/out) must
be 0, `free` should decrease as page cache fills,
`buff` / `cache` should grow with Parquet reads.

Memory alert thresholds:

| Metric      | Healthy | Warning | Critical  |
|-------------|---------|---------|-----------|
| RAM used    | > 80%   | < 60%   | < 40%     |
| Swap used   | 0 MB    | > 0 MB  | > 100 MB  |
| `si` + `so` | 0       | > 0     | > 10 MB/s |

---

## 4. Disk I/O – NVMe (Temp/Scratch)

NVMe handles all temp I/O with low latency.

Live view:

```bash
iostat -xdm /dev/nvme1n1 5
```

Key columns — `r_await` / `w_await` = read/write
latency (ms), `%util` = device saturation,
`rMB/s` / `wMB/s` = throughput.

NVMe alert thresholds:

| Metric    | Healthy  | Warning | Critical |
|-----------|----------|---------|----------|
| `r_await` | < 0.5 ms | > 1 ms  | > 5 ms   |
| `w_await` | < 0.5 ms | > 1 ms  | > 5 ms   |
| `%util`   | < 60%    | > 80%   | > 95%    |

---

## 5. Disk I/O – EBS gp3 (Durable Output)

EBS IOPS within provisioned limits, await < 3 ms.

Live view:

```bash
iostat -xdm /dev/nvme0n1 5
```

Background logger:

```bash
iostat -xdmt 10 \
  | ts '[%Y-%m-%d %H:%M:%S]' \
  >> /mnt/nvme/logs/disk_iostat.log &
```

EBS alert thresholds:

| Metric    | Healthy | Warning | Critical |
|-----------|---------|---------|----------|
| `r_await` | < 2 ms  | > 3 ms  | > 10 ms  |
| `w_await` | < 2 ms  | > 3 ms  | > 10 ms  |
| `%util`   | < 50%   | > 70%   | > 90%    |
| IOPS      | < 16k   | > 15k   | = 16k    |

If `await` spikes, the pipeline is EBS-bottlenecked.
Move temp I/O to NVMe.

---

## 6. S3 Streaming and Network

Parallel S3 downloads, minimal idle CPU from
network wait.

Live throughput:

```bash
nload -m eth0
```

Bandwidth logger:

```bash
sar -n DEV 10 \
  | ts '[%Y-%m-%d %H:%M:%S]' \
  >> /mnt/nvme/logs/net_sar.log &
```

Validate parallelism — the pipeline uses sharded
input files. Confirm multiple S3 GETs run
concurrently:

```bash
ss -tnp | grep -c ':443'
```

Network alert thresholds:

| Metric        | Healthy    | Warning    | Critical   |
|---------------|------------|------------|------------|
| RX throughput | > 500 MB/s | < 300 MB/s | < 100 MB/s |
| Active conns  | > 4        | < 3        | 1          |
| TCP retrans   | < 0.1%     | > 0.5%     | > 2%       |

---

## 7. Idle CPU Detection (I/O Wait)

No idle CPU cycles caused by I/O stalls.

Combined live view:

```bash
dstat --cpu --io --disk --net 5
```

Watch for `%wai` rising while `%usr` drops (I/O stall)
and `read`/`writ` columns going to zero (S3 gap).

Combined logger:

```bash
dstat --cpu --io --disk --net --output \
  /mnt/nvme/logs/dstat_all.csv 10 &
```

---

## 8. Process-Level Monitoring

Confirm the Python coreset processes are the
dominant consumers.

Filtered live view:

```bash
htop -p $(pgrep -d, -f coreset_builder)
```

Top CPU consumers:

```bash
ps aux --sort=-%cpu | head -20
```

Top memory consumers:

```bash
ps aux --sort=-%mem | head -20
```

---

## 9. Unified Monitoring Script

Run as a background job to capture all metrics
into `/mnt/nvme/logs/`:

```bash
#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/mnt/nvme/logs"
mkdir -p "$LOG_DIR"
INTERVAL=10

# CPU
mpstat -P ALL "$INTERVAL" \
  >> "$LOG_DIR/cpu.log" 2>&1 &

# Memory
vmstat "$INTERVAL" \
  >> "$LOG_DIR/mem.log" 2>&1 &

# Disk I/O (all devices)
iostat -xdmt "$INTERVAL" \
  >> "$LOG_DIR/disk.log" 2>&1 &

# Network
sar -n DEV "$INTERVAL" \
  >> "$LOG_DIR/net.log" 2>&1 &

# Combined view
dstat --cpu --io --disk --net --output \
  "$LOG_DIR/dstat.csv" "$INTERVAL" \
  > /dev/null 2>&1 &

echo "Monitoring started. Logs: $LOG_DIR"
echo "Stop all: kill $(jobs -p | tr '\n' ' ')"
wait
```

Save as `monitor.sh` and run:

```bash
chmod +x monitor.sh
nohup ./monitor.sh &
```

---

## 10. Quick Health Check (One-Liner)

Paste during the run for an instant snapshot:

```bash
echo "=== CPU ===" && \
  mpstat 1 1 | tail -1 && \
echo "=== MEM ===" && \
  free -h | grep Mem && \
echo "=== SWAP ===" && \
  free -h | grep Swap && \
echo "=== DISK ===" && \
  iostat -xdm 1 1 | tail -3 && \
echo "=== NET ===" && \
  sar -n DEV 1 1 | tail -2
```

---

## Observability Reference Table

| #  | Metric     | Tool     | Threshold           |
|----|------------|----------|---------------------|
| 1  | CPU usage  | `mpstat` | > 80% usr+sys       |
| 2  | CPU steal  | `mpstat` | < 1%                |
| 3  | CPU iowait | `mpstat` | < 5%                |
| 4  | RAM used   | `free`   | > 80%               |
| 5  | Swap used  | `free`   | = 0                 |
| 6  | NVMe await | `iostat` | < 0.5 ms            |
| 7  | NVMe util  | `iostat` | < 60%               |
| 8  | EBS await  | `iostat` | < 3 ms              |
| 9  | EBS util   | `iostat` | < 50%               |
| 10 | EBS IOPS   | `iostat` | < 16k provisioned   |
| 11 | Net RX     | `nload`  | > 500 MB/s          |
| 12 | S3 conns   | `ss`     | > 4 parallel        |
| 13 | Idle CPU   | `dstat`  | wai < 5%            |
