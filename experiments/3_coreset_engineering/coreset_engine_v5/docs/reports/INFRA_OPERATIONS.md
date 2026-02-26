# INFRA_OPERATIONS – Execution, Monitoring & Reporting

This document covers the full operational lifecycle
for the coreset pipeline on the **EC2 `c7gd.16xlarge`**
Ubuntu instance — from S3 ingestion through runtime
monitoring to post-run reporting.

All tools are available in default Ubuntu repos
via `apt`. No external agents are required.

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

## 2. Storage Setup

### 2.1 Local NVMe (Temp/Scratch)

Use for batch spill, chunk materialization,
shuffle buffers, and monitoring logs.

Mount and verify latency:

```bash
lsblk
sudo mkfs.ext4 /dev/nvme1n1
sudo mkdir -p /mnt/nvme
sudo mount /dev/nvme1n1 /mnt/nvme
sudo chown ubuntu:ubuntu /mnt/nvme
fio --filename=/mnt/nvme/test --rw=randread \
  --bs=4k --numjobs=4 --size=256M --runtime=5
```

### 2.2 EBS gp3 (Durable Output)

Recommended configuration:

- Size: 1–1.2 TB
- IOPS: 16k provisioned
- Throughput: 750 MB/s

Verify:

```bash
iostat -x 5
```

---

## 3. S3 Ingestion

```bash
aws s3 cp s3://bucket/path ./ --recursive
```

Validate:

- ≥500–800 MB/s sustained
- Parallel downloads enabled

Monitor:

```bash
nload -m eth0
ss -tnp | grep -c ':443'
```

---

## 4. Metric Thresholds

### 4.1 CPU

| Metric    | Healthy | Warning | Critical |
|-----------|---------|---------|----------|
| `%usr`    | > 70%   | < 50%   | < 30%    |
| `%steal`  | < 0.5%  | > 1%    | > 3%     |
| `%iowait` | < 3%    | > 5%    | > 15%    |
| `%idle`   | < 20%   | > 40%   | > 60%    |

### 4.2 Memory

| Metric      | Healthy | Warning | Critical  |
|-------------|---------|---------|-----------|
| RAM used    | > 80%   | < 60%   | < 40%     |
| Swap used   | 0 MB    | > 0 MB  | > 100 MB  |
| `si` + `so` | 0       | > 0     | > 10 MB/s |

### 4.3 NVMe (Ephemeral)

| Metric    | Healthy  | Warning | Critical |
|-----------|----------|---------|----------|
| `r_await` | < 0.5 ms | > 1 ms  | > 5 ms   |
| `w_await` | < 0.5 ms | > 1 ms  | > 5 ms   |
| `%util`   | < 60%    | > 80%   | > 95%    |

### 4.4 EBS gp3

| Metric    | Healthy | Warning | Critical |
|-----------|---------|---------|----------|
| `r_await` | < 2 ms  | > 3 ms  | > 10 ms  |
| `w_await` | < 2 ms  | > 3 ms  | > 10 ms  |
| `%util`   | < 50%   | > 70%   | > 90%    |
| IOPS      | < 16k   | > 15k   | = 16k    |

If `await` spikes, the pipeline is EBS-bottlenecked.
Move temp I/O to NVMe.

### 4.5 Network

| Metric        | Healthy    | Warning    | Critical   |
|---------------|------------|------------|------------|
| RX throughput | > 500 MB/s | < 300 MB/s | < 100 MB/s |
| Active conns  | > 4        | < 3        | 1          |
| TCP retrans   | < 0.1%     | > 0.5%     | > 2%       |

---

## 5. Live Monitoring Commands

### 5.1 CPU

```bash
mpstat -P ALL 5
```

Key: `%usr` + `%sys` = utilization,
`%steal` = hypervisor, `%iowait` < 5%.

### 5.2 Memory

```bash
free -h -s 10
```

Key: `si`/`so` must be 0, `buff`/`cache` should
grow with Parquet reads.

### 5.3 NVMe I/O

```bash
iostat -xdm /dev/nvme1n1 5
```

### 5.4 EBS I/O

```bash
iostat -xdm /dev/nvme0n1 5
```

### 5.5 Network

```bash
nload -m eth0
```

### 5.6 Combined View

```bash
dstat --cpu --io --disk --net 5
```

Watch for `%wai` rising while `%usr` drops
(I/O stall).

### 5.7 Process-Level

```bash
htop -p $(pgrep -d, -f coreset_builder)
ps aux --sort=-%cpu | head -20
ps aux --sort=-%mem | head -20
```

### 5.8 Quick Health Check (One-Liner)

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

## 6. Automated Monitoring Scripts

Three scripts in `scripts/` automate the full
monitoring lifecycle:

| Script              | Purpose                        |
|---------------------|--------------------------------|
| `monitor.sh`        | Start all metric collectors    |
| `monitor_report.sh` | Text pass/fail summary         |
| `monitor_report.py` | HTML charts + CSV + S3 upload  |

### 6.1 Start Monitoring

Run **before** launching the coreset pipeline:

```bash
chmod +x scripts/monitor.sh
nohup scripts/monitor.sh &
```

Override defaults:

```bash
export LOG_DIR=/mnt/nvme/logs
export INTERVAL=10
nohup scripts/monitor.sh &
```

Verify:

```bash
cat /mnt/nvme/logs/monitor.pid
ps aux | grep -E 'mpstat|vmstat|iostat|sar|dstat'
```

Logs created in `/mnt/nvme/logs/`:

| Log file   | Collector | Contents                 |
|------------|-----------|--------------------------|
| `cpu.log`  | `mpstat`  | Per-CPU usr/sys/steal    |
| `mem.log`  | `vmstat`  | Free/buff/cache, swap    |
| `disk.log` | `iostat`  | Per-device IOPS, await   |
| `net.log`  | `sar`     | Per-interface RX/TX      |
| `dstat.csv`| `dstat`   | Combined CSV for charts  |

### 6.2 Stop Monitoring

After the pipeline finishes:

```bash
kill $(cat /mnt/nvme/logs/monitor.pid)
```

---

## 7. Post-Run Reporting

### 7.1 Text Summary (Option 1)

```bash
./scripts/monitor_report.sh
./scripts/monitor_report.sh /path/to/logs
```

### 7.2 HTML Charts (Option 3)

```bash
python3 scripts/monitor_report.py
python3 scripts/monitor_report.py /path/to/logs
```

Download to your Mac:

```bash
scp -i key.pem \
  ubuntu@ec2:..../report_*.html ./
```

### 7.3 CSV Export (Option 2)

```bash
scp -i key.pem \
  ubuntu@ec2:/mnt/nvme/logs/dstat.csv ./
```

### 7.4 Upload to S3 (Option 4)

```bash
python3 scripts/monitor_report.py \
  --upload s3://your-bucket/infra-reports
```

---

## 8. Wall-Clock Decomposition

Log timestamps for each phase to validate
scaling assumptions:

- S3 ingest start/end
- Compute start/end per stage
- Spill/checkpoint phases
- Output write-back

---

## 9. Failure & Retry Handling

- Expect 10–15% overhead from retries
- Checkpoints persist to EBS (survives restarts)
- Temp data stays on NVMe (lost on termination)
- From the review report: Spot instances had ~20
  interruptions on a 10 GB run — prefer on-demand
  for production

---

## 10. Post-Run Validation

Confirm before finalizing cost numbers:

- CPU-bound execution (not I/O-bound)
- No swap usage
- No EBS saturation
- Output integrity verified

---

## 11. Complete Workflow

```bash
# 1. Install tools (one-time)
sudo apt-get install -y sysstat iotop nload htop dstat

# 2. Start monitors
nohup scripts/monitor.sh &

# 3. Run coreset pipeline
./shard.sh ...

# 4. Stop monitors
kill $(cat /mnt/nvme/logs/monitor.pid)

# 5. Text summary
./scripts/monitor_report.sh

# 6. HTML charts
python3 scripts/monitor_report.py

# 7. Upload to S3
python3 scripts/monitor_report.py \
  --upload s3://your-bucket/infra-reports
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
