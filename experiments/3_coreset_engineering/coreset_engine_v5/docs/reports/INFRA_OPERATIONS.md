# INFRA_OPERATIONS – Execution, Monitoring & Reporting

This document covers the full operational lifecycle for the coreset pipeline on **AWS EC2**. It prioritizes the **Production Playbook (`commands.sh`)** for automated execution while retaining all technical reference details.

---

## 1. Production Playbook: `commands.sh`

The `commands.sh` script is the primary entry point for production runs. It automates system setup, AWS validation, dependency sync, infrastructure checks, monitoring, and pipeline launch.

### Deployment Options

Choose the profile that matches your provisioned EC2 hardware:

#### Option 1: EBS Gp3 Only (`c6i`, `m6i`, `m7i-flex`)

Use this for instances without local NVMe drives.

```bash
# 1. Export Parameters
export S3_BUCKET="era4-lightening-lm-lake"
export S3_INPUT_PATH="s3://era4-lightening-lm-lake/processed_dataset/curriculum_data/"
export S3_PREFIX="processed_dataset/curriculum_data/source=C4/bands/band=B0/"

# 2. Infra Overrides
export EXPECTED_INSTANCE_TYPE="m7i-flex.large" # Change to your type
export ENABLE_NVME="false"                    # Disable NVMe checks
export SKIP_EBS_VALIDATION="true"              # Enable if using small volumes

# 3. Launch
./commands.sh
```

#### Option 2: Hybrid Storage (NVMe + EBS Gp3)

Use for `c7gd`, `m5d`, or any instance with high-speed local ephemeral storage (`/dev/nvmeXn1`).

```bash
# 1. NVMe Setup (Choose One)

**Option A: Automated (Recommended)**
Safely detects, formats (if needed), and mounts to `/mnt/nvme`.
```bash
sudo ./scripts/setup_nvme.sh
```

**Option B: Manual (Hardware Reference)**
Use if the drive is not at `/dev/nvme1n1` or for custom mounting.

```bash
sudo mkfs.ext4 /dev/nvme1n1
sudo mkdir -p /mnt/nvme
sudo mount /dev/nvme1n1 /mnt/nvme
sudo chown ubuntu:ubuntu /mnt/nvme
```

# 2. Export Parameters

export S3_BUCKET="era4-lightening-lm-lake"
export S3_PREFIX="processed_dataset/curriculum_data/source=C4/bands/band=B0/"

# 3. Infra Overrides

export EXPECTED_INSTANCE_TYPE="c7gd.16xlarge"
export ENABLE_NVME="true"

# 4. Launch

./commands.sh
```

> [!NOTE]
> `setup_nvme.sh` is a safe, one-click script that won't format your drive if it already contains a filesystem.
>
> [!IMPORTANT]
> Always use **standard straight quotes** (`"`) for exports. Bash does not recognize "smart" curly quotes (`”`) from document editors.

---

## 2. Infrastructure Prerequisites

Run these once on a fresh instance to enable full observability:

```bash
# 1. Install required tools
sudo apt-get update
sudo apt-get install -y sysstat iotop nload htop dstat fio

# 2. Enable background metric collection
sudo sed -i 's/ENABLED="false"/ENABLED="true"/' /etc/default/sysstat
sudo systemctl restart sysstat
```

---

## 3. Storage Setup (Reference)

### 3.1 Local NVMe (Temp/Scratch)

Use for batch spill, chunk materialization, shuffle buffers, and monitoring logs.

Mount and verify performance:

```bash
lsblk
sudo mkfs.ext4 /dev/nvme1n1
sudo mkdir -p /mnt/nvme
sudo mount /dev/nvme1n1 /mnt/nvme
sudo chown ubuntu:ubuntu /mnt/nvme
fio --filename=/mnt/nvme/test --rw=randread --bs=4k --numjobs=4 --size=256M --runtime=5
```

### 3.2 EBS gp3 (Durable Output)

Recommended configuration:

- Size: 1–1.2 TB
- IOPS: 16k provisioned
- Throughput: 750 MB/s

Verify:

```bash
iostat -x 5
```

---

## 4. Metric Thresholds

### 4.1 CPU

| Metric | Healthy | Warning | Critical |
| --- | --- | --- | --- |
| `%usr` | > 70% | < 50% | < 30% |
| `%steal` | < 0.5% | > 1% | > 3% |
| `%iowait` | < 3% | > 5% | > 15% |
| `%idle` | < 20% | > 40% | > 60% |

### 4.2 Memory

| Metric | Healthy | Warning | Critical |
| --- | --- | --- | --- |
| RAM used | > 80% | < 60% | < 40% |
| Swap used | 0 MB | > 0 MB | > 100 MB |
| `si` + `so` | 0 | > 0 | > 10 MB/s |

### 4.3 NVMe (Ephemeral)

| Metric | Healthy | Warning | Critical |
| --- | --- | --- | --- |
| `r_await` | < 0.5 ms | > 1 ms | > 5 ms |
| `w_await` | < 0.5 ms | > 1 ms | > 5 ms |
| `%util` | < 60% | > 80% | > 95% |

### 4.4 EBS gp3

| Metric | Healthy | Warning | Critical |
| --- | --- | --- | --- |
| `r_await` | < 2 ms | > 3 ms | > 10 ms |
| `w_await` | < 2 ms | > 3 ms | > 10 ms |
| `%util` | < 50% | > 70% | > 90% |
| IOPS | < 16k | > 15k | = 16k |

---

## 5. Live Monitoring Commands

### 5.1 CPU

```bash
mpstat -P ALL 5
```

Key: `%usr` + `%sys` = utilization, `%steal` = hypervisor, `%iowait` < 5%.

### 5.2 Memory

```bash
free -h -s 5
```

Key: `si`/`so` must be 0, `buff`/`cache` should grow with Parquet reads.

### 5.3 NVMe I/O

```bash
iostat -xdm /dev/nvme1n1 5
```

### 5.4 EBS I/O

```bash
iostat -xdm /dev/nvme0n1 5
```

### 5.5 Combined View

```bash
dstat --cpu --io --disk --net 5
```

---

## 6. Automated Monitoring Scripts

Three scripts in `scripts/` automate the full monitoring lifecycle:

| Script | Purpose |
| --- | --- |
| `monitor.sh` | Start all metric collectors |
| `monitor_report.sh` | Text pass/fail summary |
| `monitor_report.py` | HTML charts + CSV + S3 upload |

### 6.1 Start Monitoring

Run **before** launching the coreset pipeline:

```bash
chmod +x scripts/monitor.sh
nohup scripts/monitor.sh &
```

Logs created in `/mnt/nvme/logs/`:

| Log file | Collector | Contents |
| --- | --- | --- |
| `cpu.log` | `mpstat` | Per-CPU usr/sys/steal |
| `mem.log` | `vmstat` | Free/buff/cache, swap |
| `disk.log` | `iostat` | Per-device IOPS, await |
| `net.log` | `sar` | Per-interface RX/TX |
| `dstat.csv` | `dstat` | Combined CSV for charts |

### 6.2 Stop Monitoring

After the pipeline finishes:

```bash
kill $(cat /mnt/nvme/logs/monitor.pid)
```

---

## 7. Post-Run Reporting

### 7.1 Text Summary

```bash
./scripts/monitor_report.sh
```

### 7.2 HTML Charts

```bash
python3 scripts/monitor_report.py
```

### 7.3 Upload to S3

```bash
python3 scripts/monitor_report.py --upload s3://your-bucket/infra-reports
```

---

## 8. Wall-Clock Decomposition

Log timestamps for each phase to validate scaling assumptions:

- S3 ingest start/end
- Compute start/end per stage
- Spill/checkpoint phases
- Output write-back

---

## 9. Failure & Retry Handling

- Checkpoints persist to EBS (survives restarts).
- Temp data stays on NVMe (lost on termination).
- Spot instances had ~20 interruptions on a 10 GB run — prefer on-demand for production.

---

## 10. Post-Run Validation

Confirm before finalizing cost numbers:

- CPU-bound execution (not I/O-bound).
- No swap usage.
- No EBS saturation.
- Output integrity verified.

---

## 11. Complete Workflow

```bash
# 1. Start monitors
nohup scripts/monitor.sh &

# 2. Run coreset pipeline
./commands.sh

# 3. Stop monitors (if background mode used)
kill $(cat /mnt/nvme/logs/monitor.pid)

# 4. Generate Reports
python3 scripts/monitor_report.py
```

---

## Observability Reference Table

| # | Metric | Tool | Threshold |
| --- | --- | --- | --- |
| 1 | CPU usage | `mpstat` | > 80% usr+sys |
| 2 | CPU steal | `mpstat` | < 1% |
| 3 | CPU iowait | `mpstat` | < 5% |
| 4 | RAM used | `free` | > 80% |
| 5 | Swap used | `free` | = 0 |
| 6 | NVMe await | `iostat` | < 0.5 ms |
| 7 | NVMe util | `iostat` | < 60% |
| 8 | EBS await | `iostat` | < 3 ms |
| 9 | EBS util | `iostat` | < 50% |
| 10 | EBS IOPS | `iostat` | < 16k provisioned |
| 11 | Net RX | `nload` | > 500 MB/s |
| 12 | S3 conns | `ss` | > 4 parallel |
| 13 | Idle CPU | `dstat` | wai < 5% |
