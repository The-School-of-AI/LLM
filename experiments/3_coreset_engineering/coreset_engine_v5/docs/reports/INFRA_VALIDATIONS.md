# INFRA_VALIDATIONS – EC2 Pre-Run Checks

This document lists every validation check to perform on
the EC2 Ubuntu instance **before** launching the coreset
pipeline. Each section includes the exact commands and
the expected pass criteria. A summary reference table at
the end can be used to build an automated validation
script.

---

## 1. Instance Identity

Confirm the instance type matches the design target
(`c7gd.16xlarge` or equivalent).

```bash
# Instance type from metadata service
TOKEN=$(curl -sX PUT \
  "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

curl -sH "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-type
```

Expected: `c7gd.16xlarge`

---

## 2. CPU Topology

```bash
lscpu | grep -E "^CPU\(s\)|^Thread|^Core|^Socket|^NUMA"
numactl --hardware
```

Pass criteria:

- 64 vCPUs visible
- Single NUMA node (all memory local)
- No hyper-threading surprises

---

## 3. CPU Steal and Utilization Baseline

```bash
# 10-second snapshot across all CPUs
mpstat -P ALL 5 2
```

Pass criteria:

- `%steal` < 1% on every CPU
- `%idle` > 90% (instance is quiet before run)

---

## 4. Memory

```bash
free -h
cat /proc/meminfo | grep -E "MemTotal|SwapTotal"
```

Pass criteria:

- Total RAM ≥ 120 GiB
- Swap = 0 (disabled)

If swap is enabled, disable it:

```bash
sudo swapoff -a
```

---

## 5. Local NVMe Discovery and Mount

```bash
# List all block devices
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE

# Identify the ephemeral NVMe (usually /dev/nvme1n1)
sudo file -s /dev/nvme1n1
```

Format and mount (first boot only):

```bash
sudo mkfs.ext4 /dev/nvme1n1
sudo mkdir -p /mnt/nvme
sudo mount /dev/nvme1n1 /mnt/nvme
sudo chown ubuntu:ubuntu /mnt/nvme
```

Verify after mount:

```bash
df -h /mnt/nvme
```

Pass criteria:

- NVMe device present and mounted at `/mnt/nvme`
- Usable capacity matches instance spec

---

## 6. NVMe Latency Benchmark

```bash
sudo apt-get install -y fio

fio --name=nvme_randread \
    --filename=/mnt/nvme/fio_test \
    --rw=randread \
    --bs=4k \
    --ioengine=libaio \
    --iodepth=32 \
    --numjobs=4 \
    --size=1G \
    --runtime=10 \
    --time_based \
    --group_reporting
```

Pass criteria:

- Average latency < 200 µs
- IOPS > 100k (4k random read)

Clean up after test:

```bash
rm -f /mnt/nvme/fio_test
```

---

## 7. EBS gp3 Volume Check

```bash
# Confirm EBS root or data volume
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT

# Live IOPS and throughput baseline
sudo apt-get install -y sysstat
iostat -x 5 2
```

Pass criteria:

- EBS volume size ≥ 1 TB
- Provisioned IOPS ≥ 16,000
- Provisioned throughput ≥ 750 MB/s
- `await` < 3 ms under idle

Verify provisioned IOPS via AWS CLI:

```bash
VOLUME_ID=$(
  curl -sH "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/block-device-mapping/ebs1 \
  2>/dev/null || echo "check-manually"
)
aws ec2 describe-volumes \
  --volume-ids "$VOLUME_ID" \
  --query 'Volumes[0].{Size:Size,Iops:Iops,Throughput:Throughput}'
```

---

## 8. Network and S3 Connectivity

```bash
# S3 connectivity
aws s3 ls s3://your-bucket-name/ --region us-east-1

# Download speed test (small file)
time aws s3 cp s3://your-bucket-name/test_file.parquet \
  /mnt/nvme/test_file.parquet

# Network bandwidth
curl -s https://checkip.amazonaws.com
```

Pass criteria:

- S3 bucket accessible without errors
- Download speed > 500 MB/s sustained

---

## 9. Python Environment

```bash
python3 --version
pip3 --version

# Verify critical packages
python3 -c "import pyarrow; print(pyarrow.__version__)"
python3 -c "import pandas; print(pandas.__version__)"
python3 -c "import yaml; print(yaml.__version__)"
```

Pass criteria:

- Python ≥ 3.10
- `pyarrow`, `pandas`, `pyyaml` importable

---

## 10. Disk Space Budget

```bash
df -h /mnt/nvme
df -h /
```

Pass criteria:

- `/mnt/nvme` (NVMe temp): ≥ 400 GB free
- `/` or EBS data mount: ≥ 800 GB free

---

## 11. OS Limits

```bash
ulimit -n     # open files
ulimit -u     # max user processes
cat /proc/sys/vm/swappiness
```

Pass criteria:

- Open files ≥ 65536
- Max processes ≥ 4096
- Swappiness = 0 or 1

Fix if needed:

```bash
sudo sysctl -w vm.swappiness=0
ulimit -n 65536
```

---

## 12. Time Sync

```bash
timedatectl status
chronyc tracking 2>/dev/null || ntpstat 2>/dev/null
```

Pass criteria:

- NTP synchronized = yes
- Clock offset < 100 ms

---

## Validation Reference Table

The table below summarizes every check. Use
it as a blueprint for an automated validation script.

| #  | Category  | Command                    | Pass Criteria              |
|----|-----------|----------------------------|----------------------------|
| 1  | Instance  | `curl .../instance-type`   | = `c7gd.16xlarge`          |
| 2  | CPU count | `lscpu`                    | 64 vCPUs                   |
| 3  | NUMA      | `numactl --hardware`       | Single NUMA node           |
| 4  | CPU steal | `mpstat -P ALL 5 2`        | `%steal` < 1%              |
| 5  | RAM       | `free -h`                  | ≥ 120 GiB                  |
| 6  | Swap      | `cat /proc/meminfo`        | SwapTotal = 0              |
| 7  | NVMe mnt  | `df -h /mnt/nvme`          | Mounted, capacity ok       |
| 8  | NVMe lat  | `fio --rw=randread`        | Avg lat < 200 µs           |
| 9  | NVMe IOPS | `fio --rw=randread`        | > 100k IOPS (4k)           |
| 10 | EBS size  | `lsblk`                    | ≥ 1 TB                     |
| 11 | EBS IOPS  | `aws ec2 describe-volumes` | ≥ 16k provisioned          |
| 12 | EBS lat   | `iostat -x 5`              | `await` < 3 ms             |
| 13 | S3 access | `aws s3 ls`                | Bucket listing ok          |
| 14 | S3 speed  | `aws s3 cp` (timed)        | > 500 MB/s                 |
| 15 | Python    | `python3 --version`        | ≥ 3.10                     |
| 16 | Packages  | `import pyarrow,pandas`    | All importable             |
| 17 | NVMe free | `df -h /mnt/nvme`          | ≥ 400 GB free              |
| 18 | EBS free  | `df -h /`                  | ≥ 800 GB free              |
| 19 | Open file | `ulimit -n`                | ≥ 65536                    |
| 20 | Swapiness | `sysctl vm.swappiness`     | 0 or 1                     |
| 21 | Time sync | `timedatectl status`       | NTP synced, < 100 ms       |

---

## Automated Validation Script

All checks above are implemented in
`validate_infra.sh` at the project root.

### Usage

```bash
chmod +x validate_infra.sh
sudo -E ./validate_infra.sh
```

Override S3 defaults (optional):

```bash
export S3_BUCKET="other-bucket"
export S3_PREFIX="some/other/path/"
export S3_TEST_COUNT=50
sudo ./validate_infra.sh
```

### What Each Check Does

| #  | Check     | How it validates               |
|----|-----------|--------------------------------|
| 1  | Instance  | IMDS v2 metadata query         |
| 2  | CPU count | `nproc` ≥ 64                   |
| 3  | NUMA      | `numactl --hardware` = 1 node  |
| 4  | CPU steal | `mpstat` steal < 1%            |
| 5  | RAM       | `/proc/meminfo` ≥ 120 GiB      |
| 6  | Swap      | SwapTotal = 0                  |
| 7  | NVMe dev  | Scans `/dev/nvme1n1`, `2n1`    |
| 8  | NVMe mnt  | `mountpoint /mnt/nvme`         |
| 9  | NVMe lat  | 5s `fio` randread benchmark    |
| 10 | EBS size  | `df /` ≥ 1 TB                  |
| 11 | EBS IOPS  | `describe-volumes` ≥ 16k       |
| 12 | EBS lat   | `iostat` await < 3 ms          |
| 13 | S3 access | `aws s3 ls` on bucket          |
| 14 | S3 speed  | 25-file parallel download      |
| 15 | Python    | Version ≥ 3.10                 |
| 16 | Packages  | `pyarrow`, `pandas`, `yaml`    |
| 17 | NVMe free | ≥ 400 GB                       |
| 18 | EBS free  | ≥ 800 GB                       |
| 19 | Open file | `ulimit -n` ≥ 65536            |
| 20 | Swapiness | `vm.swappiness` ≤ 1            |
| 21 | Time sync | NTP via `timedatectl`          |

### Exit Behaviour

The script prints a colour-coded summary:

- ✅ **PASS** — check met the threshold
- ❌ **FAIL** — check did not meet threshold
- ⚠️ **WARN** — non-critical or tool missing

Exit code `0` = all passed, `1` = any failure.
