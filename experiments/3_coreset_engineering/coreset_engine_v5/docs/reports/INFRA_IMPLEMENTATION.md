# Implementation.md – AWS Coreset Pipeline (Execution & Validation)

## 1. Purpose

This document operationalizes **Design.md**. It defines **how to deploy,
validate, and measure** the AWS infrastructure so that execution semantics
match the local MacBook Pro baseline.

---

## 2. Pre-Run Validation Checklist

### 2.1 CPU & NUMA

```bash
lscpu
numactl --hardware
mpstat -P ALL 5
```

Requirements:

* Single NUMA node
* Minimal CPU steal
* Sustained utilization during compute

---

### 2.2 Memory & Swap

```bash
free -h
vmstat 5
```

Requirements:

* Swap disabled or unused
* Page cache growth during Parquet reads
* No allocator thrashing

---

## 3. Storage Setup

### 3.1 Local NVMe

Use for:

* Batch spill
* Chunk materialization
* Shuffle buffers

Mount and verify latency:

```bash
lsblk
fio --filename=/mnt/nvme/test --rw=randread --bs=4k --numjobs=4
```

---

### 3.2 EBS gp3

Recommended:

* Size: 1–1.2 TB
* IOPS: 16k
* Throughput: 750 MB/s

Verify:

```bash
iostat -x 5
```

---

## 4. S3 Ingestion

```bash
aws s3 cp s3://bucket/path ./ --recursive
```

Validate:

* ≥500–800 MB/s sustained
* Parallel downloads enabled

Monitor:

```bash
iftop
nload
```

---

## 5. Runtime Instrumentation

### 5.1 Disk Behavior

```bash
iostat -x 5
```

Targets:

* await < 2–3 ms (EBS)
* %util < 70%

---

### 5.2 I/O Pattern Inspection

```bash
strace -T -e trace=pread64,read python job.py
```

Confirm:

* Mostly sequential reads
* Large block sizes

---

## 6. Wall-Clock Decomposition

Log timestamps for:

* S3 ingest start/end
* Compute start/end
* Spill/checkpoint phases
* Output writes

This validates scaling assumptions.

---

## 7. Pilot Run Protocol

1. Run **50–75 GB** subset
2. Capture:

   * Runtime
   * CPU %, IO wait, network
3. Validate against Mac ratios
4. Extrapolate to 500 GB using parallelism

---

## 8. Failure & Retry Handling

* Expect 10–15% overhead
* Persist checkpoints to EBS
* Temp data remains on NVMe

---

## 9. Post-Run Validation

Confirm:

* CPU-bound execution
* No swap usage
* No EBS saturation
* Output integrity

Only then finalize cost numbers.

---

## 10. Implementation Summary

* NVMe is mandatory for temp I/O
* EBS is durable-only
* Measure before extrapolating
* Cost accuracy comes from parity, not instance size
