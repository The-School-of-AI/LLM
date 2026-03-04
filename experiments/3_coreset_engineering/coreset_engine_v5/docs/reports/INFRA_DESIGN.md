# Conversation Summary (Context from the Q&A)

This design document is the result of an end-to-end discussion comparing a
**local MacBook Pro (Apple M4, 16 cores, 64 GB RAM)** execution of a coreset
generation pipeline with an **AWS EC2-based deployment**.

Key conclusions reached during the conversation:

- Raw EC2 vCPU counts (e.g., C6i) do **not** map well to Apple Silicon performance.
- Apple M-series systems benefit from:
  - Very high IPC
  - Unified memory
  - Extremely fast local NVMe
- AWS EBS (gp3 / io2) **cannot match Mac SSD peak IOPS**.
- For parity, **local NVMe on EC2 is mandatory** for temp and spill I/O.
- The pipeline is **CPU- and throughput-bound**, not peak-IOPS-bound.
- gp3 with 12k–20k IOPS is sufficient **only when access is sequential** and temp
  I/O is offloaded.
- Correct AWS estimates require separating:
  - Input ingestion (S3)
  - Temporary working data
  - Durable outputs
- Linear scaling from 22 GB → 500 GB is pessimistic; **parallelism-aware scaling**
  is realistic.

This document captures the **design decisions**.  
`Implementation.md` defines how to execute and validate them.

---

## 1. Problem Statement

Design an AWS EC2 infrastructure that:

- Matches the performance semantics of a local Apple M4 system
- Processes **500 GB of Parquet data streamed from S3**
- Produces **~750 GB of outputs, checkpoints, and manifests**
- Enables a **non-conservative but defensible cost estimate**
- Avoids over-provisioning while preserving throughput

---

## 2. Local Baseline (Design Reference)

| Dimension | Local MacBook Pro (M4)                      |
|-----------|---------------------------------------------|
| CPU       | 16 cores (12 performance cores)             |
| Memory    | 64 GB unified memory                        |
| Storage   | Local NVMe (very high IOPS, <100 µs latency)|
| Input     | 22–23 GB (local disk)                       |
| Runtime   | ~100 minutes                                |
| Output    | Small Parquet shards + manifests            |

This defines the **performance contract** AWS must satisfy.

---

## 3. AWS Design Goals

1. Match **storage semantics**, not just CPU count
2. Avoid EBS for temp / scratch I/O
3. Sustain high CPU utilization (>80%)
4. Prevent I/O wait from dominating wall time
5. Enable predictable scaling from pilot → full run

---

## 4. Compute Architecture

### 4.1 Instance Family Selection

| Family | Rationale                                               |
|--------|---------------------------------------------------------|
| C7g    | Best price/perf, high IPC, ARM similar to Apple Silicon |
| C7gd   | Adds **local NVMe**, required for parity                |
| C7a    | Acceptable x86 fallback                                 |

#### Selected instance

```text
c7gd.16xlarge
```

- 64 vCPUs (Graviton3)
- 128 GB RAM
- Local NVMe SSD

Rationale:

- Matches Apple-class IPC behavior
- Supports shard parallelism
- Provides NVMe-backed scratch space

---

## 5. Storage Design

### 5.1 Storage Role Separation

| Data Type                         | Storage    |
|-----------------------------------|------------|
| Input Parquet                     | S3         |
| Temp chunks / spill / shuffle     | Local NVMe |
| Outputs / checkpoints / manifests | EBS gp3    |

This mirrors the **local-disk semantics** of macOS.

---

## 6. Decision Tree: gp3 vs NVMe vs io2

Is the data temporary or scratch?
└─ Yes → Local NVMe
└─ No
└─ Is access mostly sequential (>64 KB reads)?
└─ Yes → gp3 (12k–20k IOPS)
└─ No
└─ Is random I/O latency critical?
└─ Yes → io2 Block Express
└─ No → Re-evaluate batching & file layout

Key insight:

> Peak Mac SSD IOPS ≠ required AWS IOPS

Throughput and locality matter more than theoretical IOPS.

---

## 7. Scaling Model (Design-Level)

### 7.1 Empirical Baseline

- 22 GB → ~100 minutes on Mac
- ≈ 0.22 GB/min

### 7.2 Naïve Linear Scaling (Rejected)

- 500 GB → ~38 hours  
- Overly pessimistic and ignores parallelism

### 7.3 Parallel-Aware AWS Scaling

Assumptions:

- 4–6× shard parallelism
- Sustained CPU utilization
- NVMe-backed temp I/O

#### Expected wall-clock time

```text
7–10 hours
```

---

## 8. Cost Model (AWS Pricing Calculator Export)

Region: **us-east-1**  
Rate type: before discounts (on-demand)

| Component             | Detail                     | Cost (USD)    |
|-----------------------|----------------------------|---------------|
| EC2 `c7gd.16xlarge`   | 730 hrs/mo (on-demand)     | $2,200.95     |
| EBS gp3 – IOPS        | 13,000 IOPS-Mo             | $65.00        |
| EBS gp3 – storage (A) | 200 GB-Mo                  | $16.00        |
| EBS gp3 – storage (B) | 750 GB-Mo                  | $60.00        |
| EBS gp3 – throughput  | 1.83 GiBps-Mo              | $75.00        |
| Data transfer in      | 750 GB inbound             | $0.00         |
| **Total (monthly)**   |                            | **$2,416.95** |

> **Note:** This is a full-month on-demand estimate.  
> For a **48-hour full run**, pro-rate the EC2 cost:  
> $2,200.95 ÷ 730 hrs × 48 hrs ≈ **~$145**.  
> Total run cost (compute + EBS + data):  
> **~$205–220**.

---

## 9. Design Summary

- Local NVMe is **non-negotiable** for Mac parity
- gp3 is sufficient for durable outputs
- io2 only for truly random I/O workloads
- Costs scale with **parallelism**, not linear data size
- Accurate estimates require **behavioral parity**, not instance inflation

---

## 10. Design Outcome

This design provides:

- A defensible AWS architecture
- A reproducible cost model
- A clear rationale for infra choices
- A foundation for execution and validation in `Implementation.md`
