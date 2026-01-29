# Throughput & Waste Report

Date: 2026-01-29

## Summary
This report captures the metrics required to attribute slowdowns to data pipeline, synchronization, or kernel inefficiencies.

## Per-Stage Metrics (Fill During Runs)

### Stage 1 (1B Dense)
- tokens/sec per GPU: TBD
- effective FLOPs utilization: TBD
- idle time (%): TBD
- synchronization waste (%): TBD
- checkpoint overhead (%): TBD
- comm vs compute ratio: TBD

### Stage 2 (3B MoE-small)
- tokens/sec per GPU: TBD
- effective FLOPs utilization: TBD
- idle time (%): TBD
- synchronization waste (%): TBD
- checkpoint overhead (%): TBD
- comm vs compute ratio: TBD

### Stage 3 (8B Dense-deep)
- tokens/sec per GPU: TBD
- effective FLOPs utilization: TBD
- idle time (%): TBD
- synchronization waste (%): TBD
- checkpoint overhead (%): TBD
- comm vs compute ratio: TBD

### Stage 4 (70B MoE-large)
- tokens/sec per GPU: TBD
- effective FLOPs utilization: TBD
- idle time (%): TBD
- synchronization waste (%): TBD
- checkpoint overhead (%): TBD
- comm vs compute ratio: TBD

## Bottleneck Attribution
- Data pipeline: evaluate via data_wait_s ratio
- Synchronization: evaluate via NCCL/all-reduce timing
- Kernel: evaluate via occupancy and SM utilization

## Mitigation Decisions
Record any parameter changes with a reason and timestamp.
