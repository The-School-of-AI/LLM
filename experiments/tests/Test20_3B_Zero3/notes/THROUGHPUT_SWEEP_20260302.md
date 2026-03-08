# ZeRO3 Throughput Sweep (2026-03-02)

Remote run directory:
- `/mnt/local-nvme/LLM/experiments/tests/Test19_3B_Zero3/results/sweeps/20260302_063724`

Summary (31 steps each, no OOM):

| Case | Config | avg tok/s | min tok/s | max tok/s | avg dt (ms) | OOM | Completed |
|---|---|---:|---:|---:|---:|---:|---:|
| A_baseline_hardened | offload_1024_30steps + hardening on | 531.94 | 288.46 | 563.33 | 15603.87 | 0 | 1 |
| B_baseline_no_hardening | offload_1024_30steps + hardening off | 537.07 | 284.09 | 564.55 | 15471.48 | 0 | 1 |
| C_perf_no_param_offload | perf_1024_30steps + no hardening + no param offload | 554.33 | 286.92 | 569.12 | 14996.47 | 0 | 1 |
| D_perf_with_param_offload | perf_1024_30steps + no hardening + param offload | 543.16 | 284.72 | 559.26 | 15298.46 | 0 | 1 |

First-step GPU reserved memory (from log memory summary):

| Case | GPU reserved (MiB) |
|---|---:|
| A | 9524 |
| B | 9524 |
| C | 8584 |
| D | 10628 |

Key result:
- Best stable setting in this sweep: `C_perf_no_param_offload` (highest throughput, no OOM at 31 steps).
