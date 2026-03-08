# Optim Step Recovery (2026-03-02)

Objective:
- Keep same 1024 / 20-step profile setup.
- Target `optim_step` latency first.

Change tested:
- Removed ZeRO-3 `offload_optimizer` (CPU optimizer offload) while keeping ZeRO-3.
- Config used: `deepspeed/zero-3-moe-bf16-perf-gpuopt.json`
- Run config: `configs/test17_3b_moe_perf_1024_20steps_profile_gpuopt.yaml`

Run status:
- Completed: yes
- OOM: no
- ZeRO stage: 3

Throughput:
- all steps avg tok/s: 3653.51
- step>=2 avg tok/s: 3750.24
- all steps avg dt: 2310.53 ms
- step>=2 avg dt: 2187.74 ms

Profiler comparison (same profile steps 2/10/18):

| Phase | Before (CPU optimizer offload) | After (GPU optimizer) |
|---|---:|---:|
| forward | 444.22 ms | 424.32 ms |
| backward | 2675.73 ms | 1749.86 ms |
| optim_step | 11840.94 ms | 123.48 ms |
| step_total | 15005.46 ms | 2365.64 ms |

Memory:
- First-step GPU reserved memory (from log): `11546 MiB` (fits 40GB GPUs).

Conclusion:
- The dominant `optim_step` bottleneck was CPU optimizer offload in ZeRO-3.
- Removing optimizer offload recovered a large portion of performance while remaining stable at seq=1024.
