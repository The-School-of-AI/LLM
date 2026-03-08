# 20-Step Profile Breakdown (2026-03-02)

Run:
- Log: `/mnt/local-nvme/LLM/experiments/tests/Test19_3B_Zero3/results/run/t19_perf_1024_20steps_profile.log`
- Config: `configs/test17_3b_moe_perf_1024_20steps_profile.yaml`
- DS config: `deepspeed/zero-3-moe-bf16-perf-noparamoffload.json`
- Hardening toggles: all major cleanup hooks disabled for perf (`T19_STEP_*`, `T19_ZERO3_RELEASE_EVERY=0`, etc.)

Result:
- Steps: 20
- OOM: no
- avg tok/s: 545.02
- avg dt: 15387.48 ms

Step-profiler phase means (3 profiled steps):
- dataloader: 0.13 ms
- forward: 444.22 ms
- fused_ce: 18.85 ms
- fused_ce_mtp: 18.81 ms
- backward: 2675.73 ms
- optim_step: 11840.94 ms

Interpretation:
- `optim_step` is the dominant cost by far (~11.84s of ~15.39s step time).
- Forward kernels are not the primary bottleneck in this setup.
- Main throughput limiter is ZeRO-3 optimizer/communication/offload step behavior.
