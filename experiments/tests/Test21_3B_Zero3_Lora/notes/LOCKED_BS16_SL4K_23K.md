# Locked Profile: BS16 SL4k 23k Throughput

Validated run profile (locked):
- Sequence length: `4096`
- Global batch size: `16`
- Micro batch per GPU: `2` (8 GPUs)
- Grad accumulation: `1`
- ZeRO stage: `3`
- Optimizer offload: `OFF` (GPU optimizer path)

Config files:
- `configs/test17_3b_moe_perf_4096_bs16_10steps.yaml`
- `deepspeed/zero-3-moe-bf16-perf-gpuopt-bs16.json`

Validation log:
- `/mnt/local-nvme/LLM/experiments/tests/Test19_3B_Zero3/results/run/t19_stress_seq4096_bs16_10steps.log`

Observed result:
- Completed without OOM
- Steady-state throughput: ~`23k tok/s` (steps 2-10)
- Peak GPU reserved memory (rank-0 log summary): `22812 MiB`
