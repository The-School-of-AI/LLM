# Training Stack Configs (Primary + Fallback)

Date: 2026-01-29

## Stack Selection
Primary stack: DeepSpeed (stable, MoE-ready). This is selected for MoE support, ZeRO maturity, and established sharding patterns.
Fallback stack: DeepSpeed with more conservative offload and reduced overlap (same framework, safer settings).

Non-selected options (for reference):
- PyTorch FSDP: mature for dense models but less MoE operational history in this org.
- Custom Megatron-LM: higher risk and integration overhead.

## Common Settings (All Stages)
- Precision: BF16 where supported, FP16 fallback
- Grad accumulation: tuned per hardware to meet token budgets
- Logging: per-step tokens/sec, data-wait time, and loss
- Checkpoint cadence: every 1M tokens and on HALT

## Stage Config Matrix

| Stage | Model Phase | Token Budget | Primary Config | Fallback Config | Notes |
|------:|-------------|--------------|----------------|-----------------|-------|
| 1 | 1B Dense | 20B tokens | ZeRO-2 + selective activation checkpointing OFF | ZeRO-3 + activation checkpointing ON | Keep kernels simple; prioritize stability |
| 2 | 3B MoE-small | 40B tokens | ZeRO-2 + expert parallelism + activation checkpointing ON | ZeRO-3 + activation checkpointing ON | MoE adds comm; avoid aggressive overlap |
| 3 | 8B Dense-deep | 100B tokens | ZeRO-3 + activation checkpointing ON | ZeRO-2 + activation checkpointing ON | Memory pressure drives ZeRO-3 |
| 4 | 70B MoE-large | 240B tokens | ZeRO-3 + activation checkpointing ON + comm overlap tuned | ZeRO-3 + reduced overlap + conservative bucket sizes | Stability first; no speculative tuning |

## DeepSpeed Base Configs
The repo includes base ZeRO configs in:
- `experiments/9_training_stack_optimisation_and_cost_governor/deepspeed-template/config/deepspeed/zero-2.json`
- `experiments/9_training_stack_optimisation_and_cost_governor/deepspeed-template/config/deepspeed/zero-3.json`

Per-stage overrides should be captured in the runbook (batch size, grad accumulation, checkpoint intervals, activation checkpointing flags).

## Activation Checkpointing Strategy
- Stage 1: off by default, enable only if memory constrained
- Stage 2-4: on by default to avoid OOM and to reduce activation memory

## Communication Overlap
- Default: conservative overlap for predictable throughput
- Stage 4: allow overlap only after stability validation

## Sharding Strategy
- Stage 1-2: ZeRO-2 preferred for lower overhead
- Stage 3-4: ZeRO-3 preferred for memory scaling

## Notes for Team 13 Hardware Interface
- Baseline assumption: H100
- If Blackwell is delayed: stay on H100, NVFP4 only in Stage 4 after stability proof
