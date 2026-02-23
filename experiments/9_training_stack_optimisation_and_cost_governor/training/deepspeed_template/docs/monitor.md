# Training Monitor Reference

---

## Model: Custom 3B MoE

| Property | Value |
|---|---|
| Total params | ~3.9B |
| Active params | ~1.74B (sparse routing) |
| Layers | 8 total — 6 Gated DeltaNet + 2 Gated Sparse Attention (GSA) |
| FFN | MoE on every layer (20 real + 4 null experts = 24 slots) |
| Routing | Top-2 from 24 slots; target 83% of picks go to null |
| Shared expert | 1 always-active expert per layer |
| Context length | 256k tokens |
| Vocab | 131,072 (Kronecker embeddings) |
| Backprop | ReversibleMidpointStack — memory-efficient, no activation recompute |
| Prediction | Next-token + 2-token-ahead (MTP) |

---

## Metrics We Track

### Loss

| Metric | What it means |
|---|---|
| `loss` | Total loss = NTP + 0.3×MTP + aux. Should decrease over time. |
| `loss_ntp` | Next-token prediction — the primary task loss. |
| `loss_mtp` | Predicting 2 tokens ahead. Should track `loss_ntp`. |
| `loss_aux` | Combined MoE penalty (balance + z-loss + null-rate). Should stay small relative to `loss`. |

---

### MoE Router

| Metric | What it means |
|---|---|
| `moe_null_rate` | Fraction of top-k picks that went to a null slot |
| `moe_avg_real_experts` | Avg real experts hit per token |
| `moe_zero_real_frac` | Fraction of tokens that hit zero real experts (all picks null) | 
| `moe_L_bal` | Load balance penalty — penalises uneven token distribution across the real experts |
| `moe_L_null` | Squared deviation of null rate from target ρ=5/6. Max is 0.694 (null rate = 0). |
| `moe_L_z` | Z-loss — penalises large router logit magnitudes. Keeps softmax well-calibrated. | 
| `moe_expert_counts` | Token count routed to each of the real experts. Shows load imbalance. | 

---

### GPU

| Metric | What it means |
|---|---|
| `gpu_util_all_pct` | SM utilisation per GPU `{"0": %, "1": %, ...}`. All should be 80%+. |
| `gpu_mem_all_gb` | VRAM used per GPU `{"0": GB, ...}`. Should be stable — growth means memory leak. |

---

### Throughput

| Metric | What it means |
|---|---|
| `tokens_per_sec` | Tokens processed per second across all GPUs. Higher is better. |
| `step_time_s` | Wall-clock seconds per optimizer step. First step is slower (JIT warmup). |
| `total_tokens_processed` | Cumulative tokens since training started. |

---

### Training State

| Metric | What it means |
|---|---|
| `global_step` | Optimizer step count. |
| `lr` | Current learning rate from the DeepSpeed scheduler. |
| `cpu_util_pct` | CPU usage. High values mean data loading is a bottleneck. |

---

### Results for 3B MoE Model

experiments/9_training_stack_optimisation_and_cost_governor/training/deepspeed_template/checkpoints_moe_test/metrics.jsonl

https://github.com/The-School-of-AI/LLM/blob/p9/feat/reversibility_metrics/experiments/9_training_stack_optimisation_and_cost_governor/training/deepspeed_template/checkpoints_moe_test/metrics.jsonl

---

### Results for 1B dense Model

experiments/9_training_stack_optimisation_and_cost_governor/training/deepspeed_template/checkpoints_1b_dense/metrics.jsonl

https://github.com/The-School-of-AI/LLM/blob/p9/feat/reversibility_metrics/experiments/9_training_stack_optimisation_and_cost_governor/training/deepspeed_template/checkpoints_1b_dense/metrics.jsonl
