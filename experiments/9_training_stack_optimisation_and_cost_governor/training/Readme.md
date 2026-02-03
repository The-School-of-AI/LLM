# Team 9: Training Stack Optimization & Cost Governor

## Team Brief

**Category:** Infrastructure / Training Systems

We are the **cost governance and execution efficiency** team. We ensure training runs stay within budget while maximizing GPU utilization. We own the training stack, not the model design.

**Mission:** Minimize dollar-per-token across all training stages with automatic spend enforcement and zero GPU starvation.

---

## Core Responsibilities

### 1. Cost Governance (Primary)
- Track **$ per 1M tokens** as the canonical metric
- Implement automatic HALT system when budget/stability thresholds are violated
- No dollar spent without visibility and guardrails

### 2. Training Stack Optimization
- Select and configure **DeepSpeed** (primary stack)
- Benchmark ZeRO-2 vs ZeRO-3, activation checkpointing
- Eliminate GPU data starvation
- Attribute all throughput bottlenecks to named causes

### 3. Monitoring & Attribution
- Profile tokens/sec per GPU, FLOPs utilization, sync waste
- Build diagnostics for data pipeline, communication, and kernel performance
- Maintain rolling cost windows at 1M-token granularity

---

## Training Stages

Optimize **$/token** for all four stages within **hard budget ceilings**:

| Stage   | Model Phase       | Token Budget    |
| ------- | ----------------- | --------------- |
| Stage 1 | 1B Dense          | 20B tokens      |
| Stage 2 | 3B MoE-small      | 40B tokens      |
| Stage 3 | 8B Dense-deep     | 100B tokens     |
| Stage 4 | 70B MoE-large     | 240B tokens     |

---

## Key Experiments

### Experiment 1: DeepSpeed Configuration (ZeRO-2 vs ZeRO-3)
- Benchmark sharding strategies for MoE models
- Evaluate activation checkpointing and communication overlap
- Deliver one primary config + one fallback per stage

### Experiment 2: Sparse Attention Cost Modeling
- Collaborate with Team 8 on **DeepSeek Sparse Attention**
- Quantify FLOP savings, memory relief, stability boundaries
- Provide recommended sparsity targets for later stages

### Experiment 3: NVFP4 Pipeline Design
- After core optimization, design training stack for **NVFP4 pipelines**
- Cross-team integration for Stage 4 (70B MoE)
- Coordinate with Team 13 for Blackwell/H100 availability

### Experiment 4: Automatic HALT System
- Build fail-safe controller that saves checkpoints and shuts down GPUs
- Triggers: NaNs, throughput collapse, GPU starvation, cost overruns
- No human approval required for shutdown

---

## Deliverables

1. **Training Configs:** Primary + fallback per stage
2. **Cost Tables:** $ / 1M tokens (expected vs actual) per stage
3. **Throughput Report:** Bottlenecks, waste attribution, mitigation decisions
4. **HALT System:** Scripts, trigger definitions, shutdown verification
5. **Failure Logs:** Known failure modes and recovery procedures

---

## What We Control

- Training framework selection and configuration
- Sharding, checkpointing, data-loader performance
- Runtime profiling and cost accounting
- Automatic HALT mechanisms

## What We DON'T Control

- Model architecture (Team 8)
- Data mix or curriculum (Teams 1–4)
- Tokenizer behavior (Teams 6–7)
- Evaluation benchmarks

---

## Success Criteria

- All stages complete **within token budget**
- GPUs maintain high utilization (no data starvation)
- HALT system successfully prevents runaway spend
- Cost signals trusted without revalidation

---

## Timeline

- **Jan 29–30:** Stack comparison, instrumentation, budget modeling
- **Jan 30–Feb 1:** Benchmark configs, lock primary/fallback, validate HALT
- **Feb 1–7:** Live monitoring, drift detection, enforcement
- **Feb 7:** Stack locked, cost controls active

---

## Dependencies

- **Team 8:** Frozen architecture
- **Team 13:** Hardware availability and pricing
- **Team 6:** Tokenizer constraints

**Blocks:** Teams 10 and 12

---

## Tools

DeepSpeed, PyTorch, GPU/communication/memory profilers, custom cost-tracking scripts, automated checkpoint controllers
