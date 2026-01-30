# MoE Architecture Tooling Suite

## Overview

Comprehensive runtime support tools for MoE (Mixture of Experts) architecture, designed to support:
- **Training Team**: FLOPs/memory estimation, profiling, debugging
- **Evaluator Team**: Performance metrics, throughput analysis
- **Team 7 (Routing Diagnostics)**: Null expert analysis, dashboard telemetry

---

## 📦 Package Structure

```
moe_tools/
├── __init__.py                 # Package entry point
├── cli.py                      # Command-line interface
├── estimators/
│   ├── param_counter.py        # Parameter counting with breakdown
│   ├── flops_estimator.py      # FLOPs per token/step estimation
│   └── memory_estimator.py     # Memory estimation with distributed training
├── profilers/
│   └── training_profiler.py    # Training stack integration
├── diagnostics/
│   └── routing_diagnostics.py  # MoE routing analysis (Team 7)
└── dashboards/
    └── team7_dashboard.py      # Dashboard configuration for Team 7
```

---

## 🚀 Quick Start

### Installation

```bash
# No external dependencies required for core functionality
# Optional: pip install torch tensorboard wandb pynvml
```

### CLI Usage

```bash
# Run all estimations for a model
python cli.py all 70b_moe --output report.json

# Individual tools
python cli.py estimate 3b_moe
python cli.py estimate 70b_moe --output estimates.json
python cli.py profile --interval 10 --tensorboard
python cli.py diagnose --model 70b_moe --export telemetry.json
python cli.py dashboard --export dashboard_config.json
```

---

## 📊 Tool Descriptions

### 1. Parameter Counter (`param_counter.py`)

Detailed parameter counting by component:

```python
from moe_tools.estimators import ParamCounter, ParamModelConfig

config = ParamModelConfig(
    hidden_size=4096,
    num_layers=40,
    num_attention_heads=32,
    num_kv_heads=8,
    intermediate_size=2048,
    vocab_size=32000,
    num_routed_experts=64,
    num_shared_experts=4,
    top_k=4,
)

counter = ParamCounter(config)
report = counter.full_report()

print(f"Total: {report['summary']['total_params']}")
print(f"Active: {report['summary']['active_params']}")
print(f"Sparsity: {report['summary']['sparsity']}")
```

**Output Breakdown:**
- Embedding parameters
- Attention (Q, K, V, O) per layer
- Router parameters
- Expert parameters (shared vs routed)
- Dual gating (G1 + G2)
- LayerNorm

---

### 2. FLOP Estimator (`flops_estimator.py`)

FLOPs calculation with per-component breakdown:

```python
from moe_tools.estimators import FLOPEstimator, FLOPModelConfig

config = FLOPModelConfig(
    hidden_size=4096,
    num_layers=40,
    num_attention_heads=32,
    intermediate_size=2048,
    num_routed_experts=64,
    top_k=4,
    batch_size=8,
    max_seq_length=4096,
    num_gpus=32,
    gpu_flops_bf16=312e12,  # A100 peak
)

estimator = FLOPEstimator(config)
report = estimator.full_report()

print(f"FLOPs/token (forward): {report['totals']['forward_per_token']}")
print(f"FLOPs/token (total): {report['totals']['total_per_token']}")
print(f"Throughput: {report['throughput']['realistic_tokens_per_sec']} tok/s")
```

**Breakdown Includes:**
- Attention: QKV projection, scores, softmax, output
- Router: query projection, head weights, expert affinity
- Experts: shared, routed, null, gating, combine
- MFU (Model FLOPs Utilization) estimation

---

### 3. Memory Estimator (`memory_estimator.py`)

Memory estimation with distributed training support:

```python
from moe_tools.estimators import MemoryEstimator, MemoryModelConfig, DistributedConfig, ZeROStage

model_config = MemoryModelConfig(
    hidden_size=4096,
    num_layers=40,
    num_routed_experts=64,
    top_k=4,
    batch_size=8,
    max_seq_length=4096,
)

dist_config = DistributedConfig(
    num_gpus=32,
    pipeline_parallel_size=4,
    tensor_parallel_size=1,
    expert_parallel_size=1,
    zero_stage=ZeROStage.PARAMETER,
    activation_checkpointing=True,
    flash_attention=True,
)

estimator = MemoryEstimator(model_config, dist_config)
report = estimator.full_report()

print(f"Per-GPU Memory: {report['distributed']['total_per_gpu']}")
print(f"Recommendations: {report['recommendations']}")
```

**Memory Components:**
- Model weights (bf16)
- Gradients
- Optimizer states (AdamW fp32)
- Activations (with O(S²) attention)
- Distributed sharding (PP, TP, EP, ZeRO)

---

### 4. Training Profiler (`training_profiler.py`)

Integration with training loop for performance monitoring:

```python
from moe_tools.profilers import TrainingProfiler, ProfilerConfig

config = ProfilerConfig(
    profile_every_n_steps=10,
    warmup_steps=5,
    use_tensorboard=True,
    use_wandb=False,
)

profiler = TrainingProfiler(config)

# Training loop
for step in range(num_steps):
    with profiler.profile_step(batch_size=8, seq_length=2048):
        # Time specific regions
        with profiler.time_region('forward'):
            output = model(input_ids)
        
        with profiler.time_region('attention'):
            # Attention computation
            pass
        
        with profiler.time_region('router'):
            # Router computation
            pass
        
        with profiler.time_region('expert_compute'):
            # Expert FFN
            pass
        
        with profiler.time_region('backward'):
            loss.backward()
        
        with profiler.time_region('allreduce'):
            # Gradient sync
            pass
    
    # Log to TensorBoard/W&B
    profiler.log_metrics({'loss': loss.item()})

# Print summary
profiler.print_summary()
profiler.export_profiles('profiles.json')
```

**Metrics Tracked:**
- Step timing breakdown
- GPU utilization
- Memory usage
- Throughput (tokens/second)
- Communication overhead

---

### 5. Routing Diagnostics (`routing_diagnostics.py`)

**Team 7 Core Tool** - MoE routing analysis:

```python
from moe_tools.diagnostics import RoutingDiagnostics, RoutingConfig, create_diagnostics

# Quick setup
diagnostics = create_diagnostics('70b_moe')

# Or custom config
config = RoutingConfig(
    num_routed_experts=64,
    num_shared_experts=4,
    num_null_experts=2,
    top_k=4,
    num_layers=40,
    null_expert_start_idx=64,
    
    # Health gate thresholds
    min_null_junk_rate=0.60,      # Target: 60-80% junk → null
    max_null_junk_rate=0.80,
    max_null_signal_rate=0.10,    # Target: <10% signal → null
    min_routing_entropy=0.70,
    max_gini_coefficient=0.50,
    
    # Junk token IDs
    junk_token_ids=[0, 1, 2, 3],  # PAD, BOS, EOS, UNK
)

diagnostics = RoutingDiagnostics(config)

# In training loop, for each MoE layer:
for layer_idx in range(num_layers):
    diagnostics.log_batch(
        layer_idx=layer_idx,
        expert_indices=expert_indices,    # [batch*seq, top_k]
        expert_weights=expert_weights,    # [batch*seq, top_k]
        token_ids=token_ids,              # [batch*seq]
        token_texts=token_texts,          # Optional
        context_difficulties=difficulties, # Optional: for curriculum buckets
    )

# At end of training step
snapshot = diagnostics.step()

# Get dashboard metrics
metrics = diagnostics.get_dashboard_metrics()
print(f"Junk → Null: {metrics['null_expert']['junk_to_null_rate']}")
print(f"Signal → Null: {metrics['null_expert']['signal_to_null_rate']}")
print(f"Compute Savings: {metrics['null_expert']['compute_savings_pct']}")
print(f"LoRA Ready: {metrics['stability']['lora_ready']}")

# Check growth trigger
if metrics['growth_trigger']['recommend_growth']:
    print("Ready for expert expansion!")

# Export telemetry
diagnostics.export_telemetry('routing_telemetry.json')
```

**Team 7 Metrics Provided:**

| Metric | Target | Description |
|--------|--------|-------------|
| `junk_to_null_rate` | 60-80% | Junk tokens → null expert |
| `boilerplate_to_null_rate` | 40-70% | Boilerplate → null |
| `signal_to_null_rate` | <10% | Signal tokens leaked to null |
| `compute_savings_pct` | >10% | FLOPs saved by null routing |
| `routing_entropy` | >0.70 | Selection diversity (1.0 = uniform) |
| `gini_coefficient` | <0.50 | Load balance (0 = perfect) |
| `stability_score` | >0.80 | Routing stability over time |
| `lora_ready` | boolean | MoE block stable for LoRA |
| `growth_ready` | boolean | Ready for expert expansion |

---

### 6. Team 7 Dashboard (`team7_dashboard.py`)

Dashboard configuration and live metrics:

```python
from moe_tools.dashboards import Team7Dashboard

# Create dashboard (optionally connected to diagnostics)
dashboard = Team7Dashboard(diagnostics=diagnostics)

# Get live metrics for display
metrics = dashboard.get_live_metrics()

# Get panel configuration for UI
panels = dashboard.get_panel_config()

# Get alert rules
alerts = dashboard.get_alert_rules()

# Export full dashboard config (for Grafana/custom UI)
dashboard.export_dashboard_config('team7_dashboard.json')

# Print summary
dashboard.print_summary()
```

**Dashboard Panels:**

| Panel | Type | Metrics |
|-------|------|---------|
| Junk → Null Rate | Gauge | Target: 60-80% |
| Signal Leakage | Gauge | Target: <10% |
| Routing Entropy | Gauge | Target: >0.70 |
| System Status | Status | LoRA Ready, Growth Ready |
| Null Rates Trend | Line Chart | Time series |
| Health Trend | Line Chart | Entropy, Gini |
| Expert Utilization | Bar Chart | Per-expert distribution |
| Curriculum Routing | Heatmap | B0-B5 → Expert |
| Compute Savings | Gauge | % FLOPs saved |
| Active Alerts | Alert List | Health gate violations |

**Alert Rules:**

| Alert | Severity | Condition |
|-------|----------|-----------|
| Low Null Routing | WARNING | junk_to_null < 50% |
| High Signal Leakage | WARNING | signal_to_null > 15% |
| Entropy Collapse | CRITICAL | entropy < 0.50 |
| Load Imbalance | WARNING | gini > 0.50 |
| Dead Experts | WARNING | any expert < 1% util |

---

## 🔧 Integration Examples

### Training Loop Integration

```python
from moe_tools import (
    FLOPEstimator, MemoryEstimator, 
    TrainingProfiler, RoutingDiagnostics,
    Team7Dashboard
)

# Setup
flop_est = FLOPEstimator(flop_config)
mem_est = MemoryEstimator(model_config, dist_config)
profiler = TrainingProfiler(profiler_config)
diagnostics = RoutingDiagnostics(routing_config)
dashboard = Team7Dashboard(diagnostics)

# Pre-training validation
print(f"Estimated FLOPs/token: {flop_est.full_report()['totals']['forward_per_token']}")
print(f"Per-GPU Memory: {mem_est.full_report()['distributed']['total_per_gpu']}")

# Training loop
for step in range(num_steps):
    with profiler.profile_step(batch_size, seq_length):
        # Forward pass with routing logging
        for layer_idx, layer in enumerate(model.layers):
            output, expert_indices, expert_weights = layer(hidden_states)
            
            diagnostics.log_batch(
                layer_idx=layer_idx,
                expert_indices=expert_indices.tolist(),
                expert_weights=expert_weights.tolist(),
                token_ids=input_ids.flatten().tolist(),
            )
        
        # Backward
        loss.backward()
        optimizer.step()
    
    # End of step
    snapshot = diagnostics.step()
    profiler.log_metrics({
        'loss': loss.item(),
        'null_junk_rate': snapshot.null_junk_rate,
        'entropy': snapshot.avg_routing_entropy,
    })
    
    # Check health gates
    if step % 100 == 0:
        metrics = dashboard.get_live_metrics()
        if not metrics['all_healthy']:
            print(f"⚠️ Health gate failures at step {step}")
            for alert in metrics['alerts']:
                print(f"  [{alert['severity']}] {alert['message']}")
    
    # Check growth trigger
    if step % 1000 == 0:
        growth = metrics['growth_trigger']
        if growth['recommend_growth']:
            print(f"✅ Expert growth recommended (confidence: {growth['confidence']:.2f})")
```

### Curriculum Bucket Tracking (B0-B5)

```python
from moe_tools.diagnostics import CurriculumBucket

# When logging batches, include difficulty scores
diagnostics.log_batch(
    layer_idx=layer_idx,
    expert_indices=expert_indices,
    expert_weights=expert_weights,
    token_ids=token_ids,
    context_difficulties=batch_difficulties,  # 0.0-1.0 scores
)

# Get bucket → expert mapping
bucket_map = diagnostics.get_bucket_expert_map()

for bucket_name, data in bucket_map.items():
    print(f"{bucket_name}:")
    print(f"  Total tokens: {data['total_tokens']}")
    print(f"  Top experts: {data['top_experts']}")
    print(f"  Percentages: {data['percentages']}")
```

---

## 📋 Team 7 Dashboard API

### Health Gates

```python
# Check all gates
gates = snapshot.health_gates
all_pass = all(gates.values())

# Individual gates
gates['null_junk_min']      # Junk → null >= 60%
gates['null_junk_max']      # Junk → null <= 80%
gates['null_signal_max']    # Signal → null <= 10%
gates['entropy_min']        # Entropy >= 0.70
gates['gini_max']           # Gini <= 0.50
gates['no_dead_experts']    # No experts < 1% util
gates['no_overloaded']      # No experts > 3x expected
```

### Stability Milestones

```python
# LoRA readiness
if snapshot.is_stable and all(snapshot.health_gates.values()):
    print("MoE block is LoRA-ready!")

# Expert growth trigger
if diagnostics.get_dashboard_metrics()['growth_trigger']['recommend_growth']:
    print("Ready for expert expansion (8 → 64)")
```

### Loss-Free Routing Control

```python
# Get recommendations (Team 7 can inject these)
if snapshot.null_junk_rate < 0.50:
    # Recommendation: Increase null expert bias
    adjustment = {
        'action': 'increase_null_bias',
        'target_experts': [64, 65],  # Null expert indices
        'adjustment': 0.1,
        'confidence': 'high' if snapshot.null_junk_rate < 0.40 else 'medium'
    }

# If conclusions are ambiguous, refuse to change
if snapshot.stability_score < 0.60:
    adjustment = {
        'action': 'none',
        'reason': 'Routing not stable enough for intervention',
        'recommendation': 'Continue monitoring'
    }
```

---

## 📈 Output Examples

### Parameter Count Report

```
📦 PARAMETER COUNT
----------------------------------------
  Total Parameters: 70.5B
  Active Parameters: 12.2B
  Sparsity: 82.7%

Per-Component:
  Embeddings: 131.07M (0.2%)
  Attention: 13.42B (19.0%)
  Router: 1.34B (1.9%)
  Shared Experts: 5.63B (8.0%)
  Routed Experts: 45.05B (63.9%)
  Gating: 4.13B (5.9%)
  LayerNorm: 1.31M (0.0%)
```

### Memory Report

```
💾 MEMORY ESTIMATION
----------------------------------------
Per-GPU (32 GPUs, ZeRO-3, PP=4):
  Weights: 4.38 GB
  Gradients: 4.38 GB
  Optimizer: 17.50 GB
  Activations: 28.20 GB
  Total: 54.46 GB ✅ (fits A100-80GB)

Recommendations:
  ✅ Memory usage looks healthy for A100-80GB
```

### Dashboard Metrics

```
🎯 Null Expert Metrics:
  Junk → Null: 68.5% [green]
  Signal → Null: 6.2% [green]
  Compute Savings: 14.2%

📊 Routing Health:
  Entropy: 0.87 [green]
  Gini (Balance): 0.12 [green]

✅ Status:
  LoRA Ready: ✓ Ready
  Growth Ready: ✗ Not Ready (stability_score < 0.80)
  All Gates Pass: ✓
```

---

## 🎯 Team 7 Integration Checklist

- [ ] Connect `routing_diagnostics.py` to training loop
- [ ] Log all MoE layer routing decisions
- [ ] Configure junk token IDs from tokenizer (Team 6)
- [ ] Set up dashboard export for monitoring
- [ ] Implement alert handlers for health gate failures
- [ ] Track curriculum bucket progression (B0-B5)
- [ ] Implement loss-free routing control injection (optional)
- [ ] Monitor stability for LoRA-readiness milestones
- [ ] Track expert growth triggers

---

## 📝 Version History

- **v1.0.0** (January 2026)
  - Initial release
  - FLOP/Memory/Parameter estimators
  - Training profiler with region timing
  - Team 7 routing diagnostics
  - Dashboard configuration

---

## 🤝 Team Dependencies

| Team | Dependency | Status |
|------|------------|--------|
| Team 6 (Tokenizer) | Junk token ID list | Required |
| Team 7 (Telemetry) | Dashboard integration | This package |
| Team 10 (Training) | Profiler hooks | Ready |
| Infra | TensorBoard/W&B | Optional |
