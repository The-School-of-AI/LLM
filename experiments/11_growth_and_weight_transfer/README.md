# Team 11: Growth & Weight Transfer

Demonstrates stable model growth through 4 phases without loss spikes.

## Quick Start (Colab)

```python
# Clone and setup
!git clone -b p11/feat/growth-dense-to-moe https://github.com/The-School-of-AI/LLM.git
%cd LLM/experiments/11_growth_and_weight_transfer
!pip install torch pyyaml -q

# Run experiment
!python run_growth_experiment.py
```

## Growth Path (4 Phases)

```
Phase 1: Dense (70M) → Train 1000 steps
    ↓ dense_to_moe (clone FFN → 4 experts)
Phase 2: MoE 4 experts (166M) → Train 1000 steps
    ↓ add_layers + scale_hidden_dim (ghost layers + padding)
Phase 3: MoE (wider, deeper) → Train 1000 steps
    ↓ add_experts (expert explosion: 4 → 8)
Phase 4: MoE 8 experts (final) → Train 1000 steps
```

## Success Criteria

| Transition | Operation | Max Loss Delta |
|------------|-----------|----------------|
| Phase 1→2 | Dense → MoE | < 0.5 |
| Phase 2→3 | +Layers +Dim | < 0.5 |
| Phase 3→4 | ×Experts | < 0.5 |

## Project Structure

```
11_growth_and_weight_transfer/
├── config/config.yaml        # All hyperparameters
├── src/
│   ├── model.py              # Dense transformer (~70M)
│   ├── moe_model.py          # MoE variant
│   ├── growth.py             # Growth utilities
│   └── dataset.py            # TinyShakespeare
├── train.py                  # Single-phase training
└── run_growth_experiment.py  # Full 4-phase experiment
```

## Growth Methods

| Method | What It Does | Why No Spike |
|--------|--------------|--------------|
| `dense_to_moe()` | Clone FFN → all experts | All experts identical |
| `add_layers()` | Insert with zero output proj | Identity function |
| `scale_hidden_dim()` | Pad weights with zeros | Original dims unchanged |
| `add_experts()` | Clone existing expert + noise | New experts ≈ existing |

## Expected Output

```
📌 PHASE 1: Dense Model Training
✓ Created dense model: 70,793,280 parameters
  Step 1000/1000 | Loss: 0.35

📌 PHASE 2: Dense → MoE Conversion
📊 Pre-conversion loss: 0.34
📊 Post-conversion loss: 0.58
✅ Loss delta: +0.24 (STABLE!)

📌 PHASE 3: Add Layers + Scale Hidden Dimension
📊 Pre-scaling loss: 0.XX
📊 Post-scaling loss: 0.XX
✅ Loss delta: +0.XX (STABLE!)

📌 PHASE 4: Expert Explosion
📊 Pre-expert loss: 0.XX
📊 Post-expert loss: 0.XX
✅ Loss delta: +0.XX (STABLE!)

📊 EXPERIMENT SUMMARY
Phase              Final Loss      Transition Δ
Phase 1 (Dense)    0.3433          -
Phase 2 (MoE)      0.XXXX          +0.24
Phase 3 (Scaled)   0.XXXX          +0.XX
Phase 4 (Experts)  0.XXXX          +0.XX
```
