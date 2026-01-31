# Team 11: Growth & Weight Transfer

This directory contains the codebase for model growth experiments (1B → 3B → 8B → 70B trajectory).

## Objective

Demonstrate stable model growth through three phases:
1. **Phase 1**: Train a ~100M dense model for 1000 steps
2. **Phase 2**: Convert to MoE without loss spike, train 1000 more steps
3. **Phase 3**: Scale the model, train 1000 more steps

## Project Structure

```
11_growth_and_weight_transfer/
├── config/
│   └── config.yaml          # Central configuration
├── src/
│   ├── __init__.py
│   ├── model.py              # SmolLM2-style dense transformer
│   ├── moe_model.py          # MoE variant with Top-k routing
│   ├── growth.py             # Growth utilities (dense_to_moe, scale, etc.)
│   └── dataset.py            # Data loaders
├── train.py                  # Single-phase training script
├── run_growth_experiment.py  # Full 3-phase experiment
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install torch pyyaml
# Optional: pip install wandb datasets transformers
```

### 2. Run the Full Experiment

```bash
cd experiments/11_growth_and_weight_transfer
python run_growth_experiment.py
```

This will:
- Train a dense model for 1000 steps
- Convert to MoE and train for 1000 more steps
- Scale the model and train for 1000 more steps
- Save checkpoints and log transitions

### 3. Run with WandB Logging

```bash
python run_growth_experiment.py --wandb
```

### 4. Train Individual Phases

```bash
# Dense model only
python train.py --phase dense --steps 1000

# MoE model from checkpoint
python train.py --phase moe --checkpoint checkpoints/phase1_dense_step_1000.pt
```

## Configuration

Edit `config/config.yaml` to adjust:
- Model architecture (hidden_size, num_layers, etc.)
- MoE settings (num_experts, top_k)
- Training hyperparameters (LR, batch_size, steps)
- Growth methods (add_experts, add_layers, scale_hidden_dim)

## Growth Methods

### Dense → MoE (`dense_to_moe`)
Clones the dense FFN weights into all experts to preserve behavior.

### Add Experts (`add_experts`)
Clones existing experts with small noise to add capacity.

### Add Layers (`add_layers`)
Inserts new transformer blocks with identity-like initialization.

### Scale Hidden Dim (`scale_hidden_dim`)
Pads weight matrices with zeros/Gaussian to increase dimensions.

## Success Criteria

- ✅ Loss decreases smoothly in each phase
- ✅ No loss spike at Dense → MoE transition
- ✅ No loss spike at scaling transition
- ✅ Checkpoints saved at each phase
