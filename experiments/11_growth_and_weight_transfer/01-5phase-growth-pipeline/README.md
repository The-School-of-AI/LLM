# Growth & Weight Transfer Experiment

## 🎯 Overview

This experiment implements **5 function-preserving growth mechanisms** for scaling language models without losing learned capabilities. All transformations are designed to minimize loss spikes during architecture changes.

**Final Results (100 steps per phase, 500 total):**

| Phase | Operation | Final Loss | Loss Delta | Status |
|-------|-----------|------------|------------|--------|
| Phase 1 | Dense Training | 2.0140 | - | ✅ |
| Phase 2 | Dense → MoE | 1.7076 | +0.2500 | ✅ STABLE |
| Phase 3 | +Layers + Scale Dim | 1.5228 | +0.0667 | ✅ STABLE |
| Phase 4 | +Experts (4→8) | 1.4236 | +0.1266 | ✅ STABLE |
| Phase 5 | YaRN Context (256→1024) | 1.2878 | +0.3706 | ✅ STABLE |

**Final Model:** 667M params, 1024 context length

---

## 📚 The 5 Growth Mechanisms

### Phase 1: Dense Model Training
Train a base SmolLM2-style model (~70M params).

### Phase 2: Dense → MoE Conversion
- Clone dense FFN weights into ALL experts
- Initialize router with small random weights
- All experts produce identical output initially → function preserved

### Phase 3: Add Ghost Layers + Scale Hidden Dimension

**Ghost Layers:**
- Initialize `o_proj = 0` and `down_proj = 0`
- New layer becomes identity: `output = input + 0`

**Scale Hidden Dimension (THE ROPE BARRIER):**
> ⚠️ **Critical Discovery:** Changing `head_dim` breaks RoPE positional encoding!

**Failed approaches:**
| Approach | Result |
|----------|--------|
| Zero-padding | +4.67 spike |
| RMS-preserving noise | +4.57 spike |
| Interleaved weights | +1.6 spike |

**Solution: ADD HEADS, Don't Fatten Them**
```
❌ WRONG: 576 = 9 heads × 64 dim → 720 = 9 heads × 80 dim (head_dim changes!)
✅ RIGHT: 576 = 9 heads × 64 dim → 768 = 12 heads × 64 dim (head_dim preserved!)
```

### Phase 4: Expert Explosion
- Clone existing experts with noise (σ=0.01)
- Expand router for new experts
- New experts start similar, then specialize

### Phase 5: YaRN Context Extension (256 → 1024)

**YaRN (Yet another RoPE extensioN)** extends context without retraining from scratch:

1. **NTK-by-parts Interpolation**: Preserve high-freq dims (local info), interpolate low-freq dims (global info)
2. **Attention Scaling**: `√(1/t) = 0.1 × ln(scale) + 1`

**The Sandwich Protocol (3-Step Verification):**

| Step | What | Why | Success |
|------|------|-----|---------|
| **Step 1** | Measure loss at 256 context | "Do No Harm" - YaRN didn't break existing behavior | Δ < 0.3 |
| **Step 2** | Train on 1024 context | Learn new positions | Loss drops |
| **Step 3** | Measure loss at 1024 context | "Capability Check" - can now handle long context | Gain > 0.5 |

---

## 🚀 Quick Start

### Run on Kaggle/Colab

```python
%cd /kaggle/working
!rm -rf LLM

# Clone the repo
!git clone -b p11/feat/5phase-growth-pipeline https://github.com/The-School-of-AI/LLM.git
%cd LLM/experiments/11_growth_and_weight_transfer/01-5phase-growth-pipeline

# Install dependencies
!pip install torch pyyaml datasets -q

# Run 5-phase experiment (100 steps each)
!python run_growth_experiment.py
```

### Run on AWS (g5.xlarge recommended)

```bash
# Clone
git clone -b p11/feat/5phase-growth-pipeline https://github.com/The-School-of-AI/LLM.git
cd LLM/experiments/11_growth_and_weight_transfer/01-5phase-growth-pipeline

# Setup
python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install pyyaml datasets

# Run
python run_growth_experiment.py
```

### Run with 1000 Steps

```bash
sed -i 's/phase1_steps: 100/phase1_steps: 1000/g' config/config.yaml
sed -i 's/phase2_steps: 100/phase2_steps: 1000/g' config/config.yaml
sed -i 's/phase3_steps: 100/phase3_steps: 1000/g' config/config.yaml
sed -i 's/phase4_steps: 100/phase4_steps: 1000/g' config/config.yaml
sed -i 's/phase5_steps: 100/phase5_steps: 1000/g' config/config.yaml
python run_growth_experiment.py
```

### Resume Training

If training is interrupted (SSH disconnect, crash, etc.), you can resume from the last checkpoint:

```bash
# Auto-resume: detects latest checkpoint and continues
python run_growth_experiment.py --resume

# Manual: resume from a specific phase (loads checkpoint from end of previous phase)
python run_growth_experiment.py --resume-phase 3  # Resumes from Phase 3
```

**How it works:**
- Checkpoints save both model weights and `samples_seen` count
- On resume, the dataloader skips previously seen samples
- Training continues from exactly where it left off
- **Metrics saved to:** `checkpoints/experiment_results.json`
- **Logs saved to:** `checkpoints/experiment.log`

**Pro tip:** Use `screen` or `tmux` for long training runs:
```bash
screen -S training
python run_growth_experiment.py --resume
# Ctrl+A, D to detach | screen -r training to reattach
```

**Test the dataloader state:**
```bash
python test_stateful_dataloader.py
```


---

## 📁 Project Structure

```
11_growth_and_weight_transfer/
├── config/
│   └── config.yaml          # Experiment configuration
├── src/
│   ├── model.py              # Dense SmolLM2 model
│   ├── moe_model.py          # MoE variant
│   ├── growth.py             # All growth operations
│   ├── yarn.py               # YaRN RoPE implementation
│   └── dataset.py            # Dataset loaders (dummy, tinystories, any HF)
├── train.py                  # Training loop
├── run_growth_experiment.py  # Main 5-phase experiment
└── README.md
```

---

## ⚙️ Configuration

```yaml
model:
  hidden_size: 576          # 9 heads × 64 dim
  num_attention_heads: 9
  max_position_embeddings: 256

growth:
  scale_hidden_dim:
    new_hidden_size: 768    # 12 heads × 64 dim (head_dim preserved!)
  
  scale_context:
    new_max_length: 1024    # 4x extension
    alpha: 1.0              # NTK-by-parts params
    beta: 32.0

training:
  phase1_steps: 100
  phase2_steps: 100
  phase3_steps: 100
  phase4_steps: 100
  phase5_steps: 100
```

---

## 📊 Actual Output

```
======================================================================
📊 EXPERIMENT SUMMARY
======================================================================
Total steps trained: 500

Phase                          Final Loss      Transition Δ   
------------------------------------------------------------
Phase 1 (Dense)                2.0140          -              
Phase 2 (MoE)                  1.7076          +0.2500
Phase 3 (Layers+Scale)         1.5228          +0.0667
Phase 4 (×Experts)             1.4236          +0.1266
Phase 5 (YaRN Context)         1.2878          +0.3706

Final model parameters: 667,017,984
Final context length: 1024
======================================================================

🎉 SUCCESS: All transitions were STABLE (delta < 0.5)
```

---

## 🔗 References

- [YaRN Paper](https://arxiv.org/abs/2309.00071) - Context window extension
- [Net2Net](https://arxiv.org/abs/1511.05641) - Width/depth growth
- [RoPE](https://arxiv.org/abs/2104.09864) - Rotary Position Embeddings
- [Issue #229](https://github.com/The-School-of-AI/LLM/issues/229) - This implementation

---

## 👥 Team 11: Growth & Weight Transfer
