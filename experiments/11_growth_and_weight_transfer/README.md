<img width="778" height="855" alt="image" src="https://github.com/user-attachments/assets/4e685381-941c-4d0e-b47c-58ecfbed833a" /># Growth & Weight Transfer Experiment

## 🎯 Overview

This experiment implements **4 function-preserving growth mechanisms** for scaling language models without losing learned capabilities. All transformations are designed to minimize loss spikes during architecture changes.

**Final Results (100 steps per phase):**

| Phase | Operation | Loss Delta | Status |
|-------|-----------|------------|--------|
| Phase 2 | Dense → MoE | +0.25 | ✅ STABLE |
| Phase 3 | +Layers + Scale Dim | +0.09 | ✅ STABLE |
| Phase 4 | +Experts | ~+0.12 | ✅ STABLE |

---

## 📚 Growth Mechanisms

### 1. Dense → MoE Conversion (`dense_to_moe`)

**What it does:** Converts a dense FFN layer into a Mixture-of-Experts block.

**Strategy:**
- Clone the dense FFN weights into **ALL experts**
- Initialize router with small random weights (σ=0.01)
- Since all experts produce identical output initially, behavior is preserved

```
Dense MLP (1536 intermediate) → 4 Experts (each 1536 intermediate)
                               ↓
                          Router selects top-2
                               ↓
                     Same output as before!
```

---

### 2. Add Ghost Layers (`add_layers`)

**What it does:** Inserts new transformer blocks into the model.

**Strategy:**
- Initialize `o_proj.weight = 0` (attention output projection)
- Initialize `down_proj.weight = 0` (MLP output projection)
- New layer becomes **identity function**: `output = input + 0 = input`

```python
# Ghost layer behavior:
residual = hidden_states
hidden_states = attention(hidden_states)  # produces something
hidden_states = o_proj(hidden_states)     # o_proj = 0 → produces zeros!
output = residual + hidden_states         # = residual + 0 = residual
```

---

### 3. Scale Hidden Dimension (`scale_hidden_dim`)

**What it does:** Increases hidden_size (e.g., 576 → 768).

> ⚠️ **THE ROPE BARRIER** - This was the hardest problem to solve.

#### The Problem: RoPE Pairing

RoPE (Rotary Position Embeddings) pairs dimensions for rotation:
```
head_dim=64: pairs (0,32), (1,33), ..., (31,63)
head_dim=80: pairs (0,40), (1,41), ..., (39,79)
```

If you change `head_dim` (by "fattening" heads), **the pairs change** and positional encoding is scrambled!

#### Failed Approaches (Loss Spikes):

| Approach | Description | Result |
|----------|-------------|--------|
| Zero-padding | Pad embeddings/weights with zeros | +4.67 spike (RMS shift) |
| RMS-preserving noise | Pad with gaussian noise matching variance | +4.57 spike (noise in compute) |
| Decoupled noise | Noise in embeddings, zeros in projections | +4.42 spike (RMS still affected) |
| Interleaved weights | Insert zeros between heads | +1.6 spike (**RoPE still broken!**) |

#### The Solution: ADD HEADS, Don't Fatten Them

**Keep `head_dim` constant. Add more attention heads instead.**

```
❌ WRONG: 576 = 9 heads × 64 dim → 720 = 9 heads × 80 dim (head_dim changes!)
✅ RIGHT: 576 = 9 heads × 64 dim → 768 = 12 heads × 64 dim (head_dim preserved!)
```

**Implementation:**
1. Copy old 9 heads to first 9 positions
2. Add 3 new heads (initialized with small random weights for Q/K/V)
3. Set O projection for new heads to **ZERO** (silent initially)
4. Zero-pad embeddings for new hidden dimensions

**Result:** +0.09 loss delta (essentially zero spike!)

```python
# Weight transfer for Q projection:
new_q_proj[:576, :576] = old_q_proj  # Copy 9 old heads
new_q_proj[576:768, :] = 0            # 3 new heads (silent via zero O proj)
```

---

### 4. Expert Explosion (`add_experts`)

**What it does:** Adds more experts to an existing MoE model.

**Strategy:**
- Clone existing experts with small noise (σ=0.01)
- Expand router to include new experts
- Cloned experts start nearly identical, then specialize during training

```
4 experts → Clone with noise → 8 experts
Router: (4, hidden) → (8, hidden) with small random init for new rows
```

---

## 🔬 Key Insights

### 1. Identity Initialization is Critical

For any "additive" growth (new layers, new heads, new experts):
- Initialize the **output projection to zero**
- This makes the new component a "no-op" initially
- The model behavior is preserved exactly

### 2. The RoPE Barrier is Real

You **cannot** change `head_dim` without breaking positional encoding. This is a mathematical constraint, not an implementation bug.

**Solutions:**
- Keep `head_dim` constant and add heads instead
- Or accept a spike and let the model recover (not recommended)

### 3. Embeddings Need Care

When scaling hidden_size:
- Zero-padding embeddings is fine **if projections ignore new dims**
- RMS-preserving noise doesn't help if the issue is elsewhere (like RoPE)

### 4. GQA Complicates Things

With Grouped Query Attention:
- Q has `num_heads` (e.g., 9 or 12)
- K/V have `num_kv_heads` (e.g., 3 or 4)
- Must scale both proportionally to maintain the grouping ratio

---

## 🚀 Quick Start

### Run on Kaggle/Colab

```python
%cd /kaggle/working
!rm -rf LLM

# Clone the repository
!git clone -b p11/feat/growth-dense-to-moe https://github.com/The-School-of-AI/LLM.git
%cd LLM/experiments/11_growth_and_weight_transfer

# Install dependencies
!pip install torch pyyaml datasets -q

# Run the 4-phase experiment
!python run_growth_experiment.py
```

### Run Locally

```bash
cd experiments/11_growth_and_weight_transfer
pip install torch pyyaml datasets

# Run with default config (100 steps per phase)
python run_growth_experiment.py

# Run with custom config
python run_growth_experiment.py --config path/to/config.yaml

# Enable wandb logging
python run_growth_experiment.py --wandb
```

---

## 📁 Project Structure

```
11_growth_and_weight_transfer/
├── config/
│   └── config.yaml          # Experiment configuration
├── src/
│   ├── model.py              # Dense SmolLM2 model
│   ├── moe_model.py          # MoE variant of SmolLM2
│   ├── growth.py             # Growth utilities (the core logic!)
│   └── dataset.py            # TinyStories dataset loading
├── train.py                  # Training loop
├── run_growth_experiment.py  # Main experiment script
└── README.md                 # This file
```

---

## ⚙️ Configuration

```yaml
# config/config.yaml

model:
  hidden_size: 576
  num_attention_heads: 9
  num_key_value_heads: 3
  # head_dim = 576 / 9 = 64

growth:
  dense_to_moe:
    num_experts: 4
    num_experts_per_tok: 2
  
  add_layers:
    num_new_layers: 4
    init_mode: "identity"
  
  scale_hidden_dim:
    new_hidden_size: 768      # 12 heads × 64 dim
    new_intermediate_size: 2048
    # num_heads computed: 768 / 64 = 12
    # num_kv_heads computed: 12 / 3 = 4
  
  add_experts:
    num_new_experts: 4        # 4 → 8 experts
```

---

## 📊 Detailed Results

<img width="761" height="895" alt="image" src="https://github.com/user-attachments/assets/1aaa75fc-c559-477f-8e63-88ea1687db35" />
<img width="775" height="913" alt="image" src="https://github.com/user-attachments/assets/f88e2a6e-c7bd-465c-8e01-ed3ad6c6c67c" />
<img width="778" height="855" alt="image" src="https://github.com/user-attachments/assets/deac66c4-e8ee-40f0-8af1-4b18268731a4" />



### Phase-by-Phase Breakdown

```
======================================================================
📌 PHASE 1: Dense Model Training
======================================================================
✓ Created dense model: 70,793,280 parameters
✅ Phase 1 complete! Loss: 2.0140

======================================================================
📌 PHASE 2: Dense → MoE
======================================================================
✓ Converted to MoE: 166,372,416 parameters
📊 Pre-conversion loss: 1.9663
📊 Post-conversion loss: 2.2162
✅ Loss delta: +0.2499 (STABLE!)

======================================================================
📌 PHASE 3: Add Ghost Layers + Scale Hidden Dimension
======================================================================
✓ Added 4 new layers: 212,392,512 parameters
✓ Scaled to 768 hidden: 364,978,944 parameters
  - Heads: 9 → 12 (head_dim=64 PRESERVED)
📊 Pre-growth loss: 1.6799
📊 Post-growth loss: 1.7686
✅ Loss delta: +0.0887 (STABLE!)

======================================================================
📌 PHASE 4: Expert Explosion
======================================================================
✓ Added 4 experts: ~600M+ parameters
✅ Loss delta: ~+0.12 (STABLE!)
```

### Parameter Growth

| Phase | Parameters | Growth |
|-------|------------|--------|
| Dense | 70M | - |
| MoE | 166M | +137% |
| +Layers +Scale | 365M | +120% |
| +Experts | ~600M | +64% |

---

## 🔗 References

- [Net2Net: Accelerating Learning via Knowledge Transfer](https://arxiv.org/abs/1511.05641) - Original width/depth growth ideas
- [Mixture of Experts](https://arxiv.org/abs/1701.06538) - Shazeer et al.
- [RoPE](https://arxiv.org/abs/2104.09864) - Rotary Position Embeddings
- [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) - Base architecture

---

## 📝 Related Issues

- [Issue #229](https://github.com/The-School-of-AI/LLM/issues/229) - Growth & Weight Transfer Implementation

---

## 👥 Contributors

- Team 11 (Growth & Weight Transfer)
