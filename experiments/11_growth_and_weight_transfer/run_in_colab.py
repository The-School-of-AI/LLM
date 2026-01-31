# Growth Experiment - Colab Notebook
# ==================================
# Run this in Google Colab with a T4 GPU

# %% [markdown]
# # 🧪 Growth Experiment: Dense → MoE → Scale
# 
# This notebook demonstrates stable model growth through 3 phases:
# 1. Train a ~70M dense model for 1000 steps
# 2. Convert to MoE, train for 1000 more steps
# 3. Scale the model, train for 1000 more steps

# %% Cell 1: Setup
!pip install torch pyyaml wandb -q

# Clone the repo (or upload files)
!git clone https://github.com/The-School-of-AI/LLM.git
%cd LLM/experiments/11_growth_and_weight_transfer

# %% Cell 2: Check GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# %% Cell 3: Quick test - verify model loads
from src.model import SmolLM2
model = SmolLM2()
print(f"✅ Dense model: {model.num_parameters():,} parameters")

# %% Cell 4: Run the full experiment
!python run_growth_experiment.py

# %% [markdown]
# ## Expected Output
# 
# You should see:
# - Phase 1: Dense training, loss decreasing
# - Transition 1: "Pre-conversion loss: X.XX, Post-conversion loss: Y.YY" (should be close!)
# - Phase 2: MoE training, loss continuing to decrease
# - Transition 2: Same pattern
# - Phase 3: Scaled model training
# 
# **Success = No loss spikes at transitions!**
