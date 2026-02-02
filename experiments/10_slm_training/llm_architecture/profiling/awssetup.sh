#!/bin/bash
# AWS EC2 Setup Script for LLM Training

# Update system and install dependencies
sudo apt update && sudo apt -y install python3-venv python3-pip git
sudo snap install astral-uv --classic

# Configure Git (UPDATE THESE WITH YOUR DETAILS)
git config --global user.email "your-email@example.com"
git config --global user.name "Your Name"

# Generate SSH key for GitHub
ssh-keygen -t ed25519 -C "your-email@example.com" -f ~/.ssh/id_ed25519 -N ""

# Display public key - ADD THIS TO GITHUB
echo "=========================================="
echo "Copy the key below and add to GitHub:"
echo "https://github.com/settings/keys"
echo "=========================================="
cat ~/.ssh/id_ed25519.pub
echo "=========================================="
echo "Press Enter after adding the key to GitHub..."
read

# Test SSH connection
ssh -T git@github.com

# Clone repository
git clone git@github.com:The-School-of-AI/LLM.git
cd LLM

# Switch to working branch
git switch p10/feat/seed_architecture_perf_token_s_impr

# Install profiling tools
cd experiments/10_slm_training/llm_architecture/profiling
source install_profiling_tools.sh

echo "Setup complete!"
