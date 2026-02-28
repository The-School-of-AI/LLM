#!/bin/bash
# Setup script for OLMES L-Eval benchmarking
# Uses UV for fast dependency management

set -e

echo "🚀 Setting up OLMES L-Eval Benchmark Environment"
echo "================================================"

# Check UV is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV is not installed. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "✅ UV is already installed: $(uv --version)"
fi

# Create virtual environment with UV
echo ""
echo "📦 Creating virtual environment with UV..."
uv venv

# Activate virtual environment
echo ""
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo ""
echo "📥 Installing dependencies with UV (this will be fast!)..."
uv pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the validation:"
echo "  python run_validation.py"
echo ""
echo "To run with custom model:"
echo "  python run_validation.py --model Qwen/Qwen2.5-1.5B-Instruct --device mps"
