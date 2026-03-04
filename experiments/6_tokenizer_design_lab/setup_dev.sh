#!/bin/bash

echo "=========================================================="
echo "      Tokenizer Design Lab - Developer Setup"
echo "=========================================================="
echo ""
echo "This script installs the required pre-commit hooks to ensure "
echo "code formatting (black, isort, etc.) is applied locally "
echo "before any commits are made."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: python or python3 was not found. Please install Python."
    exit 1
fi

# The pre-commit config is located at the repository root. Let's find it.
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)

if [ -z "$REPO_ROOT" ]; then
    echo "ERROR: Not in a Git repository. Please clone the repo first."
    exit 1
fi

echo "[1/3] Installing pre-commit via pip..."
pip install --upgrade pre-commit --quiet

echo "[2/3] Installing Git pre-commit hooks..."
cd "$REPO_ROOT" || exit
pre-commit install

echo ""
echo "[3/3] Running hooks on all files once to ensure baseline compliance..."
pre-commit run --all-files

echo ""
echo "=========================================================="
echo "    Setup Complete! You are ready to develop and commit."
echo "    The hooks will now run automatically on every 'git commit'."
echo "=========================================================="
