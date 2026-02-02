#!/bin/bash

################################################################################
# Install NVIDIA Nsight Profiling Tools
################################################################################
# This script installs the latest versions of Nsight Systems and Nsight Compute
# on Ubuntu/Debian systems.
#
# Usage:
#   chmod +x install_profiling_tools.sh
#   ./install_profiling_tools.sh
################################################################################

# Note: Errors will stop script execution due to 'set -e', but won't exit your shell

set -e  # Stop script on error, but don't exit shell

echo "============================================================"
echo "Installing NVIDIA Nsight Profiling Tools"
echo "============================================================"

# Update package list
echo "Updating package lists..."
sudo apt-get update || {
    echo "❌ Error: Failed to update package lists"
    return 1 2>/dev/null || true
}

# Install Nsight Systems
echo ""
echo "Installing Nsight Systems 2025.5.2..."
sudo apt-get install -y nsight-systems-2025.5.2 || {
    echo "❌ Error: Failed to install Nsight Systems"
    echo "   Try manually: sudo apt-get install nsight-systems-2025.5.2"
    return 1 2>/dev/null || true
}

# Install Nsight Compute
echo ""
echo "Installing Nsight Compute 2025.4.1..."
sudo apt-get install -y nsight-compute-2025.4.1 || {
    echo "❌ Error: Failed to install Nsight Compute"
    echo "   Try manually: sudo apt-get install nsight-compute-2025.4.1"
    return 1 2>/dev/null || true
}

# Verify installations
echo ""
echo "============================================================"
echo "Verification"
echo "============================================================"

echo ""
echo "Nsight Systems:"
nsys --version || {
    echo "❌ Error: nsys command not found after installation"
    return 1 2>/dev/null || true
}

echo ""
echo "Nsight Compute:"
# ncu often installs to /opt/nvidia/nsight-compute/, need to check PATH
if ! command -v ncu &> /dev/null; then
    echo "⚠️  ncu not found in PATH, searching for installation..."
    
    # Common installation locations
    NCU_PATHS=(
        "/opt/nvidia/nsight-compute/2025.4.1/ncu"
        "/usr/local/cuda/bin/ncu"
        "/opt/nvidia/nsight-compute/*/ncu"
    )
    
    NCU_FOUND=""
    for path_pattern in "${NCU_PATHS[@]}"; do
        # Use glob expansion for wildcard paths
        for ncu_path in $path_pattern; do
            if [ -f "$ncu_path" ] && [ -x "$ncu_path" ]; then
                NCU_FOUND="$ncu_path"
                break 2
            fi
        done
    done
    
    if [ -n "$NCU_FOUND" ]; then
        echo "✅ Found ncu at: $NCU_FOUND"
        "$NCU_FOUND" --version
        
        # Extract directory
        NCU_DIR=$(dirname "$NCU_FOUND")
        
        echo ""
        echo "============================================================"
        echo "⚠️  ACTION REQUIRED: Add ncu to PATH"
        echo "============================================================"
        echo "Run the following command to add ncu to your PATH:"
        echo ""
        echo "  echo 'export PATH=\$PATH:$NCU_DIR' >> ~/.bashrc"
        echo "  source ~/.bashrc"
        echo ""
        echo "Or for current session only:"
        echo "  export PATH=\$PATH:$NCU_DIR"
        echo "============================================================"
    else
        echo "❌ Error: ncu command not found after installation"
        echo "   Checked common locations but couldn't find ncu"
        echo "   Try manually: which ncu"
        return 1 2>/dev/null || true
    fi
else
    ncu --version
fi

echo ""
echo "============================================================"
echo "✅ Installation complete!"
echo "============================================================"
echo ""
echo "You can now run profiling scripts:"
echo "  cd profiling/nsightSystemProfile && ./profile_1b_timeline.sh"
echo "  cd profiling/kernelProfile && ./profile_focused_kernels.sh"
echo "============================================================"
